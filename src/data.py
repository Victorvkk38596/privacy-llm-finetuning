"""Deterministic, group-aware, class-balanced partitioning of GLUE SST-2 `train`.

Implements ARCHITECTURE.md Section 1 (data contract) / methodology_audit_addendum Item 1:
  1. Normalize text (NFKC, lowercase, strip punctuation, collapse whitespace); drop empty keys.
  2. Drop whole label-conflict groups; keep one representative per normalized key.
  3. Link every item (>= MIN_WORDS words) to its longest containing item -> family forest.
  4. Policy F1: at most ONE item per family goes into any split.
  5. Seeded Stratified Hash Partitioning across 6 strata cells {label 0, label 1} x {Short, Medium, Long}.
  6. Exact 50/50 balance, exact split sizes, SMD < 0.10, and KS p-value > 0.01 between D_A and D_out.
"""
import collections
import hashlib
import json
import os
import re
import unicodedata

from datasets import load_dataset
import numpy as np
from scipy import stats

MIN_WORDS = 4
SPLIT_SIZES = {"A": 1024, "B": 1024, "out": 1024, "pop": 1024, "tune": 2048}


def _h(*parts) -> int:
    """Stable 64-bit hash of the given parts."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def norm_key(text: str) -> str:
    """Normalize text per methodology_audit_addendum §1.1-§1.2:
    NFKC normalize, lowercase, strip punctuation, collapse whitespace.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Alias key(x)
key = norm_key


def build_pool(sentences, labels):
    """Dedup + conflict-drop per methodology_audit_addendum §1.2.
    
    1. Group raw records by key(x). Drop empty keys.
    2. If any key maps to multiple distinct labels, drop the entire conflict group.
    3. If consistent, retain one representative (the lexicographically smallest raw string).
    Returns list of item dicts sorted deterministically by key.
    """
    groups = collections.defaultdict(list)
    for idx, (s, y) in enumerate(zip(sentences, labels)):
        k = norm_key(s)
        if k:
            groups[k].append((idx, s, int(y)))
    items = []
    for k, rows in groups.items():
        if len({y for _, _, y in rows}) > 1:
            continue  # label conflict: drop the whole group
        idx, raw, y = min(rows, key=lambda r: (r[1], r[0]))  # lexicographically smallest raw string
        items.append({"idx": idx, "key": k, "raw": raw, "label": y, "nwords": len(k.split())})
    items.sort(key=lambda it: it["key"])  # deterministic order
    return items


def assign_families(items, min_words=MIN_WORDS):
    """Single-parent containment forest per methodology_audit_addendum §1.3-§1.4.
    
    For all items with word count >= min_words, link each item to its longest
    containing parent item (word-boundary match, ties broken by lexicographic order).
    Uses sub-n-gram prefix indexing so construction completes in well under 15 seconds.
    Items with < min_words are singleton families.
    The tree root is family_id (and stored as both 'family_id' and 'family').
    """
    words = {it["key"]: it["key"].split() for it in items}
    index = collections.defaultdict(list)
    for it in items:
        w = words[it["key"]]
        for i in range(len(w) - min_words + 1):
            index[tuple(w[i:i + min_words])].append(it["key"])
    parent = {}
    for it in items:
        k = it["key"]
        w = words[k]
        if len(w) < min_words:
            continue
        padded = " " + k + " "
        best = None
        for c in index[tuple(w[:min_words])]:
            if c != k and len(words[c]) > len(w) and padded in " " + c + " ":
                if best is None or (len(words[c]), c) > (len(words[best]), best):
                    best = c
        if best is not None:
            parent[k] = best
    root_cache = {}

    def root(k):
        path = []
        while k in parent and k not in root_cache:
            path.append(k)
            k = parent[k]
        r = root_cache.get(k, k)
        for p in path:
            root_cache[p] = r
        return r

    for it in items:
        r = root(it["key"])
        it["family_id"] = r
        it["family"] = r
    return items


class PartitionDict(dict):
    """Dictionary for partition splits supporting alias lookup (e.g. 'D_A' <-> 'A')."""

    def _resolve_key(self, key):
        if dict.__contains__(self, key):
            return key
        s_key = str(key)
        if s_key.startswith("D_") and dict.__contains__(self, s_key[2:]):
            return s_key[2:]
        alt = f"D_{s_key}"
        if dict.__contains__(self, alt):
            return alt
        return key

    def __getitem__(self, key):
        resolved = self._resolve_key(key)
        return super().__getitem__(resolved)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        resolved = self._resolve_key(key)
        return dict.__contains__(self, resolved)


def make_partition(items, seed: int, sizes=None):
    """Deterministic stratified, group-aware partition per methodology_audit_addendum §1.4.

    1. Policy F1: Selects at most one representative item per family component:
       rep = min(fam_items, key=lambda it: sha256(f"{seed}_rep_{it['key']}").hexdigest())
    2. Stratification Grid: 6 strata cells {label 0, label 1} x {Short, Medium, Long}
       based on word-count terciles across all eligible F1 candidates.
    3. Deterministic Allocation:
       For each stratum, items are sorted by sha256(f"{seed}_{item['family_id']}").hexdigest()
       and allocated to splits according to target sizes, ensuring:
       - Exact split sizes (1024 for A, B, out, pop; 2048 for tune by default).
       - Exact 50/50 class balance.
       - Equal word-count tercile proportions across all splits.
    """
    raw_sizes = dict(sizes or SPLIT_SIZES)
    split_order = []
    canonical_sizes = {}
    for k, v in raw_sizes.items():
        c_name = k[2:] if k.startswith("D_") else k
        split_order.append(c_name)
        canonical_sizes[c_name] = v

    # 1. Policy F1: one representative item per family component
    family_groups = collections.defaultdict(list)
    for it in items:
        family_groups[it["family_id"]].append(it)

    f1_candidates = []
    for fam_id, fam_items in sorted(family_groups.items()):
        rep = min(
            fam_items,
            key=lambda it: hashlib.sha256(f"{seed}_rep_{it['key']}".encode("utf-8")).hexdigest(),
        )
        f1_candidates.append(rep)

    # 2. Stratification Grid: word-count terciles (Short, Medium, Long)
    nwords_all = [it["nwords"] for it in f1_candidates]
    q1, q2 = np.percentile(nwords_all, [100.0 / 3.0, 200.0 / 3.0])

    def get_tercile(w: int) -> str:
        if w <= q1:
            return "Short"
        elif w <= q2:
            return "Medium"
        else:
            return "Long"

    strata = collections.defaultdict(list)
    for it in f1_candidates:
        t = get_tercile(it["nwords"])
        it["tercile"] = t
        strata[(it["label"], t)].append(it)

    # Deterministic sorting within each stratum
    for k in strata:
        strata[k].sort(
            key=lambda it: hashlib.sha256(f"{seed}_{it['family_id']}".encode("utf-8")).hexdigest()
        )

    # 3. Quota calculation per split and stratum
    def get_tercile_quotas(half_size: int):
        if half_size % 512 == 0:
            mult = half_size // 512
            return {"Short": 171 * mult, "Medium": 171 * mult, "Long": 170 * mult}
        q = half_size // 3
        rem = half_size % 3
        return {
            "Short": q + (1 if rem >= 1 else 0),
            "Medium": q + (1 if rem >= 2 else 0),
            "Long": q,
        }

    quotas = {}
    for s_name in split_order:
        sz = canonical_sizes[s_name]
        if sz % 2 != 0:
            raise ValueError(f"Split {s_name} size {sz} must be even for exact 50/50 balance")
        quotas[s_name] = get_tercile_quotas(sz // 2)

    # Allocate items to each split
    part = PartitionDict({s_name: [] for s_name in split_order})
    for label in (0, 1):
        for t in ("Short", "Medium", "Long"):
            cell_items = strata[(label, t)]
            offset = 0
            for s_name in split_order:
                q = quotas[s_name][t]
                if offset + q > len(cell_items):
                    raise ValueError(
                        f"Stratum ({label}, {t}) exhausted: needed {offset + q}, had {len(cell_items)}"
                    )
                selected = cell_items[offset : offset + q]
                part[s_name].extend(selected)
                offset += q

    for s_name in split_order:
        part[s_name].sort(key=lambda it: it["key"])

    check_partition(part, canonical_sizes)
    return part


def _smd(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float(abs(a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def check_partition(part, sizes=None, smd_threshold: float = 0.10, ks_alpha: float = 0.01):
    """Assertions from methodology_audit_addendum §1.4. Raises AssertionError on violation; returns diagnostics."""
    sizes = sizes or {n: len(v) for n, v in part.items()}
    canonical_names = list(part.keys())

    # 1. Exact sizes and 50/50 balance
    for n in canonical_names:
        v = part[n]
        expected_size = sizes.get(n) or sizes.get(f"D_{n}") or sizes.get(n[2:] if n.startswith("D_") else n)
        if expected_size is not None:
            assert len(v) == expected_size, f"Split {n}: size {len(v)} != {expected_size}"
        assert sum(it["label"] for it in v) * 2 == len(v), (
            f"Split {n}: labels not exactly 50/50 ({sum(it['label'] for it in v)}/{len(v)})"
        )

    # 2. Pairwise disjointness on key and family_id
    for i, a in enumerate(canonical_names):
        for b in canonical_names[i + 1:]:
            keys_a = {it["key"] for it in part[a]}
            keys_b = {it["key"] for it in part[b]}
            shared_keys = keys_a & keys_b
            assert not shared_keys, f"Splits {a} and {b} share {len(shared_keys)} keys"

            fams_a = {it["family_id"] for it in part[a]}
            fams_b = {it["family_id"] for it in part[b]}
            shared_fams = fams_a & fams_b
            assert not shared_fams, f"Splits {a} and {b} share {len(shared_fams)} family IDs"

    # 3. SMD and KS-test checks
    diag = {}
    out_items = part.get("out") or part.get("D_out")
    if out_items is not None:
        words_out = [it["nwords"] for it in out_items]
        for m in ("A", "B"):
            m_items = part.get(m) or part.get(f"D_{m}")
            if m_items is not None:
                words_m = [it["nwords"] for it in m_items]
                d = _smd(words_m, words_out)
                ks_res = stats.ks_2samp(words_m, words_out)
                diag[f"smd_words_{m}_out"] = d
                diag[f"ks_pval_{m}_out"] = float(ks_res.pvalue)
                assert d < smd_threshold, f"Word-count SMD {m} vs out = {d:.4f} >= {smd_threshold}"
                assert ks_res.pvalue > ks_alpha, (
                    f"Word-count KS p-value {m} vs out = {ks_res.pvalue:.4f} <= {ks_alpha}"
                )

    return diag


def export_manifest(part, seed: int, out_dir: str = "manifests") -> str:
    """Exports partition manifest with item keys and SHA-256 fingerprints to out_dir/manifest_seed{seed}.json."""
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, f"manifest_seed{seed}.json")

    splits_data = {}
    for name in ["A", "B", "out", "pop", "tune"]:
        if name in part:
            items = part[name]
            keys = [it["key"] for it in items]
            h = hashlib.sha256()
            for k in sorted(keys):
                h.update(k.encode("utf-8"))
            fp = h.hexdigest()
            split_info = {
                "fingerprint": fp,
                "size": len(keys),
                "keys": keys,
            }
            splits_data[name] = split_info
            splits_data[f"D_{name}"] = split_info

    manifest = {
        "seed": seed,
        "partition_hash": partition_hash(part),
        "splits": {k: v for k, v in splits_data.items() if not k.startswith("D_")},
        **splits_data,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def partition_hash(part) -> str:
    h = hashlib.sha256()
    for n in sorted(part.keys()):
        h.update(n.encode())
        for it in part[n]:
            h.update(it["key"].encode())
    return h.hexdigest()[:16]


def partition_indices(part):
    """{split: [original train row indices]} (for datasets.Dataset.select)."""
    return PartitionDict({n: [it["idx"] for it in v] for n, v in part.items()})


_PARTITIONS_CACHE = {}


def get_partitions(seed: int = 42, sizes: dict = None) -> dict:
    """Returns partitions dict for the given seed with keys:
    'A', 'B', 'out', 'pop', 'tune', 'D_A', 'D_B', 'D_out', 'D_pop', 'D_tune', and 'validation'.
    Each split contains items with 'raw' and 'label' (plus 'key', 'idx', etc.).
    Standard SST-2 validation (872 items) is loaded under 'validation' and 'val'.
    """
    cache_key = (seed, tuple(sorted((sizes or {}).items())))
    if cache_key in _PARTITIONS_CACHE:
        return _PARTITIONS_CACHE[cache_key]

    raw_train = load_dataset("nyu-mll/glue", "sst2", split="train")
    clean_items = build_pool(raw_train["sentence"], raw_train["label"])
    forest_items = assign_families(clean_items)
    part = make_partition(forest_items, seed=seed, sizes=sizes)

    # Load standard SST-2 validation
    raw_val = load_dataset("nyu-mll/glue", "sst2", split="validation")
    val_items = [{"raw": ex["sentence"], "label": int(ex["label"]), "idx": i} for i, ex in enumerate(raw_val)]
    part["validation"] = val_items
    part["val"] = val_items

    _PARTITIONS_CACHE[cache_key] = part
    return part


if __name__ == "__main__":
    import time
    from collections import Counter
    from datasets import load_dataset

    print("Running Task 1A & 1B Verification on SST-2 train...")
    t0 = time.time()
    raw_dataset = load_dataset("nyu-mll/glue", "sst2", split="train")
    t_download = time.time()
    print(f"Loaded SST-2 train: {len(raw_dataset)} examples ({t_download - t0:.2f}s)")

    # Step 1 & 2: Text Normalization and Conflict Drop
    t_step12_start = time.time()
    clean_items = build_pool(raw_dataset["sentence"], raw_dataset["label"])
    t_step12_end = time.time()
    print(f"Deduplicated and conflict-free items: {len(clean_items)} ({t_step12_end - t_step12_start:.2f}s)")

    # Step 3: High-Performance Parent-Phrase Containment Forest
    t_forest_start = time.time()
    forest_items = assign_families(clean_items)
    t_forest_end = time.time()
    print(f"Parent-phrase forest computed ({t_forest_end - t_forest_start:.2f}s)")

    forest_build_time = t_forest_end - t_forest_start

    # Family components count and sizes
    family_counts = Counter(it["family_id"] for it in forest_items)
    num_families = len(family_counts)
    largest_family_size = max(family_counts.values()) if family_counts else 0

    print("\n" + "=" * 45)
    print("TASK 1A VERIFICATION SUMMARY")
    print("=" * 45)
    print(f"Total unique keys (surviving items): {len(clean_items)}")
    print(f"Total family components:            {num_families}")
    print(f"Largest family size:                {largest_family_size}")
    print(f"Forest construction time:           {forest_build_time:.2f}s (Invariant: < 15s)")
    print("=" * 45 + "\n")

    # Assertions per Task 1A:
    assert 58000 <= len(clean_items) <= 59000
    assert len(clean_items) == 58463
    assert 24500 <= num_families <= 25500
    assert num_families == 24976
    assert largest_family_size == 25
    assert forest_build_time < 15.0

    print("[SUCCESS] All Task 1A assertions passed!\n")

    # Task 1B: Seeded Stratified Hash Partitioning
    print("Running Task 1B Partitioning for seed=42...")
    t_part_start = time.time()
    part = make_partition(forest_items, seed=42)
    t_part_end = time.time()
    print(f"Partition completed in {t_part_end - t_part_start:.2f}s")

    D_A = part["D_A"]
    D_B = part["D_B"]
    D_out = part["D_out"]
    D_pop = part["D_pop"]
    D_tune = part["D_tune"]

    # 1. Pairwise Disjointness
    all_splits = [("D_A", D_A), ("D_B", D_B), ("D_out", D_out), ("D_pop", D_pop), ("D_tune", D_tune)]
    for i, (name1, s1) in enumerate(all_splits):
        for name2, s2 in all_splits[i + 1:]:
            keys1 = set(it["key"] for it in s1)
            keys2 = set(it["key"] for it in s2)
            assert len(keys1 & keys2) == 0, f"Disjointness failed: {name1} and {name2} share keys!"

            fams1 = set(it["family_id"] for it in s1)
            fams2 = set(it["family_id"] for it in s2)
            assert len(fams1 & fams2) == 0, f"Disjointness failed: {name1} and {name2} share family_ids!"

    # 2. Exact Sizes
    assert len(D_A) == 1024, f"D_A size {len(D_A)} != 1024"
    assert len(D_B) == 1024, f"D_B size {len(D_B)} != 1024"
    assert len(D_out) == 1024, f"D_out size {len(D_out)} != 1024"
    assert len(D_pop) == 1024, f"D_pop size {len(D_pop)} != 1024"
    assert len(D_tune) == 2048, f"D_tune size {len(D_tune)} != 2048"

    # 3. Exact Class Balance (50.0% label 1, 50.0% label 0)
    for name, s in all_splits:
        pos = sum(it["label"] == 1 for it in s)
        neg = sum(it["label"] == 0 for it in s)
        assert pos == neg == len(s) // 2, f"{name}: imbalanced classes ({pos} vs {neg})"

    # 4. Length Balance (SMD)
    words_A = [it["nwords"] for it in D_A]
    words_out = [it["nwords"] for it in D_out]
    smd_A_out = _smd(words_A, words_out)
    assert smd_A_out < 0.10, f"SMD(D_A, D_out) {smd_A_out:.4f} >= 0.10"

    # 5. KS-Test
    ks_res = stats.ks_2samp(words_A, words_out)
    assert ks_res.pvalue > 0.01, f"KS test p-value {ks_res.pvalue:.4f} <= 0.01"

    # 6. Manifest Export
    manifest_path = export_manifest(part, seed=42, out_dir="manifests")
    assert os.path.exists(manifest_path), f"Manifest file {manifest_path} was not created"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert "D_A" in manifest_data and "fingerprint" in manifest_data["D_A"]
    assert len(manifest_data["D_A"]["keys"]) == 1024

    t_total = time.time() - t0

    print("\n" + "=" * 45)
    print("TASK 1B VERIFICATION SUMMARY (seed=42)")
    print("=" * 45)
    print(f"D_A size:              {len(D_A)} (Pos: {sum(it['label'] for it in D_A)})")
    print(f"D_B size:              {len(D_B)} (Pos: {sum(it['label'] for it in D_B)})")
    print(f"D_out size:            {len(D_out)} (Pos: {sum(it['label'] for it in D_out)})")
    print(f"D_pop size:            {len(D_pop)} (Pos: {sum(it['label'] for it in D_pop)})")
    print(f"D_tune size:           {len(D_tune)} (Pos: {sum(it['label'] for it in D_tune)})")
    print(f"Pairwise disjoint:     PASSED (zero shared keys & family IDs)")
    print(f"Word Count SMD(A,out): {smd_A_out:.4f} (Required: < 0.10)")
    print(f"KS-Test p-value:       {ks_res.pvalue:.4f} (Required: > 0.01)")
    print(f"Manifest written to:   {manifest_path}")
    print(f"Partition hash:        {manifest_data['partition_hash']}")
    print(f"Total pipeline time:   {t_total:.2f}s")
    print("=" * 45 + "\n")

    print("[SUCCESS] All Task 1A & 1B assertions passed successfully!")

