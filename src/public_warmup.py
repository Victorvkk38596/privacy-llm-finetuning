"""Public classifier-head warmup data.

Only used when a config sets `head_warmup_source: public_imdb`.

Source: `stanfordnlp/imdb`, split `train` (25,000 movie reviews from IMDb, Maas et al. 2011),
using IMDb's own review-level labels (0 = negative, 1 = positive; same convention as SST-2).
This data is public and is not derived from SST-2 / Rotten Tomatoes, so a non-private
warmup on it does not touch the private training split and needs no privacy budget,
under the assumption that IMDb and the pretrained roberta-base weights are public and
do not contain the private SST-2 records.

The sample is a seeded, class-balanced draw that depends only on `seed` and `n_examples`,
so every DP configuration within a seed sees the identical warmup data.
"""
import hashlib
import re
from typing import Any, Dict, List

PUBLIC_WARMUP_SOURCES = {
    "public_imdb": {"hf_name": "stanfordnlp/imdb", "split": "train", "text_field": "text", "label_field": "label"},
}


def _hash_order(seed: int, idx: int) -> str:
    return hashlib.sha256(f"public_warmup_{seed}_{idx}".encode("utf-8")).hexdigest()


def load_public_warmup_items(source: str, seed: int, n_examples: int = 4096) -> List[Dict[str, Any]]:
    if source not in PUBLIC_WARMUP_SOURCES:
        raise ValueError(f"Unknown public warmup source '{source}'. Options: {sorted(PUBLIC_WARMUP_SOURCES)}")
    if n_examples % 2:
        raise ValueError("n_examples must be even (class-balanced sample)")
    spec = PUBLIC_WARMUP_SOURCES[source]
    from datasets import load_dataset  # local import: keeps module importable without datasets

    ds = load_dataset(spec["hf_name"], split=spec["split"])
    texts, labels = ds[spec["text_field"]], ds[spec["label_field"]]
    by_class = {0: [], 1: []}
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)
    per_class = n_examples // 2
    chosen = []
    for y in (0, 1):
        idxs = sorted(by_class[y], key=lambda i: _hash_order(seed, i))[:per_class]
        if len(idxs) < per_class:
            raise RuntimeError(f"{source}: only {len(idxs)} examples of class {y}")
        chosen.extend(idxs)
    chosen.sort(key=lambda i: _hash_order(seed, i))
    items = []
    for i in chosen:
        raw = re.sub(r"<br\s*/?>", " ", texts[i])
        raw = re.sub(r"\s+", " ", raw).strip()
        items.append({"raw": raw, "label": int(labels[i]), "idx": int(i)})
    return items


def warmup_fingerprint(items: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(str(it["idx"]).encode())
    return h.hexdigest()[:16]
