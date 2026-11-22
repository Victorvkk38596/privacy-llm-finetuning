"""Single acceptance rule for per-model records and swap pairs (pure Python, no torch).

Used by run_experiment.py (status written into each record), scripts/run_core_grid.py (resume:
which pairs still need running) and scripts/aggregate_grid.py (which pairs enter statistics).
Acceptance is always re-derived from record contents; the stored `status` field alone never makes
a pair valid, so a stale or hand-edited "ok" cannot slip through.

Pair statuses:
  ok                  both models valid, matching nonempty partition hash
  pending             no record for either model
  incomplete          exactly one model has a record
  failed              a model's record has status "failed" (crash, OOM, worker exit, exception)
  invalid             records exist but violate a rule (reasons listed)
  partition_mismatch  pair is otherwise ok but its hash differs from the seed's reference hash
                      (other configurations / manifest), so it cannot be compared within the seed
"""
import json
import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

EPS_TOL = 1e-6  # absolute tolerance on accounted <= target epsilon
ROLES = ("M_A", "M_B")
NOT_OK = ("pending", "incomplete", "failed", "invalid", "partition_mismatch")


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def validate_model(rec: dict, expected_max_steps: Optional[int] = None) -> List[str]:
    """Reasons this model record is unacceptable (empty list = acceptable)."""
    if rec is None:
        return ["missing record"]
    if rec.get("status") == "failed":
        first = (str(rec.get("error") or "").strip().splitlines() or [""])[0][:200]
        return [f"run failed: {first}" if first else "run failed"]
    reasons = []
    tr = rec.get("training") or {}
    pv = rec.get("privacy") or {}
    m = rec.get("metrics") or {}
    if not rec.get("partition_hash"):
        reasons.append("missing partition hash")
    for k in ("val_accuracy", "rmia_auc", "rmia_tpr_at_1fpr"):
        if not _finite(m.get(k)):
            reasons.append(f"metric {k} missing or non-finite")
    steps, max_steps = tr.get("steps_completed"), tr.get("max_steps")
    if steps is None or max_steps is None or int(steps) != int(max_steps):
        reasons.append(f"steps_completed={steps} != max_steps={max_steps}")
    if expected_max_steps is not None and max_steps is not None and int(max_steps) != int(expected_max_steps):
        reasons.append(f"max_steps={max_steps} != grid spec {expected_max_steps}")
    if tr.get("use_dp"):
        target, acc = pv.get("target_epsilon"), pv.get("accounted_epsilon")
        if not _finite(target):
            reasons.append("target epsilon missing or non-finite")
        if not _finite(acc):
            reasons.append("accounted epsilon missing or non-finite")
        elif _finite(target) and float(acc) > float(target) + EPS_TOL:
            reasons.append(f"accounted epsilon {float(acc):.4f} > target {float(target):g}")
        if not _finite(pv.get("target_delta")):
            reasons.append("target delta missing or non-finite")
        led = pv.get("accounting_ledger") or {}
        if led.get("ordering_ok") is not True:
            reasons.append("accounting-order check failed or missing")
        if led.get("steps_match_ok") is not True:
            reasons.append("accountant steps != logical optimizer steps (or check missing)")
        if led and steps is not None and led.get("accountant_total_steps") is not None \
                and int(led["accountant_total_steps"]) != int(steps):
            reasons.append(f"accountant_total_steps={led['accountant_total_steps']} != steps_completed={steps}")
        warm = (tr.get("head_warmup") or {}).get("source", "none")
        if warm not in ("none", "public_imdb"):
            reasons.append(f"head warmup source '{warm}' not allowed")
    return reasons


def model_status(rec: dict, expected_max_steps: Optional[int] = None) -> str:
    """Status to store in a freshly written record."""
    if rec.get("status") == "failed":
        return "failed"
    return "ok" if not validate_model(rec, expected_max_steps) else "invalid"


def validate_pair(rec_a: Optional[dict], rec_b: Optional[dict],
                  expected_max_steps: Optional[int] = None) -> Tuple[str, List[str]]:
    """(pair status, reasons). Does not include the cross-configuration hash check."""
    if rec_a is None and rec_b is None:
        return "pending", []
    if rec_a is None or rec_b is None:
        return "incomplete", [f"missing {'M_A' if rec_a is None else 'M_B'} record"]
    reasons = []
    failed = False
    for role, r in (("M_A", rec_a), ("M_B", rec_b)):
        if r.get("status") == "failed":
            failed = True
        reasons += [f"{role}: {x}" for x in validate_model(r, expected_max_steps)]
    ha, hb = rec_a.get("partition_hash"), rec_b.get("partition_hash")
    if ha and hb and ha != hb:
        reasons.append(f"partition hash differs within pair ({ha} vs {hb})")
    if failed:
        return "failed", reasons
    return ("ok", []) if not reasons else ("invalid", reasons)


def manifest_hash(manifests_dir: Optional[str], seed: int) -> Optional[str]:
    if not manifests_dir:
        return None
    p = os.path.join(manifests_dir, f"manifest_seed{seed}.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("partition_hash")


def seed_reference_hashes(pairs: Iterable[Tuple[int, str, Optional[str]]],
                          manifests_dir: Optional[str] = None) -> Dict[int, Tuple[Optional[str], str]]:
    """pairs: (seed, pair status, pair hash). Reference hash per seed and where it came from.

    The seed's manifest wins when present. Otherwise the hash shared by the most valid pairs is used;
    a tie leaves the reference undetermined (None), and every pair of that seed is then rejected."""
    by_seed: Dict[int, Counter] = {}
    seeds = set()
    for seed, status, h in pairs:
        seeds.add(seed)
        if status == "ok" and h:
            by_seed.setdefault(seed, Counter())[h] += 1
    out = {}
    for s in seeds:
        mh = manifest_hash(manifests_dir, s)
        if mh:
            out[s] = (mh, "manifest")
            continue
        c = by_seed.get(s)
        if not c:
            out[s] = (None, "no valid pairs")
            continue
        top = c.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            out[s] = (None, f"tie between hashes {[h for h, _ in top]}")
        else:
            out[s] = (top[0][0], "majority of valid pairs")
    return out


def evaluate_pairs(index: Dict[tuple, dict], wanted: Iterable[Tuple[str, int]],
                   expected_steps: Optional[Dict[str, int]] = None,
                   manifests_dir: Optional[str] = None) -> Dict[Tuple[str, int], dict]:
    """Full acceptance for a set of (exp_name, seed) pairs, including the cross-configuration check.

    index: (exp_name, seed, role) -> record. Returns {(exp_name, seed): {status, reasons, hash}}."""
    res = {}
    for name, seed in wanted:
        a, b = index.get((name, seed, "M_A")), index.get((name, seed, "M_B"))
        st, reasons = validate_pair(a, b, (expected_steps or {}).get(name))
        h = None
        if a is not None and b is not None and a.get("partition_hash") == b.get("partition_hash"):
            h = a.get("partition_hash") or None
        res[(name, seed)] = {"status": st, "reasons": reasons, "hash": h}
    refs = seed_reference_hashes([(s, v["status"], v["hash"]) for (n, s), v in res.items()], manifests_dir)
    for (name, seed), v in res.items():
        ref, src = refs.get(seed, (None, ""))
        v["reference_hash"], v["reference_source"] = ref, src
        if v["status"] == "ok" and v["hash"] != ref:
            v["status"] = "partition_mismatch"
            v["reasons"] = [f"partition hash {v['hash']} != seed {seed} reference {ref} ({src})"]
    return res
