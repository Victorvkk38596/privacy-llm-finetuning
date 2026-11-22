"""Loading per-model records and the pre-registered collapse / gate rules (pure Python)."""
import glob
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_spec import CONFIGS, GATE_CONFIGS, exp_name  # noqa: E402

# Pre-registered rule (fixed before any corrected run):
# a model "stays near the majority-class baseline" (collapsed) if its SST-2 validation accuracy
# is below 55% OR it predicts the positive class for <5% or >95% of validation examples.
COLLAPSE_ACC = 55.0
COLLAPSE_POS_RATE = (0.05, 0.95)


def load_records(records_dir: str) -> List[dict]:
    recs = []
    for p in sorted(glob.glob(os.path.join(records_dir, "*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        r["_path"] = p
        recs.append(r)
    return recs


def model_collapsed(rec: dict) -> bool:
    m = rec.get("metrics") or {}
    acc = m.get("val_accuracy")
    pos = m.get("val_pred_pos_rate")
    if acc is None:
        return True
    if acc < COLLAPSE_ACC:
        return True
    if pos is not None and not (COLLAPSE_POS_RATE[0] <= pos <= COLLAPSE_POS_RATE[1]):
        return True
    return False


def index_records(recs: List[dict]) -> Dict[tuple, dict]:
    """(exp_name, seed, role) -> record (latest file wins)."""
    out = {}
    for r in recs:
        out[(r.get("exp_name"), int(r.get("seed")), r.get("role"))] = r
    return out


def pair_status(idx: Dict[tuple, dict], name: str, seed: int) -> str:
    """'ok' if both swap-pair models have status ok; 'failed'/'privacy_check_failed'/'missing' otherwise."""
    st = []
    for role in ("M_A", "M_B"):
        r = idx.get((name, seed, role))
        st.append("missing" if r is None else r.get("status", "missing"))
    if all(s == "ok" for s in st):
        return "ok"
    for s in ("privacy_check_failed", "failed", "missing"):
        if s in st:
            return s
    return st[0]


def gate_decision(records_dir: str, warmup: str = "none", seed: int = 42) -> dict:
    """Step-1 decision: do early-3, late-3 and all-12 at eps=5 learn without warmup (seed 42)?"""
    idx = index_records(load_records(records_dir))
    rows, all_present, any_collapsed = [], True, False
    for c in GATE_CONFIGS:
        name = exp_name(c, warmup)
        st = pair_status(idx, name, seed)
        entry = {"config": c, "exp_name": name, "status": st}
        if st == "ok":
            ms = [idx[(name, seed, r)] for r in ("M_A", "M_B")]
            entry["val_accuracy"] = [m["metrics"]["val_accuracy"] for m in ms]
            entry["val_pred_pos_rate"] = [m["metrics"].get("val_pred_pos_rate") for m in ms]
            entry["collapsed"] = [model_collapsed(m) for m in ms]
            any_collapsed |= any(entry["collapsed"])
        else:
            all_present = False
        rows.append(entry)
    if not all_present:
        verdict = "incomplete"
    elif any_collapsed:
        verdict = "collapse: rerun gate with --warmup public_imdb and use it for all DP configs"
    else:
        verdict = f"learns: use --warmup {warmup} for all DP configs"
    return {"warmup": warmup, "seed": seed, "rule": {"acc_below": COLLAPSE_ACC, "pos_rate_outside": COLLAPSE_POS_RATE,
            "config_fails_if": "either swap-pair model collapses"}, "configs": rows, "verdict": verdict}
