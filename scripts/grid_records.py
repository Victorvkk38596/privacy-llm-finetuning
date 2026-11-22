"""Loading per-model records and the pre-registered collapse / gate rules (pure Python)."""
import glob
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from grid_spec import CONFIGS, GATE_CONFIGS, SEEDS, CLEAN_BASE, DP_BASE, exp_name  # noqa: E402
from src.record_validation import evaluate_pairs  # noqa: E402

MANIFESTS_DIR = os.path.join(ROOT, "manifests")

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


def expected_steps(warmup: str) -> Dict[str, int]:
    """Grid spec step counts per experiment name: DP 300, clean 1000."""
    return {exp_name(c, warmup): (CLEAN_BASE if CONFIGS[c][1] is None else DP_BASE)["max_steps"] for c in CONFIGS}


def grid_pair_states(idx: Dict[tuple, dict], warmup: str, seeds=SEEDS, configs=None,
                     manifests_dir: str = MANIFESTS_DIR) -> Dict[tuple, dict]:
    """Acceptance for every grid swap pair of one warmup variant: {(config_name, seed): state}.

    Same rule as run_experiment.py and aggregate_grid.py (src/record_validation.py), plus the grid
    step counts and the same-seed cross-configuration partition-hash check."""
    configs = list(CONFIGS) if configs is None else list(configs)
    # the cross-configuration check always looks at every config of the seed, not just the subset asked for
    wanted = [(exp_name(c, warmup), s) for c in CONFIGS for s in seeds]
    res = evaluate_pairs(idx, wanted, expected_steps(warmup), manifests_dir)
    return {(c, s): res[(exp_name(c, warmup), s)] for c in configs for s in seeds}


def pair_status(idx: Dict[tuple, dict], name: str, seed: int, expected_max_steps: int = None,
                manifests_dir: str = MANIFESTS_DIR) -> str:
    """Status of one pair under the shared rule (used for smoke records and ad-hoc checks)."""
    res = evaluate_pairs(idx, [(name, seed)], {name: expected_max_steps} if expected_max_steps else None,
                         manifests_dir)
    return res[(name, seed)]["status"]


def gate_decision(records_dir: str, warmup: str = "none", seed: int = 42) -> dict:
    """Step-1 decision: do early-3, late-3 and all-12 at eps=5 learn without warmup (seed 42)?"""
    idx = index_records(load_records(records_dir))
    states = grid_pair_states(idx, warmup, seeds=[seed])
    rows, all_present, any_collapsed = [], True, False
    for c in GATE_CONFIGS:
        name = exp_name(c, warmup)
        st = states[(c, seed)]["status"]
        entry = {"config": c, "exp_name": name, "status": st, "reasons": states[(c, seed)]["reasons"]}
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
