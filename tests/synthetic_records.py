"""SYNTHETIC per-model records for tests. Shaped like the records run_experiment.py writes.
Only ever written to temporary directories; never to results/."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
from grid_spec import CONFIGS, CLEAN_BASE, DP_BASE, exp_name, expected_trainable  # noqa: E402

SEED42_MANIFEST_HASH = "7afbcaf0706ba21a"   # manifests/manifest_seed42.json
HASH = {42: SEED42_MANIFEST_HASH, 43: "hash43", 44: "hash44"}


def make_record(cfg, seed, role, acc=80.0, auc=0.51, tpr=0.012, pos=0.5, warmup="none", **over):
    layers, eps, comps, _ = CONFIGS[cfg]
    dp = eps is not None
    steps = (DP_BASE if dp else CLEAN_BASE)["max_steps"]
    r = {
        "schema": "grid-record-v1", "status": "ok", "exp_name": exp_name(cfg, warmup), "config_name": cfg,
        "comparisons": comps, "seed": seed, "role": role, "partition_hash": HASH[seed],
        "train_layers_spec": layers, "trainable_params": expected_trainable(layers),
        "training": {"use_dp": dp, "max_steps": steps, "steps_completed": steps,
                     "head_warmup": {"source": warmup if dp else "none"}},
        "privacy": {"target_epsilon": eps, "target_delta": 1e-5 if dp else None,
                    "accounted_epsilon": (eps - 0.006) if dp else None,
                    "accounting_ledger": ({"ordering_ok": True, "steps_match_ok": True,
                                           "accountant_total_steps": steps} if dp else
                                          {"ordering_ok": True, "steps_match_ok": True})},
        "metrics": {"val_accuracy": acc, "val_pred_pos_rate": pos, "rmia_auc": auc, "rmia_tpr_at_1fpr": tpr},
        "wall_time_sec": 600.0,
    }
    for k, v in over.items():          # dotted keys, e.g. **{"privacy.accounted_epsilon": 6.0}
        d = r
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = v
    return r


def failed_record(cfg, seed, role, warmup="none"):
    return {"schema": "grid-record-v1", "status": "failed", "exp_name": exp_name(cfg, warmup), "config_name": cfg,
            "seed": seed, "role": role, "error": "synthetic failure\ntraceback...",
            "training": {"use_dp": CONFIGS[cfg][1] is not None}, "privacy": {}}


def write_records(d, recs):
    os.makedirs(d, exist_ok=True)
    for r in recs:
        with open(os.path.join(d, f"{r['exp_name']}__seed{r['seed']}__{r['role']}.json"), "w") as f:
            json.dump(r, f)
