"""Tests for scripts/aggregate_grid.py on SYNTHETIC records (written to a temp dir, never to results/).

  python tests/test_aggregate_grid.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from grid_spec import CONFIGS, SEEDS, exp_name, expected_trainable  # noqa: E402

rng = np.random.default_rng(0)
HASH = {42: "7afbcaf0706ba21a", 43: "hash43", 44: "hash44"}


def fake(cfg, seed, role, acc, auc, status="ok", pos=0.5):
    layers, eps, comps, _ = CONFIGS[cfg]
    r = {"schema": "grid-record-v1", "status": status, "exp_name": exp_name(cfg, "none"), "config_name": cfg,
         "comparisons": comps, "seed": seed, "role": role, "partition_hash": HASH[seed],
         "train_layers_spec": layers, "trainable_params": expected_trainable(layers),
         "training": {"use_dp": eps is not None, "head_warmup": {"source": "none"}},
         "privacy": {"target_epsilon": eps, "target_delta": 1e-5 if eps else None,
                     "accounted_epsilon": (eps - 0.01) if eps else None,
                     "accounting_ledger": {"ordering_ok": True, "steps_match_ok": True}},
         "metrics": {"val_accuracy": acc, "val_pred_pos_rate": pos, "rmia_auc": auc, "rmia_tpr_at_1fpr": 0.01},
         "wall_time_sec": 600.0}
    if status != "ok":
        r["metrics"] = {}
        r["error"] = "synthetic failure"
    return r


def write(d, recs):
    os.makedirs(d, exist_ok=True)
    for r in recs:
        with open(os.path.join(d, f"{r['exp_name']}__seed{r['seed']}__{r['role']}.json"), "w") as f:
            json.dump(r, f)


def main():
    base = {c: 60 + 3 * i for i, c in enumerate(CONFIGS)}
    recs, truth = [], {}
    for c in CONFIGS:
        for s in SEEDS:
            if c == "dp_late3_e1" and s == 44:
                recs += [fake(c, s, "M_A", 0, 0, status="failed"), fake(c, s, "M_B", 0, 0, status="failed")]
                continue
            if c == "dp_last1_e5" and s == 43:
                continue  # missing -> pending
            a, b = base[c] + rng.normal(0, 1), base[c] + rng.normal(0, 1)
            pos_b = 0.99 if (c == "dp_early3_e5" and s == 42) else 0.5   # one collapsed model
            recs += [fake(c, s, "M_A", a, 0.5 + 0.01 * s % 3), fake(c, s, "M_B", b, 0.51, pos=pos_b)]
            truth[(c, s)] = (a + b) / 2
    with tempfile.TemporaryDirectory() as td:
        rd, out = os.path.join(td, "records"), os.path.join(td, "analysis")
        write(rd, recs)
        res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "aggregate_grid.py"), "--records", rd,
                              "--out", out, "--estimate"], capture_output=True, text=True)
        print(res.stdout[-800:], res.stderr[-2000:])
        assert res.returncode == 0
        seeds = pd.read_csv(os.path.join(out, "seed_level.csv"))
        summ = pd.read_csv(os.path.join(out, "summary.csv")).set_index("config_name")
        diffs = pd.read_csv(os.path.join(out, "matched_diffs.csv"))
        # 1. seed level = mean of the two swap-pair models
        for (c, s), v in truth.items():
            got = seeds[(seeds.config_name == c) & (seeds.seed == s)].val_accuracy.iloc[0]
            assert abs(got - v) < 1e-9, (c, s, got, v)
        # 2. failed / pending stay visible
        st = seeds.set_index(["config_name", "seed"]).status
        assert st[("dp_late3_e1", 44)] == "failed" and st[("dp_last1_e5", 43)] == "pending"
        assert summ.loc["dp_late3_e1", "n_seeds_ok"] == 2 and "44:failed" in summ.loc["dp_late3_e1", "seeds_not_ok"]
        # 3. mean / SD (ddof=1) across seeds
        v = [truth[("dp_mid3_e5", s)] for s in SEEDS]
        assert abs(summ.loc["dp_mid3_e5", "val_accuracy_mean"] - np.mean(v)) < 1e-9
        assert abs(summ.loc["dp_mid3_e5", "val_accuracy_sd"] - np.std(v, ddof=1)) < 1e-9
        # 4. matched seeds only
        row = diffs[(diffs.comparison == "budget") & (diffs.arm == "dp_late3_e1")].iloc[0]
        assert row.matched_seeds == "42,43" and row.n_matched == 2
        d = [truth[("dp_late3_e1", s)] - truth[("dp_late3_e5", s)] for s in (42, 43)]
        assert abs(row.val_accuracy_diff_mean - np.mean(d)) < 1e-9
        # 5. collapse flag
        assert summ.loc["dp_early3_e5", "n_collapsed_models"] == 1
        tex = open(os.path.join(out, "table_core.tex")).read()
        assert "dagger" in tex and "failed" in tex and "pending" in tex
        for comp in ("location", "budget", "count"):
            assert os.path.getsize(os.path.join(out, f"fig_{comp}.png")) > 10_000
        # 6. mismatched partitions are rejected
        bad = fake("dp_mid3_e5", 42, "M_A", 70, 0.5)
        bad["partition_hash"] = "different"
        write(rd, [bad])
        res2 = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "aggregate_grid.py"), "--records", rd,
                               "--out", out, "--no_figures"], capture_output=True, text=True)
        assert res2.returncode != 0 and "partition hashes" in (res2.stderr + res2.stdout)
        # keep a copy of the synthetic figure for visual inspection
        import shutil
        os.makedirs("/tmp/synthetic_check", exist_ok=True)
        for comp in ("location", "budget", "count"):
            shutil.copy(os.path.join(out, f"fig_{comp}.png"), f"/tmp/synthetic_check/fig_{comp}.png")
        shutil.copy(os.path.join(out, "table_core.tex"), "/tmp/synthetic_check/table_core.tex")
    print("all aggregate_grid tests passed (synthetic data)")


if __name__ == "__main__":
    main()
