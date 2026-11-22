"""Tests for scripts/aggregate_grid.py on SYNTHETIC records (temp dirs only; never results/).

  python tests/test_aggregate_grid.py      (or: pytest tests/)

Set PRIVACY_GRID_KEEP_DIR=<dir> to keep the synthetic figures/tables for visual inspection
(any OS; nothing is kept by default).
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from grid_spec import CONFIGS, SEEDS  # noqa: E402
from synthetic_records import make_record, failed_record, write_records  # noqa: E402

AGG = os.path.join(ROOT, "scripts", "aggregate_grid.py")


def _build(rng):
    base = {c: 60 + 3 * i for i, c in enumerate(CONFIGS)}
    recs, truth_acc, truth_auc = [], {}, {}
    for c in CONFIGS:
        for s in SEEDS:
            if c == "dp_late3_e1" and s == 44:
                recs += [failed_record(c, s, "M_A"), failed_record(c, s, "M_B")]
                continue
            if c == "dp_last1_e5" and s == 43:
                continue                                    # pending
            if c == "dp_late3_e8" and s == 43:              # known-bug pattern: invalid, must be excluded
                a = make_record(c, s, "M_A", **{"privacy.accounted_epsilon": 9.0})
                b = make_record(c, s, "M_B", **{"privacy.accounted_epsilon": 9.0})
                b["partition_hash"] = "zzzz"
                recs += [a, b]
                continue
            acc = (base[c] + rng.normal(0, 1), base[c] + rng.normal(0, 1))
            auc = (0.5 + 0.01 * rng.random(), 0.5 + 0.01 * rng.random())
            pos_b = 0.99 if (c == "dp_early3_e5" and s == 42) else 0.5     # one collapsed model
            recs += [make_record(c, s, "M_A", acc=acc[0], auc=auc[0]),
                     make_record(c, s, "M_B", acc=acc[1], auc=auc[1], pos=pos_b)]
            truth_acc[(c, s)] = (acc[0] + acc[1]) / 2
            truth_auc[(c, s)] = (auc[0] + auc[1]) / 2
    # cross-config mismatch: seed 44 has no manifest; majority hash is 'hash44', this pair differs
    for r in recs:
        if r["config_name"] == "dp_mid3_e5" and r["seed"] == 44:
            r["partition_hash"] = "other44"
    truth_acc.pop(("dp_mid3_e5", 44))
    truth_auc.pop(("dp_mid3_e5", 44))
    return recs, truth_acc, truth_auc


def _run(rec_dir, out, *extra):
    res = subprocess.run([sys.executable, AGG, "--records", rec_dir, "--out", out, *extra],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-3000:]
    return res.stdout


def test_aggregate_synthetic_grid():
    recs, truth_acc, truth_auc = _build(np.random.default_rng(0))
    with tempfile.TemporaryDirectory() as td:
        rd, out = os.path.join(td, "records"), os.path.join(td, "analysis")
        write_records(rd, recs)
        stdout = _run(rd, out, "--estimate")
        seeds = pd.read_csv(os.path.join(out, "seed_level.csv"))
        summ = pd.read_csv(os.path.join(out, "summary.csv")).set_index("config_name")
        diffs = pd.read_csv(os.path.join(out, "matched_diffs.csv"))
        pareto = pd.read_csv(os.path.join(out, "pareto_status.csv")).set_index("config_name")
        st = seeds.set_index(["config_name", "seed"]).status

        # 1. seed level = mean of the two swap-pair models (before any across-seed statistic)
        for (c, s), v in truth_acc.items():
            row = seeds[(seeds.config_name == c) & (seeds.seed == s)].iloc[0]
            assert abs(row.val_accuracy - v) < 1e-9 and abs(row.rmia_auc - truth_auc[(c, s)]) < 1e-9, (c, s)
            assert abs(row.val_accuracy - (row.val_accuracy_M_A + row.val_accuracy_M_B) / 2) < 1e-9

        # 2. failed / pending / invalid / partition mismatch stay visible and are excluded
        assert st[("dp_late3_e1", 44)] == "failed"
        assert st[("dp_last1_e5", 43)] == "pending"
        assert st[("dp_late3_e8", 43)] == "invalid"
        assert st[("dp_mid3_e5", 44)] == "partition_mismatch"
        bad = seeds[seeds.status != "ok"]
        assert bad.val_accuracy.isna().all()
        assert "dp_late3_e8__w-none seed 43: invalid" in stdout
        r = seeds[(seeds.config_name == "dp_late3_e8") & (seeds.seed == 43)].iloc[0]
        assert "> target 8" in r.reasons and "differs within pair" in r.reasons
        assert summ.loc["dp_late3_e1", "n_seeds_ok"] == 2 and "44:failed" in summ.loc["dp_late3_e1", "seeds_not_ok"]
        assert summ.loc["dp_late3_e8", "n_seeds_ok"] == 2 and "43:invalid" in summ.loc["dp_late3_e8", "seeds_not_ok"]

        # 3. mean / SD (ddof=1) across seed-level values
        v = [truth_acc[("dp_late3_e5", s)] for s in SEEDS]
        assert abs(summ.loc["dp_late3_e5", "val_accuracy_mean"] - np.mean(v)) < 1e-9
        assert abs(summ.loc["dp_late3_e5", "val_accuracy_sd"] - np.std(v, ddof=1)) < 1e-9

        # 4. matched seeds only
        row = diffs[(diffs.comparison == "budget") & (diffs.arm == "dp_late3_e1")].iloc[0]
        assert row.matched_seeds == "42,43" and row.n_matched == 2
        d = [truth_acc[("dp_late3_e1", s)] - truth_acc[("dp_late3_e5", s)] for s in (42, 43)]
        assert abs(row.val_accuracy_diff_mean - np.mean(d)) < 1e-9
        row = diffs[(diffs.comparison == "location") & (diffs.arm == "dp_mid3_e5")].iloc[0]
        assert row.matched_seeds == "42,43"                         # seed 44 mismatch excluded

        # 5. collapse flag; Pareto eligibility (3 valid seeds, no collapsed model)
        assert summ.loc["dp_early3_e5", "n_collapsed_models"] == 1
        assert not pareto.loc["dp_early3_e5", "eligible"] and "collapsed" in pareto.loc["dp_early3_e5", "excluded_because"]
        for c in ("dp_late3_e1", "dp_last1_e5", "dp_late3_e8", "dp_mid3_e5"):
            assert not pareto.loc[c, "eligible"], c
        for c in ("dp_late3_e5", "dp_all_e5", "dp_last6_e5", "clean_all"):
            assert pareto.loc[c, "eligible"], c
        el = pareto[pareto.eligible]
        for c in el.index:          # frontier = not dominated by another eligible config
            dom = any((el.loc[o, "val_accuracy_mean"] >= el.loc[c, "val_accuracy_mean"]) and
                      (el.loc[o, "rmia_auc_mean"] <= el.loc[c, "rmia_auc_mean"]) and
                      ((el.loc[o, "val_accuracy_mean"] > el.loc[c, "val_accuracy_mean"]) or
                       (el.loc[o, "rmia_auc_mean"] < el.loc[c, "rmia_auc_mean"])) for o in el.index if o != c)
            assert pareto.loc[c, "on_frontier"] == (not dom), c
        assert not pareto[~pareto.eligible].on_frontier.any()

        tex = open(os.path.join(out, "table_core.tex")).read()
        assert "dagger" in tex and "failed" in tex and "pending" in tex and r"partition\_mismatch" in tex
        for name in ("location", "budget", "count", "pareto"):
            assert os.path.getsize(os.path.join(out, f"fig_{name}.png")) > 10_000, name
        keep = os.environ.get("PRIVACY_GRID_KEEP_DIR")
        if keep:
            shutil.copytree(out, os.path.join(keep, "synthetic_analysis"), dirs_exist_ok=True)


def test_aggregate_with_no_records():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "analysis")
        _run(os.path.join(td, "records_missing"), out)
        seeds = pd.read_csv(os.path.join(out, "seed_level.csv"))
        assert len(seeds) == 27 and set(seeds.status) == {"pending"}
        assert not pd.read_csv(os.path.join(out, "pareto_status.csv")).eligible.any()
        assert os.path.exists(os.path.join(out, "fig_pareto.png"))


if __name__ == "__main__":
    test_aggregate_synthetic_grid()
    print("[PASS] test_aggregate_synthetic_grid")
    test_aggregate_with_no_records()
    print("[PASS] test_aggregate_with_no_records")
    print("all aggregate_grid tests passed (synthetic records)")
