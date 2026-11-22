"""Regression tests for the shared acceptance rule (src/record_validation.py) and its use in the
runner (scripts/grid_records.grid_pair_states), the aggregator (scripts/aggregate_grid.seed_table)
and run_experiment.finalize_record_status. SYNTHETIC records in temp dirs only.

  python tests/test_record_validation.py      (or: pytest tests/)
"""
import math
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
from synthetic_records import make_record, failed_record, write_records, HASH  # noqa: E402
from src.record_validation import validate_model, validate_pair, evaluate_pairs  # noqa: E402
from grid_records import load_records, index_records, grid_pair_states, MANIFESTS_DIR  # noqa: E402
from grid_spec import exp_name  # noqa: E402

CFG = "dp_late3_e5"


def _pair(**over):
    return make_record(CFG, 42, "M_A", **over), make_record(CFG, 42, "M_B", **over)


def test_valid_pair_ok():
    a, b = _pair()
    assert validate_model(a, 300) == []
    assert validate_pair(a, b, 300) == ("ok", [])


def test_clean_pair_ok_without_epsilon():
    a, b = make_record("clean_all", 42, "M_A"), make_record("clean_all", 42, "M_B")
    assert validate_pair(a, b, 1000)[0] == "ok"


def test_mismatched_hash_within_pair():
    a, b = _pair()
    b["partition_hash"] = "other"
    st, why = validate_pair(a, b, 300)
    assert st == "invalid" and any("differs within pair" in w for w in why)


def test_missing_hash():
    for missing in (None, ""):
        a, b = _pair()
        a["partition_hash"] = missing
        st, why = validate_pair(a, b, 300)
        assert st == "invalid" and any("missing partition hash" in w for w in why), (missing, why)


def test_invalid_epsilon_values():
    for bad, msg in ((None, "missing or non-finite"), (float("nan"), "missing or non-finite"),
                     (float("inf"), "missing or non-finite"), ("abc", "missing or non-finite"),
                     (5.01, "> target"), (6.0, "> target")):
        a, b = _pair(**{"privacy.accounted_epsilon": bad})
        st, why = validate_pair(a, b, 300)
        assert st == "invalid" and any(msg in w for w in why), (bad, why)
    a, b = _pair(**{"privacy.accounted_epsilon": 5.0})      # exactly at target is accepted
    assert validate_pair(a, b, 300)[0] == "ok"


def test_accounting_checks_required_for_dp():
    for key in ("ordering_ok", "steps_match_ok"):
        for val in (False, None):
            a, b = _pair(**{f"privacy.accounting_ledger.{key}": val})
            assert validate_pair(a, b, 300)[0] == "invalid", (key, val)
    a, b = _pair(**{"privacy.accounting_ledger.accountant_total_steps": 299})
    assert validate_pair(a, b, 300)[0] == "invalid"
    a, b = _pair()
    a["privacy"]["accounting_ledger"] = {}
    assert validate_pair(a, b, 300)[0] == "invalid"


def test_steps_and_spec():
    a, b = _pair(**{"training.steps_completed": 250})
    assert validate_pair(a, b, 300)[0] == "invalid"
    a, b = _pair(**{"training.max_steps": 3, "training.steps_completed": 3,
                    "privacy.accounting_ledger.accountant_total_steps": 3})
    assert validate_pair(a, b, 300)[0] == "invalid"           # smoke-length run is not a grid run
    assert validate_pair(a, b, 3)[0] == "ok"


def test_private_warmup_record_rejected():
    a, b = _pair(**{"training.head_warmup": {"source": "private"}})
    assert validate_pair(a, b, 300)[0] == "invalid"


def test_failed_pending_incomplete():
    a = make_record(CFG, 42, "M_A")
    assert validate_pair(None, None)[0] == "pending"
    assert validate_pair(a, None)[0] == "incomplete"
    assert validate_pair(a, failed_record(CFG, 42, "M_B"))[0] == "failed"


def test_non_finite_metrics():
    a, b = _pair(**{"metrics.rmia_auc": float("nan")})
    assert validate_pair(a, b, 300)[0] == "invalid"


def test_cross_config_hash_reference():
    # seed 43 has no manifest: majority of valid pairs is the reference; the odd one is rejected
    idx = {}
    for cfg, h in (("dp_early3_e5", "X"), ("dp_mid3_e5", "X"), ("dp_late3_e5", "Y")):
        for role in ("M_A", "M_B"):
            r = make_record(cfg, 43, role)
            r["partition_hash"] = h
            idx[(r["exp_name"], 43, role)] = r
    wanted = [(exp_name(c, "none"), 43) for c in ("dp_early3_e5", "dp_mid3_e5", "dp_late3_e5")]
    res = evaluate_pairs(idx, wanted, None, manifests_dir=None)
    assert res[(exp_name("dp_early3_e5", "none"), 43)]["status"] == "ok"
    assert res[(exp_name("dp_late3_e5", "none"), 43)]["status"] == "partition_mismatch"
    # tie -> no reference -> nothing accepted for that seed
    res = evaluate_pairs(idx, wanted[1:], None, manifests_dir=None)
    assert {v["status"] for v in res.values()} == {"partition_mismatch"}
    # seed 42: manifest hash is the reference even if it is the minority
    idx42 = {}
    for cfg, h in (("dp_early3_e5", "Z"), ("dp_mid3_e5", "Z"), ("dp_late3_e5", HASH[42])):
        for role in ("M_A", "M_B"):
            r = make_record(cfg, 42, role)
            r["partition_hash"] = h
            idx42[(r["exp_name"], 42, role)] = r
    res = evaluate_pairs(idx42, [(n, 42) for n, _ in wanted], None, manifests_dir=MANIFESTS_DIR)
    assert res[(exp_name("dp_late3_e5", "none"), 42)]["status"] == "ok"
    assert res[(exp_name("dp_early3_e5", "none"), 42)]["status"] == "partition_mismatch"


def test_known_bug_mismatched_hash_and_eps6_not_ok_everywhere():
    """Regression: stored status 'ok', hashes differ within the pair, accounted eps 6 vs target 5.
    Before the fix the runner (pair_status) and aggregator (seed_table) both reported 'ok'."""
    import aggregate_grid
    import run_experiment
    a, b = _pair(**{"privacy.accounted_epsilon": 6.0})
    a["partition_hash"], b["partition_hash"] = "aaaa", "bbbb"
    assert a["status"] == b["status"] == "ok"
    with tempfile.TemporaryDirectory() as td:
        write_records(td, [a, b])
        recs = load_records(td)
        # runner
        st = grid_pair_states(index_records(recs), "none")[(CFG, 42)]
        assert st["status"] == "invalid", st
        assert any("> target 5" in r for r in st["reasons"]) and any("differs within pair" in r for r in st["reasons"])
        # aggregator: visible as invalid, excluded from statistics
        seeds = aggregate_grid.seed_table(recs, "none", records_dir=td)
        row = seeds[(seeds.config_name == CFG) & (seeds.seed == 42)].iloc[0]
        assert row.status == "invalid" and math.isnan(row.val_accuracy)
        summ = aggregate_grid.summary_table(seeds).set_index("config_name")
        assert summ.loc[CFG, "n_seeds_ok"] == 0 and "42:invalid" in summ.loc[CFG, "seeds_not_ok"]
    # run_experiment would no longer write status ok for such a model
    rec = make_record(CFG, 42, "M_A", **{"privacy.accounted_epsilon": 6.0})
    run_experiment.finalize_record_status(rec)
    assert rec["status"] == "invalid" and rec["invalid_reasons"]
    ok = make_record(CFG, 42, "M_A")
    run_experiment.finalize_record_status(ok)
    assert ok["status"] == "ok" and "invalid_reasons" not in ok


def test_runner_and_aggregator_agree():
    """Same acceptance in both places, on a mix of valid, invalid, failed and pending pairs."""
    import aggregate_grid
    recs = []
    recs += list(_pair())                                                        # ok
    recs += [make_record("dp_mid3_e5", 42, r, **{"privacy.accounted_epsilon": float("nan")}) for r in ("M_A", "M_B")]
    recs += [make_record("dp_all_e5", 42, "M_A"), failed_record("dp_all_e5", 42, "M_B")]
    recs += [make_record("clean_all", 42, "M_A")]                                # incomplete
    with tempfile.TemporaryDirectory() as td:
        write_records(td, recs)
        loaded = load_records(td)
        runner = grid_pair_states(index_records(loaded), "none")
        seeds = aggregate_grid.seed_table(loaded, "none", records_dir=td).set_index(["config_name", "seed"])
        for key, st in runner.items():
            assert seeds.loc[key, "status"] == st["status"], (key, st["status"], seeds.loc[key, "status"])
        assert runner[(CFG, 42)]["status"] == "ok"
        assert runner[("dp_mid3_e5", 42)]["status"] == "invalid"
        assert runner[("dp_all_e5", 42)]["status"] == "failed"
        assert runner[("clean_all", 42)]["status"] == "incomplete"
        assert runner[("dp_last1_e5", 44)]["status"] == "pending"


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"[PASS] {name}")
    print(f"all {n} record-validation tests passed (synthetic records)")
