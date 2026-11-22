"""Static checks that the runner stages cover the intended grid (no training).

  python tests/test_grid_wiring.py      (or: pytest tests/)
"""
import os
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import grid_spec  # noqa: E402
import run_core_grid as rcg  # noqa: E402


def test_spec_checks():
    grid_spec.check_spec()
    assert grid_spec.SEEDS == [42, 43, 44]
    assert len(grid_spec.CONFIGS) == 9


def test_stage_coverage():
    gate, val, grid = rcg.planned("gate"), rcg.planned("validate"), rcg.planned("grid")
    assert all(s == 42 for _, s in gate + val)
    assert {c for c, _ in gate} | {c for c, _ in val} == set(grid_spec.CONFIGS)   # every config validated once
    assert not ({c for c, _ in gate} & {c for c, _ in val})
    assert {c for c, _ in gate} == {"dp_early3_e5", "dp_late3_e5", "dp_all_e5"}
    assert len(grid) == 27 and len(set(grid)) == 27
    assert set(grid) == {(c, s) for c in grid_spec.CONFIGS for s in (42, 43, 44)}


def test_step_counts_and_configs():
    for warmup in ("none", "public_imdb"):
        for c, (layers, eps, comps, _) in grid_spec.CONFIGS.items():
            cfg = grid_spec.build_config(c, warmup)
            assert cfg["max_steps"] == (1000 if eps is None else 300), c
            assert cfg["use_dp"] == (eps is not None)
            assert cfg["train_layers"] == layers
            if eps is None:
                assert cfg["exp_name"] == c and cfg["head_warmup_source"] == "none"
            else:
                assert cfg["exp_name"] == f"{c}__w-{warmup}" and cfg["head_warmup_source"] == warmup
                assert cfg["target_epsilon"] == eps and cfg["target_delta"] == 1e-5


def test_commands_and_generated_yaml():
    old = rcg.CFG_DIR
    with tempfile.TemporaryDirectory() as td:
        rcg.CFG_DIR = td
        try:
            cmd = rcg.command_for("dp_last6_e5", "none", 43)
            assert cmd[:2] == ["python", "run_experiment.py"] and "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "43"
            path = os.path.join(td, "dp_last6_e5__w-none.yaml")
            cfg = yaml.safe_load(open(path))
            assert isinstance(cfg["target_delta"], float) and cfg["max_steps"] == 300
            assert str(cfg["train_layers"]) == "6,7,8,9,10,11"
            smoke = rcg.command_for("dp_late3_e5", "none", 42, extra=["--max_steps", str(rcg.SMOKE_STEPS)])
            assert smoke[-2:] == ["--max_steps", "3"]
        finally:
            rcg.CFG_DIR = old


def test_committed_grid_yaml_match_spec():
    d = os.path.join(ROOT, "configs", "grid")
    for warmup in ("none", "public_imdb"):
        for c in grid_spec.CONFIGS:
            cfg = grid_spec.build_config(c, warmup)
            path = os.path.join(d, f"{cfg['exp_name']}.yaml")
            on_disk = yaml.safe_load(open(path))
            assert on_disk == cfg, path


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"[PASS] {name}")
    print(f"all {n} grid-wiring tests passed")
