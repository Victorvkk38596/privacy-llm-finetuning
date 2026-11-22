"""Stage runner for the second-half core grid.

Stages (run in order; each is resumable and skips swap pairs whose two records have status ok):

  smoke     3-step DP run (late-3, eps=5, seed 42) + checks that accounting starts before the first
            private-data update, and that private-split warmup is rejected.
  gate      Step 1: seed 42, early-3 / late-3 / all-12 at eps=5 with the chosen --warmup.
            Applies the pre-registered collapse rule (scripts/grid_records.py) and writes
            results/grid/warmup_decision_<warmup>.json.
  validate  One run (seed 42) at every setting not covered by the gate.
  grid      All configs x seeds 42, 43, 44. Failed runs get a failure record and the loop continues.

Examples:
  python scripts/run_core_grid.py --stage smoke
  python scripts/run_core_grid.py --stage gate --warmup none
  python scripts/run_core_grid.py --stage gate --warmup public_imdb      # only if the first gate collapses
  python scripts/run_core_grid.py --stage validate --warmup <chosen>
  python scripts/run_core_grid.py --stage grid --warmup <chosen>
  python scripts/run_core_grid.py --stage grid --warmup <chosen> --plan   # list pending runs, no training
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from grid_spec import CONFIGS, SEEDS, GATE_CONFIGS, NEW_SETTING_CONFIGS, build_config, exp_name  # noqa: E402
from grid_records import load_records, index_records, pair_status, gate_decision  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "grid")
RECORDS = os.path.join(RESULTS, "records")
LOGITS = os.path.join(RESULTS, "logits")
CFG_DIR = os.path.join(RESULTS, "configs_used")
LOG_DIR = os.path.join(RESULTS, "logs")


def _yaml_dump(cfg: dict) -> str:
    import yaml
    return yaml.safe_dump(cfg, sort_keys=False)


def _rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def command_for(config_name: str, warmup: str, seed: int, records_dir: str = RECORDS,
                logits_dir: str = LOGITS, extra=()) -> list:
    cfg = build_config(config_name, warmup)
    os.makedirs(CFG_DIR, exist_ok=True)
    cfg_path = os.path.join(CFG_DIR, f"{cfg['exp_name']}.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(_yaml_dump(cfg))
    # repo-relative paths; commands run with cwd=ROOT
    return ["python", "run_experiment.py", "--config", _rel(cfg_path),
            "--seed", str(seed), "--output_dir", _rel(logits_dir), "--records_dir", _rel(records_dir),
            "--summary_csv", _rel(os.path.join(os.path.dirname(records_dir), "summary_rows.csv")), *extra]


def run_pair(config_name: str, warmup: str, seed: int, records_dir: str = RECORDS, extra=()) -> str:
    name = exp_name(config_name, warmup)
    cmd = command_for(config_name, warmup, seed, records_dir=records_dir,
                      logits_dir=os.path.join(os.path.dirname(records_dir), "logits"), extra=extra)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{name}__seed{seed}.log")
    print(f"\n=== {name} seed {seed} ===\n$ {' '.join(cmd)}\n(log: {log_path})", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable] + cmd[1:], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    st = pair_status(index_records(load_records(records_dir)), name, seed)
    if proc.returncode != 0 and st == "missing":
        # hard crash (e.g. OOM-kill) before run_experiment could write a failure record
        sys.path.insert(0, ROOT)
        from run_experiment import write_failure_records
        tail = open(log_path, encoding="utf-8", errors="replace").read()[-4000:]
        write_failure_records(records_dir, build_config(config_name, warmup), seed,
                              f"exit code {proc.returncode}\n{tail}")
        st = "failed"
    print(f"--> {name} seed {seed}: exit={proc.returncode} status={st} ({dt/60:.1f} min)", flush=True)
    return st


def planned(stage: str):
    if stage == "gate":
        return [(c, 42) for c in GATE_CONFIGS]
    if stage == "validate":
        return [(c, 42) for c in NEW_SETTING_CONFIGS]
    if stage == "grid":
        return [(c, s) for s in SEEDS for c in CONFIGS]
    raise ValueError(stage)


def stage_smoke() -> int:
    """Accounting-order smoke test on the real pipeline (3 DP steps)."""
    smoke_dir = os.path.join(RESULTS, "smoke", "records")
    failures = []
    st = run_pair("dp_late3_e5", "none", 42, records_dir=smoke_dir, extra=["--max_steps", "3"])
    idx = index_records(load_records(smoke_dir))
    for role in ("M_A", "M_B"):
        r = idx.get((exp_name("dp_late3_e5", "none"), 42, role))
        if r is None or r.get("status") != "ok":
            failures.append(f"{role}: run status {None if r is None else r.get('status')}")
            continue
        led = r["privacy"]["accounting_ledger"]
        checks = {
            "accounting activated before first private read":
                led["accounting_activated_event"] is not None and led["first_private_read_event"] is not None
                and led["accounting_activated_event"] < led["first_private_read_event"],
            "zero private reads before accounting": led["n_private_reads_before_accounting"] == 0,
            "accountant had 0 steps at first private read": led["accountant_steps_at_first_private_read"] == 0,
            "accountant steps == logical optimizer steps == 3":
                led["accountant_total_steps"] == led["logical_steps"] == 3,
            "no public warmup in this run": led["public_warmup"] is None,
            "accounted epsilon <= target": r["privacy"]["accounted_epsilon"] <= r["privacy"]["target_epsilon"] + 1e-6,
        }
        for k, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {role}: {k}")
            if not ok:
                failures.append(f"{role}: {k}")
    # Private-split warmup must be rejected outright.
    cfg = build_config("dp_late3_e5", "none")
    sys.path.insert(0, ROOT)
    try:
        from src.train import train_single_model
        cfg["head_warmup_source"] = "private"
        try:
            train_single_model("M_A", "A", cfg, 42, output_dir=os.path.join(RESULTS, "smoke", "logits"))
            failures.append("private warmup was NOT rejected")
            print("  [FAIL] head_warmup_source=private rejected")
        except ValueError as e:
            print(f"  [PASS] head_warmup_source=private rejected ({str(e)[:80]}...)")
    except ImportError as e:
        failures.append(f"could not import src.train: {e}")
    out = {"stage": "smoke", "status": st, "failures": failures, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(os.path.join(RESULTS, "smoke"), exist_ok=True)
    with open(os.path.join(RESULTS, "smoke", "smoke_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSMOKE TEST:", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "gate", "validate", "grid"])
    ap.add_argument("--warmup", default="none", choices=["none", "public_imdb"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="restrict grid stage to these seeds")
    ap.add_argument("--configs", nargs="*", default=None, help="restrict to these config names")
    ap.add_argument("--plan", action="store_true", help="print pending runs and commands, do not train")
    args = ap.parse_args()

    if args.stage == "smoke":
        if args.plan:
            print(" ".join(command_for("dp_late3_e5", "none", 42, extra=["--max_steps", "3"])))
            return 0
        return stage_smoke()

    todo = planned(args.stage)
    if args.seeds:
        todo = [(c, s) for c, s in todo if s in args.seeds]
    if args.configs:
        todo = [(c, s) for c, s in todo if c in args.configs]
    idx = index_records(load_records(RECORDS))
    pending = [(c, s) for c, s in todo if pair_status(idx, exp_name(c, args.warmup), s) != "ok"]
    print(f"Stage {args.stage} (warmup={args.warmup}): {len(todo)} swap pairs, {len(todo) - len(pending)} done, "
          f"{len(pending)} pending")
    if args.plan:
        for c, s in pending:
            print(f"  {exp_name(c, args.warmup):28s} seed {s}: "
                  f"{pair_status(idx, exp_name(c, args.warmup), s)}")
            print("    " + " ".join(command_for(c, args.warmup, s)))
        return 0

    results = {}
    for c, s in pending:
        results[(c, s)] = run_pair(c, args.warmup, s)

    if args.stage == "gate":
        dec = gate_decision(RECORDS, warmup=args.warmup)
        path = os.path.join(RESULTS, f"warmup_decision_{args.warmup}.json")
        with open(path, "w") as f:
            json.dump(dec, f, indent=2)
        print(json.dumps(dec, indent=2))
        print(f"Decision written to {path}")
    bad = {k: v for k, v in results.items() if v != "ok"}
    if bad:
        print(f"\n{len(bad)} swap pair(s) not ok (records kept, see results/grid/logs): {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
