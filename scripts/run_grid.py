"""Resumable driver for the second-half core grid (3 seeds x swap pairs).

Usage:
  python scripts/run_grid.py --list                       # show planned / done / pending runs
  python scripts/run_grid.py                              # run everything still pending
  python scripts/run_grid.py --only late3_eps5 --seeds 42 # run a subset

Each (config, seed) runs `run_experiment.py` in its own process, which trains M_A on D_A and
M_B on D_B and appends two per-model records to results/grid/records.jsonl.
A (config, seed) pair counts as done when both roles have a status="ok" record.
A failed pair gets a status="failed" record (with the log path) so failures stay visible.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "results/grid/records.jsonl"
LOGITS = ROOT / "results/grid/logits"
LOGS = ROOT / "results/grid/logs"
SEEDS = [42, 43, 44]

# Order matters: shared configs (late3_eps5, all12_eps5) run first so every comparison
# accumulates matched seeds as early as possible.
GRID = [
    "late3_eps5",      # location + budget + count (shared)
    "early3_eps5",     # location
    "mid3_eps5",       # location
    "all12_eps5",      # location + count (shared)
    "clean_all12",     # utility context (no DP)
    "late3_eps1",      # budget
    "late3_eps8",      # budget
    "last1_eps5",      # count
    "last6_eps5",      # count
]


def config_path(name: str) -> Path:
    return ROOT / "configs/grid" / f"{name}.yaml"


def load_records():
    if not RECORDS.exists():
        return []
    with open(RECORDS) as f:
        return [json.loads(l) for l in f if l.strip()]


def done_pairs(records):
    ok = {}
    for r in records:
        if r.get("status") == "ok":
            ok.setdefault((r["exp_name"], r["seed"]), set()).add(r["role"])
    return {k for k, roles in ok.items() if roles >= {"M_A", "M_B"}}


def run_one(name: str, seed: int) -> bool:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}_s{seed}.log"
    cmd = [sys.executable, str(ROOT / "run_experiment.py"),
           "--config", str(config_path(name).relative_to(ROOT)),
           "--seed", str(seed),
           "--output_dir", str(LOGITS.relative_to(ROOT)),
           "--records", str(RECORDS.relative_to(ROOT))]
    t0 = time.time()
    print(f"[run] {name} seed={seed} -> {log.relative_to(ROOT)}", flush=True)
    with open(log, "w") as fh:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    dt = time.time() - t0
    if rc != 0:
        tail = log.read_text().splitlines()[-15:]
        rec = {"status": "failed", "exp_name": name, "seed": seed, "role": "pair",
               "returncode": rc, "log": str(log.relative_to(ROOT)), "error_tail": tail,
               "wall_time_sec": dt, "config_path": str(config_path(name).relative_to(ROOT))}
        RECORDS.parent.mkdir(parents=True, exist_ok=True)
        with open(RECORDS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[FAIL] {name} seed={seed} rc={rc} after {dt/60:.1f} min (see {log})", flush=True)
        return False
    print(f"[ok] {name} seed={seed} in {dt/60:.1f} min", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    names = args.only or GRID
    for n in names:
        if not config_path(n).exists():
            sys.exit(f"missing config {config_path(n)}")
    done = done_pairs(load_records())
    plan = [(n, s) for s in args.seeds for n in names]  # seed-major: finish a seed before the next
    pending = [(n, s) for n, s in plan if (n, s) not in done]

    for n, s in plan:
        print(f"  {'done   ' if (n, s) in done else 'PENDING'}  {n:14s} seed={s}")
    print(f"{len(plan) - len(pending)} done, {len(pending)} pending")
    if args.list:
        return
    failures = 0
    for n, s in pending:
        failures += not run_one(n, s)
    print(f"Finished: {len(pending) - failures} ok, {failures} failed")


if __name__ == "__main__":
    main()
