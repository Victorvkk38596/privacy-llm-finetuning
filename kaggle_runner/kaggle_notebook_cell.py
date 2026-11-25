"""Kaggle notebook cell used for the corrected three-seed core grid (run on 2026-11-25).

This is the exact single cell that produced results/grid: it ran as one background job
("Save Version -> Save & Run All (Commit)") on a Kaggle T4 with Internet on, cloned commit
04424d3 of this repository, and took about 4.3 hours. Every record in results/grid/records
stores that commit hash in env.git_commit.

How to reuse it:
  1. New Kaggle notebook; Session options: Accelerator = GPU T4, Internet = On.
  2. Paste this file's contents as the notebook's only cell.
  3. Save Version -> Save & Run All (Commit). Do not use the interactive Run button for long jobs.
  4. When the version finishes, download the `grid` output folder and copy it to results/grid.
  5. To resume a run that stopped early, add the previous version's output as an input and run again:
     completed swap pairs are found under /kaggle/input and skipped.

Stages: smoke test -> public-warmup gate -> validation runs -> full grid, then the no-warmup gate is
rerun to keep that evidence. It stops early only if the smoke test fails or the gate collapses.
WARMUP is "public_imdb" because the no-warmup gate collapsed (see results/grid/warmup_decision_none.json).
"""
import os, sys, subprocess, shutil, json, glob

WARMUP = "public_imdb"                          # chosen because the no-warmup gate collapsed
STAGES = ["smoke", "gate", "validate", "grid"]
RERUN_NO_WARMUP_GATE = True                     # re-creates the no-warmup evidence at the end (~25 min)

REPO = "https://github.com/Victorvkk38596/privacy-llm-finetuning.git"
WORK = "/kaggle/working/privacy-llm-finetuning"
OUT  = "/kaggle/working/grid"

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    if check and rc != 0:
        raise SystemExit(f"failed ({rc}): {cmd}")
    return rc

# 0. GPU
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if not torch.cuda.is_available():
    raise SystemExit("No GPU: set the accelerator to GPU T4.")

# 1. Fresh clone
os.chdir("/kaggle/working")
shutil.rmtree(WORK, ignore_errors=True)
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
url = REPO.replace("https://", f"https://oauth2:{token}@") if token else REPO
r = subprocess.run(["git", "clone", "--depth", "1", url, WORK], capture_output=True, text=True,
                   env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
if r.returncode:
    print(r.stderr.replace(token, "***") if token else r.stderr)
    raise SystemExit("clone failed (is Internet on?)")
os.chdir(WORK)
sh("git log --oneline -1")

# 2. Dependencies, 3. fast CPU preflight
sh(f"{sys.executable} -m pip install -q -r requirements.txt")
for t in ("tests/test_privacy_guard.py", "tests/test_record_validation.py", "tests/test_grid_wiring.py"):
    sh(f"{sys.executable} {t}")

# 4. Resume: reuse records from this session or from an earlier run attached as input
def find_previous():
    for c in [OUT] + sorted(glob.glob("/kaggle/input/**/grid", recursive=True)):
        if os.path.isdir(os.path.join(c, "records")):
            return c
prev = find_previous()
if prev:
    shutil.copytree(prev, "results/grid", dirs_exist_ok=True)
    md = os.path.join(prev, "manifests")
    for f in (os.listdir(md) if os.path.isdir(md) else []):
        if not os.path.exists(os.path.join("manifests", f)):
            shutil.copy(os.path.join(md, f), "manifests")
    print("restored previous results from", prev)

def export():
    os.makedirs(OUT, exist_ok=True)
    if os.path.isdir("results/grid"):
        shutil.copytree("results/grid", OUT, dirs_exist_ok=True)
    os.makedirs(f"{OUT}/manifests", exist_ok=True)
    for s in (43, 44):
        p = f"manifests/manifest_seed{s}.json"
        if os.path.exists(p):
            shutil.copy(p, f"{OUT}/manifests/")

# 5. Stages, exporting after each one
for stage in STAGES:
    cmd = f"{sys.executable} scripts/run_core_grid.py --stage {stage}"
    if stage != "smoke":
        cmd += f" --warmup {WARMUP}"
    sh(cmd, check=False)
    export()
    if stage == "smoke":
        p = "results/grid/smoke/smoke_result.json"
        ok = os.path.exists(p) and not json.load(open(p))["failures"]
        print("\nSMOKE:", "passed" if ok else "FAILED", flush=True)
        if not ok:
            break
    if stage == "gate":
        p = f"results/grid/warmup_decision_{WARMUP}.json"
        verdict = json.load(open(p))["verdict"] if os.path.exists(p) else "missing decision file"
        print("\nGATE VERDICT:", verdict, flush=True)
        if not verdict.startswith("learns"):
            print("Stopping here: send the decision file before changing anything.")
            break
else:
    if RERUN_NO_WARMUP_GATE:
        sh(f"{sys.executable} scripts/run_core_grid.py --stage gate --warmup none", check=False)
        export()

# 6. Analysis
sh(f"{sys.executable} scripts/aggregate_grid.py --records results/grid/records "
   f"--out {OUT}/analysis --warmup {WARMUP} --estimate", check=False)
for f in (f"warmup_decision_{WARMUP}.json", "warmup_decision_none.json", "analysis/runtime_estimate.txt"):
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        print(f"\n--- {f} ---\n" + open(p).read())
print(f"\nDone. Outputs in {OUT}")
