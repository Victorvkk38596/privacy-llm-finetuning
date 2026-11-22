import os
import sys
import subprocess

print("=" * 70, flush=True)
print("[KAGGLE WORKER] Starting Autonomous Experiment Grid (Pure Git Clone)", flush=True)
print("=" * 70, flush=True)

# 1. Clone fresh repository from GitHub (Strict: NO encoded fallback)
repo_url = "https://github.com/Victorvkk38596/privacy-llm-finetuning.git"
token = None
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    token = user_secrets.get_secret("GITHUB_TOKEN")
    if token:
        print("[+] Retrieved GITHUB_TOKEN from Kaggle Secrets (Authenticated Clone).", flush=True)
except Exception as e:
    print(f"[-] Note: No Kaggle Secrets GITHUB_TOKEN detected ({e}).", flush=True)

clone_url = f"https://oauth2:{token}@github.com/Victorvkk38596/privacy-llm-finetuning.git" if token else repo_url
repo_dir = "/kaggle/working/privacy-llm-finetuning"

if os.path.exists(repo_dir):
    import shutil
    shutil.rmtree(repo_dir)

print(f"[1/4] Cloning repository directly from {repo_url}...", flush=True)
clone_cmd = f"GIT_TERMINAL_PROMPT=0 git clone {clone_url} {repo_dir}"
clone_res = subprocess.run(clone_cmd, shell=True, capture_output=True, text=True)

if clone_res.returncode != 0:
    print("\n" + "=" * 70, flush=True)
    print("[FATAL ERROR] Repository clone from GitHub failed!", flush=True)
    print(f"Exit code: {clone_res.returncode}", flush=True)
    print(f"Details:\n{clone_res.stderr.strip()}", flush=True)
    print("=" * 70, flush=True)
    print("DIAGNOSTIC ACTION REQUIRED:", flush=True)
    print("The repository Victorvkk38596/privacy-llm-finetuning is private on GitHub.", flush=True)
    print("To allow Kaggle to clone directly without an encoded file fallback:", flush=True)
    print("  Option A: Make the repository Public on GitHub (Settings -> General -> Danger Zone -> Make public).", flush=True)
    print("  Option B: Add a 'GITHUB_TOKEN' (Personal Access Token with 'repo' scope) to Kaggle Secrets (Add-ons -> Secrets).", flush=True)
    print("=" * 70, flush=True)
    sys.exit(1)

print("[✓] Git clone from GitHub repository successful!", flush=True)
os.chdir(repo_dir)

# 2. Install dependencies
print("\n[2/4] Installing requirements from cloned repo...", flush=True)
subprocess.run(f"{sys.executable} -m pip install -q -r requirements.txt", shell=True, check=True)

# 3. Second-half core grid (scripts/run_core_grid.py). Edit STAGE / WARMUP before each push:
#    smoke -> gate (none) -> [gate (public_imdb) only if the first gate collapses] -> validate -> grid
STAGE = os.environ.get("GRID_STAGE", "smoke")      # smoke | gate | validate | grid
WARMUP = os.environ.get("GRID_WARMUP", "none")     # none | public_imdb (from results/grid/warmup_decision_*.json)
SEEDS = os.environ.get("GRID_SEEDS", "")           # e.g. "43" to split the grid across sessions

# Restore records from earlier sessions (attach the previous kernel output as a dataset) so
# completed swap pairs are skipped.
prev = "/kaggle/input/privacy-llm-grid-results/grid"
if os.path.isdir(prev):
    subprocess.run(f"mkdir -p results && cp -rn {prev} results/", shell=True)
    print(f"[+] Restored previous results from {prev}", flush=True)

print(f"\n[3/4] Running stage={STAGE} warmup={WARMUP} seeds={SEEDS or 'all'}...", flush=True)
cmd = f"{sys.executable} scripts/run_core_grid.py --stage {STAGE}"
if STAGE != "smoke":
    cmd += f" --warmup {WARMUP}"
if SEEDS and STAGE == "grid":
    cmd += f" --seeds {SEEDS}"
rc = subprocess.run(cmd, shell=True).returncode

# 4. Export results (records, logits, logs, decision files) for download with `kaggle kernels output`
print("\n[4/4] Exporting results/grid to /kaggle/working/grid ...", flush=True)
subprocess.run("rm -rf /kaggle/working/grid && cp -r results/grid /kaggle/working/grid", shell=True)
subprocess.run(f"{sys.executable} scripts/aggregate_grid.py --records results/grid/records "
               f"--out /kaggle/working/grid/analysis --warmup {WARMUP} --estimate", shell=True)
print(f"\n[done] stage={STAGE} exit={rc}", flush=True)
sys.exit(rc)
