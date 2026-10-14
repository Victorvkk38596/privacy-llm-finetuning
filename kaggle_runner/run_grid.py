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

# 3. Define the 8 core grid configs
configs = [
    "baseline_full_clean.yaml",
    "baseline_full_dp.yaml",
    "head_only_dp.yaml",
    "selective_early_k3.yaml",
    "selective_mid_k3.yaml",
    "selective_late_k3.yaml",
    "selective_dpselft_k3.yaml",
    "control_H_prime.yaml"
]

# 4. Execute all runs sequentially
print("\n[3/4] Running Core Grid (8 Configurations sequentially)...", flush=True)
sep = "=" * 70
for idx, cfg in enumerate(configs, 1):
    print(f"\n{sep}\n[KAGGLE WORKER {idx}/8] Executing: {cfg}\n{sep}", flush=True)
    cmd = f"{sys.executable} run_experiment.py --config configs/{cfg}"
    subprocess.run(cmd, shell=True, check=True)

# 5. Export results to /kaggle/working so Kaggle CLI can download them
print("\n[4/4] Exporting results to /kaggle/working...", flush=True)
subprocess.run("cp results/summary.csv /kaggle/working/summary.csv", shell=True)
subprocess.run("cp -r results/logits /kaggle/working/", shell=True)
print("\n[✓] ALL 8 CONFIGURATIONS COMPLETED SUCCESSFULLY!", flush=True)
