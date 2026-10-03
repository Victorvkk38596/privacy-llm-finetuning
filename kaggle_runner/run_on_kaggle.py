import os
import sys
import shutil
import subprocess

def run_cmd(cmd, cwd=None):
    """Executes a shell command and streams output."""
    print(f"\n[EXEC] {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")

def main():
    print("=" * 70, flush=True)
    print("=== Privacy LLM Fine-Tuning - Kaggle GPU Runner ===", flush=True)
    print("=" * 70, flush=True)

    # 1. System & GPU Diagnosis
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"PyTorch Version: {torch.__version__}", flush=True)
        print(f"CUDA Available:  {cuda_avail}", flush=True)
        if cuda_avail:
            print(f"Device Name:     {torch.cuda.get_device_name(0)}", flush=True)
            print(f"Device Count:    {torch.cuda.device_count()}", flush=True)
        else:
            print("[WARNING] CUDA is NOT available. Running on CPU.", flush=True)
    except Exception as e:
        print(f"[!] PyTorch diagnostic error: {e}", flush=True)

    # 2. Check Working Directory
    work_dir = os.path.abspath("/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd())
    repo_dir = os.path.join(work_dir, "repo")
    print(f"Working Directory: {work_dir}", flush=True)
    print(f"Target Repo Path:  {repo_dir}", flush=True)

    # 3. Clone Repository
    if os.path.exists(repo_dir):
        print(f"Cleaning existing directory {repo_dir}...", flush=True)
        shutil.rmtree(repo_dir)

    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not github_token:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            github_token = user_secrets.get_secret("GITHUB_TOKEN")
            print("[+] Retrieved GITHUB_TOKEN from Kaggle Secrets.", flush=True)
        except Exception:
            pass

    if github_token:
        clone_url = f"https://oauth2:{github_token}@github.com/Victorvkk38596/privacy-llm-finetuning.git"
        print("[+] Cloning via authenticated HTTPS...", flush=True)
    else:
        clone_url = "https://github.com/Victorvkk38596/privacy-llm-finetuning.git"
        print("[+] Cloning via public HTTPS...", flush=True)

    try:
        run_cmd(f"git clone {clone_url} {repo_dir}")
    except RuntimeError:
        print("\n[!] Git clone failed. If the repo is private, either:", flush=True)
        print("    1. Make the repository public on GitHub, OR", flush=True)
        print("    2. Add 'GITHUB_TOKEN' to your Kaggle Secrets (Add-ons -> Secrets).", flush=True)
        sys.exit(1)

    # 4. Install Dependencies
    print("\n[+] Installing required dependencies from requirements.txt...", flush=True)
    req_file = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(req_file):
        run_cmd(f"{sys.executable} -m pip install -q -r {req_file}")
    else:
        run_cmd(f"{sys.executable} -m pip install -q opacus datasets accelerate")

    # 5. Execute Experiment
    # Forward sys.argv if provided, or use first GPU run defaults
    if len(sys.argv) > 1:
        exp_args = " ".join(sys.argv[1:])
    else:
        # Default first GPU run: Head Only with DP (eps=5.0)
        exp_args = "--train_layers head --use_dp --target_epsilon 5.0 --max_train_samples 1024 --epochs 2 --batch_size 32 --exp_name kaggle_gpu_head_eps5"

    print(f"\n[+] Executing: run_experiment.py {exp_args}\n", flush=True)
    run_cmd(f"{sys.executable} run_experiment.py {exp_args}", cwd=repo_dir)

    # 6. Preserve and Display Results
    repo_results = os.path.join(repo_dir, "results", "summary.csv")
    out_results_dir = os.path.join(work_dir, "results")
    out_results_csv = os.path.join(out_results_dir, "summary.csv")
    root_summary_csv = os.path.join(work_dir, "summary.csv")

    os.makedirs(out_results_dir, exist_ok=True)
    if os.path.exists(repo_results):
        shutil.copy(repo_results, out_results_csv)
        shutil.copy(repo_results, root_summary_csv)
        print("\n" + "=" * 70, flush=True)
        print(f"[✓] Experiment Completed Successfully!", flush=True)
        print(f"[✓] Results copied to: {out_results_csv}", flush=True)
        print(f"[✓] Results copied to: {root_summary_csv}", flush=True)
        print("=" * 70, flush=True)
        with open(out_results_csv, "r") as f:
            print("\nSummary Results CSV Content:\n", flush=True)
            print(f.read(), flush=True)
    else:
        print("[!] Warning: results/summary.csv was not found.", flush=True)

if __name__ == "__main__":
    main()
