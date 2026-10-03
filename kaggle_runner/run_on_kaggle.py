import os
import sys
import shutil
import subprocess

FALLBACK_FILES = {
  "requirements.txt": "torch\ntransformers\ndatasets\nopacus\nscikit-learn\nnumpy\npandas\naccelerate",
  "src/__init__.py": "",
  "src/models.py": "import torch\nfrom transformers import AutoModelForSequenceClassification\n\ndef get_model(model_name: str, num_labels: int, train_layers: str):\n    \"\"\"\n    Loads model and selectively freezes parameters.\n    train_layers:\n      - 'all': All parameters trainable\n      - 'head': Only classification head trainable\n      - '9,10,11': Only specified transformer layers + classification head\n    \"\"\"\n    model = AutoModelForSequenceClassification.from_pretrained(\n        model_name,\n        num_labels=num_labels\n    )\n    \n    total_params = sum(p.numel() for p in model.parameters())\n\n    if train_layers == \"all\":\n        pass\n    else:\n        # 1. Freeze all backbone parameters\n        for param in model.parameters():\n            param.requires_grad = False\n            \n        # 2. Always unfreeze the classification head\n        if hasattr(model, \"classifier\"):\n            for param in model.classifier.parameters():\n                param.requires_grad = True\n\n        # 3. Unfreeze specific layer indices (e.g. \"0,1,2\" or \"9,10,11\")\n        if train_layers != \"head\":\n            selected_indices = [int(i.strip()) for i in train_layers.split(\",\")]\n            for idx in selected_indices:\n                layer_module = model.roberta.encoder.layer[idx]\n                for param in layer_module.parameters():\n                    param.requires_grad = True\n\n    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n    fraction = (trainable_params / total_params) * 100\n\n    print(\"=\" * 60)\n    print(f\"Model: {model_name} | Configuration: {train_layers}\")\n    print(f\"Total Parameters:     {total_params:,}\")\n    print(f\"Trainable Parameters: {trainable_params:,} ({fraction:.2f}%)\")\n    print(\"=\" * 60)\n\n    return model, trainable_params, fraction\n",
  "src/audit.py": "import torch\nimport numpy as np\nfrom sklearn.metrics import roc_auc_score\nfrom torch.utils.data import DataLoader\n\ndef compute_sample_losses(model, dataset, device, batch_size=32, collate_fn=None):\n    \"\"\"Computes cross-entropy loss for each individual sample.\"\"\"\n    model.eval()\n    criterion = torch.nn.CrossEntropyLoss(reduction=\"none\")\n    losses = []\n\n    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)\n    with torch.no_grad():\n        for batch in dataloader:\n            input_ids = batch[\"input_ids\"].to(device)\n            attention_mask = batch[\"attention_mask\"].to(device)\n            labels = (batch[\"labels\"] if \"labels\" in batch else batch[\"label\"]).to(device)\n\n            outputs = model(input_ids=input_ids, attention_mask=attention_mask)\n            loss = criterion(outputs.logits, labels)\n            losses.extend(loss.cpu().tolist())\n\n    return np.array(losses)\n\ndef run_mia_audit(model, train_subset, test_subset, device, collate_fn=None):\n    \"\"\"\n    Evaluates empirical privacy leakage via Membership Inference.\n    Members (train) = 1, Non-members (test) = 0.\n    Attacker score = -loss (Lower loss indicates training member).\n    AUC = 0.50 means random guessing (Completely Safe).\n    AUC = 1.00 means total privacy leak (Unsafe).\n    \"\"\"\n    print(\"\\n[+] Running Membership Inference Attack (MIA) Audit...\")\n    train_losses = compute_sample_losses(model, train_subset, device, collate_fn=collate_fn)\n    test_losses = compute_sample_losses(model, test_subset, device, collate_fn=collate_fn)\n\n    y_true = np.concatenate([np.ones_like(train_losses), np.zeros_like(test_losses)])\n    # Attacker predicts member if loss is low -> score = -loss\n    scores = np.concatenate([-train_losses, -test_losses])\n\n    mia_auc = roc_auc_score(y_true, scores)\n    print(f\"[+] Audit Finished -> Empirical MIA AUC: {mia_auc:.4f}\")\n    return mia_auc\n",
  "run_experiment.py": "import argparse\nimport os\nimport torch\nimport pandas as pd\nfrom datasets import load_dataset\nfrom transformers import AutoTokenizer, DataCollatorWithPadding\nfrom torch.utils.data import DataLoader\nfrom opacus import PrivacyEngine\nfrom opacus.validators import ModuleValidator\n\nfrom src.models import get_model\nfrom src.audit import run_mia_audit\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\"--model_name\", type=str, default=\"roberta-base\")\n    parser.add_argument(\"--train_layers\", type=str, default=\"all\", help=\"'all', 'head', or '9,10,11'\")\n    parser.add_argument(\"--use_dp\", action=\"store_true\", help=\"Enable DP-SGD\")\n    parser.add_argument(\"--target_epsilon\", type=float, default=5.0)\n    parser.add_argument(\"--target_delta\", type=float, default=1e-5)\n    parser.add_argument(\"--max_grad_norm\", type=float, default=1.0)\n    parser.add_argument(\"--epochs\", type=int, default=2)\n    parser.add_argument(\"--batch_size\", type=int, default=32)\n    parser.add_argument(\"--lr\", type=float, default=2e-5)\n    parser.add_argument(\"--max_train_samples\", type=int, default=1024, help=\"Subsample as in DP-SelFT\")\n    parser.add_argument(\"--exp_name\", type=str, default=\"test_run\")\n    args = parser.parse_args()\n\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    print(f\"Active Device: {device}\")\n\n    # 1. Dataset & Tokenization (SST-2)\n    print(\"\\n[1/5] Loading and tokenizing SST-2...\")\n    tokenizer = AutoTokenizer.from_pretrained(args.model_name)\n    raw_dataset = load_dataset(\"nyu-mll/glue\", \"sst2\")\n\n    def tokenize_fn(examples):\n        return tokenizer(examples[\"sentence\"], truncation=True, max_length=128)\n\n    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)\n    tokenized_dataset = tokenized_dataset.remove_columns([\"sentence\", \"idx\"])\n    tokenized_dataset.set_format(\"torch\")\n\n    # Subsample training data\n    n_train = min(args.max_train_samples, len(tokenized_dataset[\"train\"]))\n    train_dataset = tokenized_dataset[\"train\"].select(range(n_train))\n    val_dataset = tokenized_dataset[\"validation\"]\n\n    collator = DataCollatorWithPadding(tokenizer=tokenizer)\n    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)\n    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)\n\n    # 2. Model Setup\n    print(\"\\n[2/5] Initializing Model...\")\n    model, trainable_params, fraction = get_model(args.model_name, num_labels=2, train_layers=args.train_layers)\n    model.to(device)\n    model.train()\n\n    # 3. Privacy Engine (Opacus)\n    privacy_engine = None\n    if args.use_dp:\n        print(\"\\n[3/5] Setting up Differential Privacy (Opacus)...\")\n        if not ModuleValidator.is_valid(model):\n            model = ModuleValidator.fix(model)\n            model.to(device)\n\n        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)\n        privacy_engine = PrivacyEngine()\n        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(\n            module=model,\n            optimizer=optimizer,\n            data_loader=train_loader,\n            target_epsilon=args.target_epsilon,\n            target_delta=args.target_delta,\n            epochs=args.epochs,\n            max_grad_norm=args.max_grad_norm,\n        )\n        print(f\"PrivacyEngine active: Target eps={args.target_epsilon}, delta={args.target_delta}\")\n    else:\n        print(\"\\n[3/5] Standard optimization (No DP)...\")\n        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)\n\n    # 4. Training Loop\n    print(\"\\n[4/5] Starting Training...\")\n    criterion = torch.nn.CrossEntropyLoss()\n    model.train()\n    for epoch in range(args.epochs):\n        total_loss = 0.0\n        for step, batch in enumerate(train_loader):\n            optimizer.zero_grad()\n            input_ids = batch[\"input_ids\"].to(device)\n            attention_mask = batch[\"attention_mask\"].to(device)\n            labels = (batch[\"labels\"] if \"labels\" in batch else batch[\"label\"]).to(device)\n\n            outputs = model(input_ids=input_ids, attention_mask=attention_mask)\n            loss = criterion(outputs.logits, labels)\n            loss.backward()\n            optimizer.step()\n\n            total_loss += loss.item()\n\n        avg_loss = total_loss / len(train_loader)\n        eps_str = f\" | Epsilon spent: {privacy_engine.get_epsilon(args.target_delta):.2f}\" if args.use_dp else \"\"\n        print(f\"  Epoch {epoch + 1}/{args.epochs} - Training Loss: {avg_loss:.4f}{eps_str}\")\n\n    # 5. Evaluate Utility & Privacy\n    print(\"\\n[5/5] Evaluating Utility & Auditing Privacy...\")\n    model.eval()\n    correct, total = 0, 0\n    with torch.no_grad():\n        for batch in val_loader:\n            input_ids = batch[\"input_ids\"].to(device)\n            attention_mask = batch[\"attention_mask\"].to(device)\n            labels = (batch[\"labels\"] if \"labels\" in batch else batch[\"label\"]).to(device)\n            outputs = model(input_ids=input_ids, attention_mask=attention_mask)\n            preds = torch.argmax(outputs.logits, dim=1)\n            correct += (preds == labels).sum().item()\n            total += labels.size(0)\n\n    val_accuracy = (correct / total) * 100\n    print(f\"Validation Accuracy: {val_accuracy:.2f}%\")\n\n    # Run MIA audit\n    raw_model = model._module if args.use_dp else model\n    audit_train = train_dataset.select(range(min(200, len(train_dataset))))\n    audit_test = val_dataset.select(range(min(200, len(val_dataset))))\n    mia_auc = run_mia_audit(raw_model, audit_train, audit_test, device, collate_fn=collator)\n\n    # 6. Save to CSV\n    os.makedirs(\"results\", exist_ok=True)\n    csv_path = \"results/summary.csv\"\n    record = {\n        \"exp_name\": args.exp_name,\n        \"train_layers\": args.train_layers,\n        \"trainable_params\": trainable_params,\n        \"param_fraction_%\": round(fraction, 2),\n        \"use_dp\": args.use_dp,\n        \"target_epsilon\": args.target_epsilon if args.use_dp else \"None\",\n        \"val_accuracy\": round(val_accuracy, 2),\n        \"mia_auc\": round(mia_auc, 4),\n    }\n\n    df = pd.DataFrame([record])\n    if not os.path.exists(csv_path):\n        df.to_csv(csv_path, index=False)\n    else:\n        df.to_csv(csv_path, mode=\"a\", header=False, index=False)\n\n    print(f\"\\n[\u2713] Results logged successfully to: {csv_path}\\n\")\n\nif __name__ == \"__main__\":\n    main()\n"
}

def run_cmd(cmd, cwd=None):
    print(f"\n[EXEC] {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")

def extract_fallback_files(dest_dir):
    print(f"\n[+] Extracting embedded codebase into {dest_dir}...", flush=True)
    for rel_path, content in FALLBACK_FILES.items():
        full_path = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
    print(f"[✓] Codebase extracted successfully ({len(FALLBACK_FILES)} files).", flush=True)

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

    # 3. Clone Repository or Fallback
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

    cloned_ok = False
    try:
        run_cmd(f"git clone {clone_url} {repo_dir}")
        cloned_ok = True
    except RuntimeError:
        print("\n[!] Git clone failed. Falling back to embedded codebase...", flush=True)
        print("    (To clone directly from GitHub, make the repo public or add GITHUB_TOKEN to Kaggle Secrets.)", flush=True)
        extract_fallback_files(repo_dir)

    # 4. Install Dependencies
    print("\n[+] Installing required dependencies...", flush=True)
    req_file = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(req_file):
        run_cmd(f"{sys.executable} -m pip install -q -r {req_file}")
    else:
        run_cmd(f"{sys.executable} -m pip install -q opacus datasets accelerate")

    # 5. Execute Experiment
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
