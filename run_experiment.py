"""Master subprocess orchestrator for swap-pair training and bidirectional MIA cross-auditing.

Implements methodology_audit.md §3.1–§3.3, §4.1 and methodology_audit_addendum.md §4, §6.2:
  1. Subprocess Isolation:
     - Model M_A is trained on partition D_A in a clean subprocess, saves logits, and exits.
     - Model M_B is trained on partition D_B in a clean subprocess, saves logits, and exits.
     - Guarantees full memory release and zero PyTorch/CUDA leakage across swap pairs.
  2. CPU Post-Processing & Bidirectional Cross-Audit:
     - Loads compact cached logits for M_A and M_B across all 5 splits.
     - Audit 1: Target = M_A, Shadow Reference = M_B, Members = D_A, Non-members = D_out, Population = D_pop.
     - Audit 2: Target = M_B, Shadow Reference = M_A, Members = D_B, Non-members = D_out, Population = D_pop.
  3. Results Logging:
     - Appends two 22-column summary rows to --summary_csv (default results/grid/summary_rows.csv;
       the historical pilot CSVs in results/ are never written by default).
     - Writes one JSON record per model to --records_dir (default results/grid/records), holding
       seed, swap-pair role, partition hash / manifest, trainable layers and parameter counts,
       training settings, target and accounted epsilon, delta, accounting-order ledger, validation
       accuracy, RMIA AUC and TPR at 1% FPR. Failed runs also get a record (status="failed").
"""
import argparse
import csv
import json
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml

from src.data import get_partitions, export_manifest, partition_hash
from src.audit import evaluate_attacks, load_logits
from src.train import train_single_model

RECORD_SCHEMA = "grid-record-v1"

# --------------------------------------------------------------------------- Schema Contract
CSV_COLUMNS = [
    "exp_name",
    "train_layers",
    "trainable_params",
    "param_fraction_%",
    "use_dp",
    "target_epsilon",
    "val_accuracy",
    "mia_auc",
    "seed",
    "model_name",
    "n_train",
    "epochs",
    "batch_size",
    "lr",
    "target_delta",
    "max_grad_norm",
    "noise_multiplier",
    "epsilon_spent",
    "final_train_loss",
    "n_audit_in",
    "n_audit_out",
    "mia_tpr_at_1fpr",
]


def _worker_process(role: str, train_split: str, config: dict, seed: int, output_dir: str, queue: multiprocessing.Queue):
    """Isolated subprocess worker executing single-model training and logit extraction."""
    try:
        telemetry = train_single_model(
            role=role,
            train_split_name=train_split,
            config=config,
            seed=seed,
            output_dir=output_dir,
        )
        queue.put(("SUCCESS", telemetry))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        queue.put(("ERROR", str(e), tb))


def build_summary_row(
    role: str,
    exp_name: str,
    config: dict,
    seed: int,
    telemetry: dict,
    attack_metrics: dict,
    n_members: int = 1024,
    n_nonmembers: int = 1024,
) -> dict:
    """Builds a single validated 22-column row for results/summary.csv."""
    trainable_params = int(telemetry.get("trainable_params", 0))
    total_params = 124647170
    param_frac = round(100.0 * trainable_params / total_params, 2)
    use_dp = bool(config.get("use_dp", False))
    target_eps = config.get("target_epsilon", "None") if use_dp else "None"
    val_acc = round(float(telemetry.get("val_accuracy", 0.0)), 2)
    mia_auc = round(float(attack_metrics.get("rmia_auc", attack_metrics.get("loss_auc", 0.5))), 4)
    max_steps = int(config.get("max_steps", 1000))
    batch_size = int(config.get("logical_batch_size", config.get("batch_size", 64)))
    epochs = round((max_steps * batch_size) / n_members, 1)

    eps_spent = telemetry.get("epsilon_spent")
    if eps_spent is not None and use_dp:
        eps_spent_str = f"{float(eps_spent):.4f}"
    else:
        eps_spent_str = ""

    noise_mult = telemetry.get("noise_multiplier", 0.0)
    noise_mult_str = f"{float(noise_mult):.4f}" if use_dp else ""
    target_delta = f"{float(config.get('target_delta', 1e-5)):.1e}" if use_dp else ""
    max_grad_norm = f"{float(config.get('max_grad_norm', 1.0)):.2f}" if use_dp else ""

    row = {
        "exp_name": f"{exp_name}_{role}",
        "train_layers": str(config.get("train_layers", "all")),
        "trainable_params": trainable_params,
        "param_fraction_%": param_frac,
        "use_dp": use_dp,
        "target_epsilon": target_eps,
        "val_accuracy": val_acc,
        "mia_auc": mia_auc,
        "seed": seed,
        "model_name": config.get("model_name", "roberta-base"),
        "n_train": n_members,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": float(config.get("lr", 1e-4)),
        "target_delta": target_delta,
        "max_grad_norm": max_grad_norm,
        "noise_multiplier": noise_mult_str,
        "epsilon_spent": eps_spent_str,
        "final_train_loss": round(float(telemetry.get("final_loss", 0.0)), 4),
        "n_audit_in": n_members,
        "n_audit_out": n_nonmembers,
        "mia_tpr_at_1fpr": round(float(attack_metrics.get("rmia_tpr_at_1fpr", 0.0)), 4),
    }
    return row


def append_to_summary_csv(csv_path: str, rows: List[dict]):
    """Safely appends rows to results/summary.csv, archiving incompatible legacy headers."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    need_header = True

    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        existing_cols = [c.strip() for c in first_line.split(",") if c.strip()]
        if existing_cols != CSV_COLUMNS:
            # Never overwrite an existing archive (results/summary_legacy.csv is a pilot record).
            base = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "summary_legacy")
            legacy_path = base + ".csv"
            if os.path.exists(legacy_path):
                legacy_path = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            print(f"[!] Warning: {csv_path} header does not match 22-column schema v2.")
            print(f"    Archiving existing summary to {legacy_path}")
            shutil.copy2(csv_path, legacy_path)
            need_header = True
            mode = "w"
        else:
            need_header = False
            mode = "a"
    else:
        mode = "w"

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if need_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[✓] Appended {len(rows)} verified 22-column summary rows to {csv_path}")


def _env_info() -> dict:
    info = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "opacus", "datasets", "numpy", "sklearn"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["device"] = None
    try:
        import subprocess
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(project_root),
                                            capture_output=True, text=True).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info


def record_path(records_dir: str, exp_name: str, seed: int, role: str) -> str:
    return os.path.join(records_dir, f"{exp_name}__seed{seed}__{role}.json")


def build_record(role: str, config: dict, seed: int, telemetry: dict, attack_metrics: dict,
                 manifest_path: str, wall_time_sec: float, env: dict) -> dict:
    use_dp = bool(config.get("use_dp", False))
    other = "M_B" if role == "M_A" else "M_A"
    return {
        "schema": RECORD_SCHEMA,
        "status": "ok",
        "exp_name": config.get("exp_name"),
        "config_name": config.get("config_name", config.get("exp_name")),
        "comparisons": config.get("comparisons", []),
        "seed": int(seed),
        "role": role,
        "train_split": "D_A" if role == "M_A" else "D_B",
        "audit": {"members": "D_A" if role == "M_A" else "D_B", "nonmembers": "D_out",
                  "population": "D_pop", "reference_model": other},
        "partition_hash": telemetry.get("partition_hash"),
        "manifest_path": manifest_path,
        "train_layers_spec": str(config.get("train_layers")),
        "trainable_blocks": telemetry.get("trainable_blocks"),
        "trainable_groups": telemetry.get("trainable_groups"),
        "trainable_params": int(telemetry.get("trainable_params", 0)),
        "total_params": telemetry.get("total_params"),
        "trainable_fraction_pct": telemetry.get("trainable_fraction_pct"),
        "training": {
            "model_name": config.get("model_name", "roberta-base"),
            "use_dp": use_dp,
            "freeze_embeddings": bool(config.get("freeze_embeddings", True)),
            "optimizer": "AdamW",
            "lr_backbone": float(config.get("lr")),
            "lr_head": telemetry.get("head_lr"),
            "weight_decay": float(config.get("weight_decay", 0.0)),
            "lr_schedule": f"linear, {float(config.get('warmup_frac', 0.05)):g} warmup fraction",
            "max_steps": int(config.get("max_steps")),
            "steps_completed": telemetry.get("steps_completed"),
            "logical_batch_size": int(config.get("logical_batch_size", 64)),
            "physical_batch_size": int(config.get("physical_batch_size", 16)),
            "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
            "max_length": int(config.get("max_length", 128)),
            "head_warmup": telemetry.get("head_warmup"),
        },
        "privacy": {
            "target_epsilon": float(config["target_epsilon"]) if use_dp else None,
            "target_delta": float(config.get("target_delta", 1e-5)) if use_dp else None,
            "accounted_epsilon": telemetry.get("epsilon_spent"),
            "accountant": telemetry.get("accountant"),
            "noise_multiplier": float(telemetry.get("noise_multiplier", 0.0)) if use_dp else None,
            "sample_rate": telemetry.get("sample_rate"),
            "accounting_ledger": telemetry.get("privacy_ledger"),
        },
        "metrics": {
            "val_accuracy": telemetry.get("val_accuracy"),
            "val_pred_pos_rate": telemetry.get("val_pred_pos_rate"),
            **{k: float(v) for k, v in attack_metrics.items()},
        },
        "final_train_loss": telemetry.get("final_loss"),
        "clip_fraction": telemetry.get("clip_fraction"),
        "train_time_sec": telemetry.get("train_time_sec"),
        "wall_time_sec": wall_time_sec,
        "logit_paths": telemetry.get("logit_paths"),
        "env": env,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_record(records_dir: str, rec: dict, role: str) -> str:
    os.makedirs(records_dir, exist_ok=True)
    path = record_path(records_dir, rec["exp_name"], rec["seed"], role)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return path


def write_failure_records(records_dir: str, config: dict, seed: int, error: str, roles=("M_A", "M_B")) -> None:
    for role in roles:
        rec = {
            "schema": RECORD_SCHEMA, "status": "failed", "exp_name": config.get("exp_name"),
            "config_name": config.get("config_name", config.get("exp_name")),
            "comparisons": config.get("comparisons", []), "seed": int(seed), "role": role,
            "train_layers_spec": str(config.get("train_layers")),
            "privacy": {"target_epsilon": config.get("target_epsilon") if config.get("use_dp") else None,
                        "target_delta": config.get("target_delta") if config.get("use_dp") else None},
            "training": {"use_dp": bool(config.get("use_dp")),
                         "head_warmup": {"source": config.get("head_warmup_source", "none")}},
            "error": error[-4000:], "env": _env_info(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_record(records_dir, rec, role)


def ensure_manifest(seed: int, out_dir: str = "manifests") -> str:
    """Exports manifests/manifest_seed{seed}.json if missing; never overwrites an existing one."""
    path = os.path.join(project_root, out_dir, f"manifest_seed{seed}.json")
    part = get_partitions(seed=seed)
    core = {k: part[k] for k in ("A", "B", "out", "pop", "tune")}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f).get("partition_hash")
        if existing != partition_hash(core):
            raise RuntimeError(f"Partition for seed {seed} ({partition_hash(core)}) does not match {path} ({existing})")
        return path
    return export_manifest(core, seed=seed, out_dir=os.path.join(project_root, out_dir))


def run_orchestrator(config: dict, seed: int = 42, output_dir: str = "results/logits",
                     summary_csv: str = "results/grid/summary_rows.csv",
                     records_dir: str = "results/grid/records") -> dict:
    """Executes the swap pair in isolated subprocesses and performs bidirectional cross-auditing."""
    try:
        return _run_orchestrator(config, seed, output_dir, summary_csv, records_dir)
    except Exception as e:
        import traceback
        write_failure_records(records_dir, config, seed, f"{e}\n{traceback.format_exc()}")
        raise


def _run_orchestrator(config: dict, seed: int, output_dir: str, summary_csv: str, records_dir: str) -> dict:
    exp_name = config.get("exp_name", "experiment")
    t_wall = time.time()
    manifest_path = ensure_manifest(seed)
    env = _env_info()
    print("\n" + "=" * 70)
    print(f"MASTER SUBPROCESS ORCHESTRATOR: {exp_name} (Seed {seed})")
    print(f"Layers: {config.get('train_layers')} | DP: {config.get('use_dp')} | Max Steps: {config.get('max_steps')}")
    print("=" * 70)

    ctx = multiprocessing.get_context("spawn")

    # 1. Subprocess M_A
    print(f"\n[1/3] Spawning Subprocess for Target Model M_A (train on D_A)...")
    q_A = ctx.Queue()
    p_A = ctx.Process(target=_worker_process, args=("M_A", "A", config, seed, output_dir, q_A))
    t0_A = time.time()
    p_A.start()
    print(f"  Subprocess M_A active with PID: {p_A.pid}")
    status_A = q_A.get()
    p_A.join()
    elapsed_A = time.time() - t0_A
    if status_A[0] != "SUCCESS":
        raise RuntimeError(f"Subprocess M_A failed: {status_A[1]}\n{status_A[2]}")
    if p_A.exitcode != 0:
        raise RuntimeError(f"Subprocess M_A exited with non-zero code {p_A.exitcode}")
    res_A = status_A[1]
    print(f"[✓] Subprocess M_A completed in {elapsed_A:.2f}s and terminated cleanly.")

    # 2. Subprocess M_B
    print(f"\n[2/3] Spawning Subprocess for Target Model M_B (train on D_B)...")
    q_B = ctx.Queue()
    p_B = ctx.Process(target=_worker_process, args=("M_B", "B", config, seed, output_dir, q_B))
    t0_B = time.time()
    p_B.start()
    print(f"  Subprocess M_B active with PID: {p_B.pid}")
    status_B = q_B.get()
    p_B.join()
    elapsed_B = time.time() - t0_B
    if status_B[0] != "SUCCESS":
        raise RuntimeError(f"Subprocess M_B failed: {status_B[1]}\n{status_B[2]}")
    if p_B.exitcode != 0:
        raise RuntimeError(f"Subprocess M_B exited with non-zero code {p_B.exitcode}")
    res_B = status_B[1]
    print(f"[✓] Subprocess M_B completed in {elapsed_B:.2f}s and terminated cleanly.")

    # 3. CPU Post-Processing & Bidirectional Cross-Audit
    print(f"\n[3/3] CPU Post-Processing & Bidirectional Cross-Audit...")
    partitions = get_partitions(seed=seed)
    member_labels_A = np.array([it["label"] for it in partitions["D_A"]], dtype=np.int64)
    member_labels_B = np.array([it["label"] for it in partitions["D_B"]], dtype=np.int64)
    nonmember_labels = np.array([it["label"] for it in partitions["D_out"]], dtype=np.int64)
    pop_labels = np.array([it["label"] for it in partitions["D_pop"]], dtype=np.int64)

    # Load M_A logit caches
    logits_A_DA = load_logits(os.path.join(output_dir, f"M_A_{exp_name}_seed{seed}_D_A.npy"))
    logits_A_DB = load_logits(os.path.join(output_dir, f"M_A_{exp_name}_seed{seed}_D_B.npy"))
    logits_A_Dout = load_logits(os.path.join(output_dir, f"M_A_{exp_name}_seed{seed}_D_out.npy"))
    logits_A_Dpop = load_logits(os.path.join(output_dir, f"M_A_{exp_name}_seed{seed}_D_pop.npy"))

    # Load M_B logit caches
    logits_B_DA = load_logits(os.path.join(output_dir, f"M_B_{exp_name}_seed{seed}_D_A.npy"))
    logits_B_DB = load_logits(os.path.join(output_dir, f"M_B_{exp_name}_seed{seed}_D_B.npy"))
    logits_B_Dout = load_logits(os.path.join(output_dir, f"M_B_{exp_name}_seed{seed}_D_out.npy"))
    logits_B_Dpop = load_logits(os.path.join(output_dir, f"M_B_{exp_name}_seed{seed}_D_pop.npy"))

    # Audit 1: Target = M_A, Shadow Reference = M_B
    # Members = D_A, Non-members = D_out, Population = D_pop
    metrics_A = evaluate_attacks(
        target_logits=(logits_A_DA, logits_A_Dout),
        ref_logits=(logits_B_DA, logits_B_Dout),
        member_labels=member_labels_A,
        nonmember_labels=nonmember_labels,
        pop_target_logits=logits_A_Dpop,
        pop_ref_logits=logits_B_Dpop,
        pop_labels=pop_labels,
        alpha=0.0,
        gamma=1.0,
    )

    # Audit 2: Target = M_B, Shadow Reference = M_A
    # Members = D_B, Non-members = D_out, Population = D_pop
    metrics_B = evaluate_attacks(
        target_logits=(logits_B_DB, logits_B_Dout),
        ref_logits=(logits_A_DB, logits_A_Dout),
        member_labels=member_labels_B,
        nonmember_labels=nonmember_labels,
        pop_target_logits=logits_B_Dpop,
        pop_ref_logits=logits_A_Dpop,
        pop_labels=pop_labels,
        alpha=0.0,
        gamma=1.0,
    )

    print("\n" + "=" * 70)
    print(f"BIDIRECTIONAL CROSS-AUDIT METRICS: {exp_name} (Seed {seed})")
    print("=" * 70)
    print(f"Target M_A (Trained on D_A, Shadow Reference M_B):")
    print(f"  Val Accuracy:       {res_A['val_accuracy']:.2f}%")
    print(f"  Final Train Loss:   {res_A['final_loss']:.4f}")
    print(f"  RMIA AUC (Primary): {metrics_A['rmia_auc']:.4f}")
    print(f"  Loss AUC (Naive):   {metrics_A['loss_auc']:.4f}")
    print(f"  RMIA Quantile AUC:  {metrics_A['rmia_quant_auc']:.4f}")
    print(f"  RMIA TPR @ 5% FPR:  {metrics_A['rmia_tpr_at_5fpr']:.4f}")
    print(f"  RMIA TPR @ 1% FPR:  {metrics_A['rmia_tpr_at_1fpr']:.4f}")
    print(f"  Calibration Gap:    {metrics_A['calibration_gap']:.4f}")
    print(f"  Generalization Gap: {metrics_A['generalization_gap']:.4f}")

    print(f"\nTarget M_B (Trained on D_B, Shadow Reference M_A):")
    print(f"  Val Accuracy:       {res_B['val_accuracy']:.2f}%")
    print(f"  Final Train Loss:   {res_B['final_loss']:.4f}")
    print(f"  RMIA AUC (Primary): {metrics_B['rmia_auc']:.4f}")
    print(f"  Loss AUC (Naive):   {metrics_B['loss_auc']:.4f}")
    print(f"  RMIA Quantile AUC:  {metrics_B['rmia_quant_auc']:.4f}")
    print(f"  RMIA TPR @ 5% FPR:  {metrics_B['rmia_tpr_at_5fpr']:.4f}")
    print(f"  RMIA TPR @ 1% FPR:  {metrics_B['rmia_tpr_at_1fpr']:.4f}")
    print(f"  Calibration Gap:    {metrics_B['calibration_gap']:.4f}")
    print(f"  Generalization Gap: {metrics_B['generalization_gap']:.4f}")

    # Build and append the two summary rows
    row_A = build_summary_row("M_A", exp_name, config, seed, res_A, metrics_A)
    row_B = build_summary_row("M_B", exp_name, config, seed, res_B, metrics_B)
    append_to_summary_csv(summary_csv, [row_A, row_B])

    wall = time.time() - t_wall
    for role, res, met in (("M_A", res_A, metrics_A), ("M_B", res_B, metrics_B)):
        rec = build_record(role, config, seed, res, met, manifest_path, wall, env)
        led = (rec["privacy"].get("accounting_ledger") or {})
        if config.get("use_dp") and not (led.get("ordering_ok") and led.get("steps_match_ok")):
            rec["status"] = "privacy_check_failed"
        print(f"[✓] Record written: {write_record(records_dir, rec, role)} (status={rec['status']})")

    return {
        "exp_name": exp_name,
        "seed": seed,
        "M_A": {"telemetry": res_A, "metrics": metrics_A, "row": row_A},
        "M_B": {"telemetry": res_B, "metrics": metrics_B, "row": row_B},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Master Subprocess Orchestrator for Swap-Pair Fine-Tuning & MIA Audit")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--dry_run", action="store_true", help="Execute rapid dry run on CPU (5 steps).")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment identifier name.")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--train_layers", type=str, default=None, help="'all', 'head', or comma-separated indices e.g. '0,1,2'")
    parser.add_argument("--freeze_embeddings", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--use_dp", action="store_true", help="Enable Opacus DP-SGD.")
    parser.add_argument("--target_epsilon", type=float, default=None)
    parser.add_argument("--target_delta", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="Logical batch size")
    parser.add_argument("--logical_batch_size", type=int, default=None)
    parser.add_argument("--physical_batch_size", type=int, default=None)
    parser.add_argument("--physical_batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/logits")
    parser.add_argument("--summary_csv", type=str, default="results/grid/summary_rows.csv",
                        help="22-column row log. Default avoids touching the pilot CSVs in results/.")
    parser.add_argument("--records_dir", type=str, default="results/grid/records")
    parser.add_argument("--head_warmup_source", type=str, default=None,
                        help="Override config: 'none' or 'public_imdb'. Private-split warmup is not allowed.")

    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # CLI flags override YAML config
    if args.exp_name is not None:
        config["exp_name"] = args.exp_name
    elif "exp_name" not in config:
        if args.config:
            config["exp_name"] = Path(args.config).stem
        else:
            config["exp_name"] = "dry_run" if args.dry_run else "experiment"

    if args.train_layers is not None:
        config["train_layers"] = str(args.train_layers)
    elif "train_layers" not in config:
        config["train_layers"] = "head" if args.dry_run else "all"

    if args.use_dp or "use_dp" not in config:
        config["use_dp"] = args.use_dp or config.get("use_dp", False)

    if args.target_epsilon is not None:
        config["target_epsilon"] = args.target_epsilon
    elif "target_epsilon" not in config and config.get("use_dp"):
        config["target_epsilon"] = 5.0

    if args.target_delta is not None:
        config["target_delta"] = args.target_delta
    elif "target_delta" not in config:
        config["target_delta"] = 1.0e-5

    if args.max_grad_norm is not None:
        config["max_grad_norm"] = args.max_grad_norm
    elif "max_grad_norm" not in config:
        config["max_grad_norm"] = 1.0

    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    elif "max_steps" not in config:
        config["max_steps"] = 5 if args.dry_run else 1000

    b = args.logical_batch_size or args.batch_size
    if b is not None:
        config["logical_batch_size"] = b
    elif "logical_batch_size" not in config:
        config["logical_batch_size"] = 16 if args.dry_run else 64

    pb = args.physical_batch_size or args.physical_batch
    if pb is not None:
        config["physical_batch_size"] = pb
    elif "physical_batch_size" not in config:
        config["physical_batch_size"] = 8 if args.dry_run else 16

    if args.lr is not None:
        config["lr"] = args.lr
    elif "lr" not in config:
        config["lr"] = 1e-3 if args.dry_run else (3e-4 if config.get("use_dp") else 3e-5)

    if args.model_name is not None:
        config["model_name"] = args.model_name

    config["freeze_embeddings"] = args.freeze_embeddings
    if args.head_warmup_source is not None:
        config["head_warmup_source"] = args.head_warmup_source
    config.setdefault("head_warmup_source", "none")

    # Dry run presets
    if args.dry_run:
        if args.max_steps is None:
            config["max_steps"] = 5
        if b is None:
            config["logical_batch_size"] = 16
        if pb is None:
            config["physical_batch_size"] = 8

    return args, config


def main():
    args, config = parse_args()
    seed = args.seed
    output_dir = args.output_dir
    summary_csv = args.summary_csv

    t_start = time.time()
    results = run_orchestrator(
        config=config,
        seed=seed,
        output_dir=output_dir,
        summary_csv=summary_csv,
        records_dir=args.records_dir,
    )
    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print(f"SWAP-PAIR RUN COMPLETE in {total_time:.2f}s")
    print(f"Summary log: {summary_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
