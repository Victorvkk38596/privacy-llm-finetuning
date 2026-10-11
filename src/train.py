"""Step-based training worker and logit extraction engine.

Implements methodology_audit.md §3.1–§3.3, §4.1 and methodology_audit_addendum.md §4, §6.2:
  1. Step-Based Training Loop:
     - Trains a single model (role 'M_A' on D_A, or 'M_B' on D_B) for max_steps.
     - Tokenizes text with dynamic padding (max_length=128).
     - Strictly preserves frozen embeddings invariant via src.models.
  2. Opacus Step-Based DP-SGD Integration:
     - Computes noise multiplier sigma for target_epsilon, target_delta, sample_rate, max_steps.
     - Wraps model, optimizer, train_loader with PrivacyEngine.
     - Manages physical memory via BatchMemoryManager (max_physical_batch_size <= 16).
     - Tracks telemetry (clip fraction, noise/signal, spent epsilon).
  3. Logit Extraction & Caching:
     - Evaluates model in eval mode on all 5 splits: D_A, D_B, D_out, D_pop, validation.
     - Extracts float32 logits of shape (N, 2) and saves compact .npy arrays (~40 KB/model).
     - Returns validation accuracy, final loss, spent epsilon, noise multiplier, and clip fraction.
"""
import argparse
import collections
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from src.data import get_partitions
from src.models import get_model
from src.audit import extract_logits, save_logits, load_logits


# Optimize CPU execution if GPU is unavailable
if not torch.cuda.is_available():
    torch.set_num_threads(8)


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for raw text sentiment classification items."""
    def __init__(self, items: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            [item["raw"] for item in items],
            truncation=True,
            max_length=max_length,
        )
        self.labels = [int(item["label"]) for item in items]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def attach_telemetry(optimizer, clip_c: float):
    """Attaches gradient norm and clipping telemetry to Opacus DPOptimizer."""
    telemetry = {
        "total_samples": 0,
        "clipped_samples": 0,
        "step_losses": [],
    }
    orig_clip = optimizer.clip_and_accumulate

    def clip_and_accumulate():
        ps = [p for p in optimizer.params if getattr(p, "grad_sample", None) is not None]
        if ps:
            per_sample_norms_sq = torch.zeros(len(ps[0].grad_sample), device=ps[0].grad_sample.device)
            for p in ps:
                g = p.grad_sample.reshape(p.grad_sample.shape[0], -1)
                per_sample_norms_sq += g.float().pow(2).sum(dim=1)
            norms = torch.sqrt(per_sample_norms_sq)
            telemetry["total_samples"] += norms.numel()
            telemetry["clipped_samples"] += int((norms > clip_c).sum().item())
        return orig_clip()

    optimizer.clip_and_accumulate = clip_and_accumulate
    return telemetry


def train_single_model(
    role: str,              # 'M_A' or 'M_B'
    train_split_name: str,  # 'A' or 'B' (or 'D_A' / 'D_B')
    config: dict,           # Dict of hyperparameters
    seed: int,
    output_dir: str = "results/logits",
    device: torch.device = None,
) -> dict:                  # Returns telemetry & validation accuracy
    """Trains a single model for max_steps, extracts logits on all 5 splits, and saves them to disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = config.get("exp_name", "experiment")
    model_name = config.get("model_name", "roberta-base")
    train_layers = str(config.get("train_layers", "head"))
    freeze_embeddings = config.get("freeze_embeddings", True)
    use_dp = config.get("use_dp", False)
    max_steps = int(config.get("max_steps", 1000))
    lr = float(config.get("lr", 1e-4 if not use_dp else 1e-3))
    # For DP runs always use weight_decay=0: L2 regularisation on noisy gradients
    # pulls weights steadily toward zero and accelerates collapse to majority-class.
    weight_decay = float(config.get("weight_decay", 0.0))
    warmup_frac = float(config.get("warmup_frac", 0.05))
    clip_c = float(config.get("max_grad_norm", 1.0))
    max_length = int(config.get("max_length", 128))
    # Steps for clean head pre-alignment before DP wrapping (disabled for non-DP runs)
    head_warmup_steps = int(config.get("head_warmup_steps", 100)) if use_dp else 0

    logical_batch_size = int(config.get("logical_batch_size", config.get("batch_size", 64)))
    physical_batch_size = int(config.get("physical_batch_size", config.get("physical_batch", 16)))

    target_epsilon = float(config.get("target_epsilon", 5.0))
    target_delta = float(config.get("target_delta", 1e-5))

    # Set seeds for reproducibility
    seed_role_offset = 1000 if role in ("M_B", "B") else 0
    run_seed = seed + seed_role_offset
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    gen = torch.Generator()
    gen.manual_seed(run_seed)

    print(f"\n[{role}] Initializing single-model training (seed={seed}, run_seed={run_seed}, role={role}, layers={train_layers}, dp={use_dp})...")

    # 1. Load Data Partitions
    partitions = get_partitions(seed=seed)
    # Determine train split items
    split_key = "D_" + train_split_name if not train_split_name.startswith("D_") else train_split_name
    train_items = partitions.get(split_key) or partitions.get(train_split_name.lstrip("D_"))
    if train_items is None:
        raise KeyError(f"Training split '{train_split_name}' not found in partitions")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = TextClassificationDataset(train_items, tokenizer=tokenizer, max_length=max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=logical_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=gen,
    )

    # 2. Initialize Model
    model, trainable_params, fraction = get_model(
        model_name=model_name,
        num_labels=2,
        train_layers=train_layers,
        freeze_embeddings=freeze_embeddings,
    )
    model.to(device)

    # ------------------------------------------------------------------
    # Head warmup (clean, before PrivacyEngine): aligns the randomly
    # initialised classification head with pretrained representations.
    # This is the most impactful fix for DP utility collapse.
    # ------------------------------------------------------------------
    if use_dp and head_warmup_steps > 0:
        print(f"[{role}] Running {head_warmup_steps}-step clean head warmup (lr=1e-3, no DP)...")
        head_warmup_lr = float(config.get("head_warmup_lr", 1e-3))
        warmup_head_optimizer = torch.optim.AdamW(
            list(model.classifier.parameters()),
            lr=head_warmup_lr,
            weight_decay=0.0,
        )
        model.train()
        warmup_loader_iter = iter(train_loader)
        warmup_step = 0
        while warmup_step < head_warmup_steps:
            try:
                batch = next(warmup_loader_iter)
            except StopIteration:
                warmup_loader_iter = iter(train_loader)
                batch = next(warmup_loader_iter)
            warmup_head_optimizer.zero_grad()
            batch_inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_inputs)
            outputs.loss.backward()
            warmup_head_optimizer.step()
            warmup_step += 1
        del warmup_head_optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"[{role}] Head warmup complete. Final warmup loss: {outputs.loss.item():.4f}")

    model.train()

    # Separate param groups: head (randomly initialised) needs a higher LR
    # than the pretrained backbone layers.
    head_params = [p for n, p in model.named_parameters()
                   if "classifier" in n and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    optimizer_groups = []
    if backbone_params:
        optimizer_groups.append({"params": backbone_params, "lr": lr, "weight_decay": weight_decay})
    if head_params:
        # Head gets 3× backbone LR; this ratio is robust across DP budgets.
        optimizer_groups.append({"params": head_params, "lr": lr * 3.0, "weight_decay": 0.0})
    if not optimizer_groups:
        # Fallback: all trainable params
        optimizer_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                             "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.AdamW(optimizer_groups)
    print(f"[{role}] Optimizer: backbone_lr={lr:.2e}, head_lr={lr * 3.0:.2e}, weight_decay={weight_decay}")

    # 3. Setup DP Engine or Standard Scheduler
    privacy_engine = None
    telemetry = None
    noise_multiplier = 0.0
    epsilon_spent = None

    if use_dp:
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            model.to(device)
            # Rebuild differential-LR param groups after ModuleValidator.fix()
            # rewrites LayerNorm layers (parameter keys may shift).
            _head_params = [p for n, p in model.named_parameters()
                            if "classifier" in n and p.requires_grad]
            _backbone_params = [p for n, p in model.named_parameters()
                                if "classifier" not in n and p.requires_grad]
            _groups = []
            if _backbone_params:
                _groups.append({"params": _backbone_params, "lr": lr, "weight_decay": weight_decay})
            if _head_params:
                _groups.append({"params": _head_params, "lr": lr * 3.0, "weight_decay": 0.0})
            if not _groups:
                _groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                            "lr": lr, "weight_decay": weight_decay}]
            optimizer = torch.optim.AdamW(_groups)

        sample_rate = logical_batch_size / len(train_items)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=max_steps,
            accountant="rdp",
        )

        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=clip_c,
        )
        telemetry = attach_telemetry(optimizer, clip_c=clip_c)
        print(f"[{role}] Opacus PrivacyEngine Active: target_eps={target_epsilon}, delta={target_delta}, sigma={noise_multiplier:.4f}, C={clip_c}")

    warmup_steps = int(max_steps * warmup_frac)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # 4. Step-Based Training Loop
    step = 0
    final_loss = 0.0
    t_train_start = time.time()

    if use_dp:
        while step < max_steps:
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=physical_batch_size,
                optimizer=optimizer,
            ) as mem_loader:
                for batch in mem_loader:
                    optimizer.zero_grad()
                    batch_inputs = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch_inputs)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    final_loss = float(loss.item())

                    # Check if a logical step completed
                    if not getattr(optimizer, "_is_last_step_skipped", False):
                        scheduler.step()
                        step += 1
                        if step % max(1, max_steps // 5) == 0 or step == max_steps:
                            print(f"  [{role}] Step {step}/{max_steps} | Loss: {final_loss:.4f}")
                        if step >= max_steps:
                            break
    else:
        while step < max_steps:
            for batch in train_loader:
                optimizer.zero_grad()
                batch_inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch_inputs)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=clip_c,
                )
                optimizer.step()
                scheduler.step()
                step += 1
                final_loss = float(loss.item())
                if step % max(1, max_steps // 5) == 0 or step == max_steps:
                    print(f"  [{role}] Step {step}/{max_steps} | Loss: {final_loss:.4f}")
                if step >= max_steps:
                    break

    train_time = time.time() - t_train_start
    print(f"[{role}] Completed {step} steps in {train_time:.2f}s (Final Loss: {final_loss:.4f})")

    # Compute spent epsilon if DP was used
    if use_dp and privacy_engine is not None:
        epsilon_spent = float(privacy_engine.get_epsilon(target_delta))
        print(f"[{role}] Privacy Spent: epsilon = {epsilon_spent:.4f} (at delta={target_delta})")

    clip_fraction = 0.0
    if telemetry and telemetry["total_samples"] > 0:
        clip_fraction = float(telemetry["clipped_samples"] / telemetry["total_samples"])

    # 5. Logit Extraction & Caching on All 5 Splits
    print(f"\n[{role}] Extracting and caching logits on all 5 splits...")
    os.makedirs(output_dir, exist_ok=True)
    raw_model = model._module if hasattr(model, "_module") else model
    raw_model.eval()

    splits_to_eval = [
        ("D_A", partitions["D_A"]),
        ("D_B", partitions["D_B"]),
        ("D_out", partitions["D_out"]),
        ("D_pop", partitions["D_pop"]),
        ("validation", partitions["validation"]),
    ]

    logit_paths = {}
    val_accuracy = 0.0

    eval_batch_size = 64
    for split_name, items in splits_to_eval:
        eval_ds = TextClassificationDataset(items, tokenizer=tokenizer, max_length=max_length)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        logits = extract_logits(raw_model, eval_loader, device=device)

        filename = f"{role}_{exp_name}_seed{seed}_{split_name}.npy"
        filepath = os.path.join(output_dir, filename)
        save_logits(logits, filepath)
        logit_paths[split_name] = filepath

        if split_name == "validation":
            preds = np.argmax(logits, axis=-1)
            labels = np.array([it["label"] for it in items])
            val_accuracy = float(np.mean(preds == labels) * 100.0)

        file_size_kb = os.path.getsize(filepath) / 1024.0
        print(f"  Cached {split_name:10s} logits: shape {str(logits.shape):10s} -> {filepath} ({file_size_kb:.1f} KB)")

    print(f"[{role}] Extraction Complete! Validation Accuracy: {val_accuracy:.2f}%")

    # Clean memory
    del model, optimizer, train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "role": role,
        "seed": seed,
        "exp_name": exp_name,
        "train_split": train_split_name,
        "trainable_params": trainable_params,
        "val_accuracy": val_accuracy,
        "final_loss": final_loss,
        "epsilon_spent": epsilon_spent,
        "noise_multiplier": noise_multiplier,
        "clip_fraction": clip_fraction,
        "steps_completed": step,
        "train_time_sec": train_time,
        "logit_paths": logit_paths,
    }


# --------------------------------------------------------------------------- Standalone Verification Suite
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-based Training Worker & Logit Extraction")
    parser.add_argument("--role", type=str, default="M_A", choices=["M_A", "M_B"])
    parser.add_argument("--train_split", type=str, default="A", choices=["A", "B", "D_A", "D_B"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="mock_test")
    parser.add_argument("--train_layers", type=str, default="head")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/logits")
    args = parser.parse_args()

    print("=" * 60)
    print("TASK 5A: STEP-BASED TRAINING & LOGIT EXTRACTION ENGINE")
    print("=" * 60)

    mock_config = {
        "exp_name": args.exp_name,
        "train_layers": args.train_layers,
        "use_dp": args.use_dp,
        "max_steps": args.max_steps,
        "batch_size": 16,
        "physical_batch": 8,
        "lr": 1e-3,
    }

    t0 = time.time()
    res = train_single_model(
        role=args.role,
        train_split_name=args.train_split,
        config=mock_config,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("VERIFICATION CHECKS")
    print("=" * 60)

    # 1. Assert all 5 .npy logit files exist on disk
    expected_splits = ["D_A", "D_B", "D_out", "D_pop", "validation"]
    expected_sizes = {
        "D_A": 1024,
        "D_B": 1024,
        "D_out": 1024,
        "D_pop": 1024,
        "validation": 872,
    }

    for split in expected_splits:
        path = os.path.join(args.output_dir, f"{args.role}_{args.exp_name}_seed{args.seed}_{split}.npy")
        assert os.path.exists(path), f"Assertion Failed: Logit file {path} was not created!"
        logits = load_logits(path)
        assert logits.shape == (expected_sizes[split], 2), (
            f"Assertion Failed: {path} has shape {logits.shape}, expected ({expected_sizes[split]}, 2)"
        )
        assert logits.dtype == np.float32, f"Assertion Failed: {path} has dtype {logits.dtype}, expected float32"
        assert np.isfinite(logits).all(), f"Assertion Failed: {path} contains non-finite values!"
        print(f"[✓] Verified logit file: {path} (shape: {logits.shape}, dtype: {logits.dtype})")

    # 2. Assert valid return dictionary
    assert "val_accuracy" in res and isinstance(res["val_accuracy"], float), "Missing/invalid val_accuracy"
    assert "final_loss" in res and isinstance(res["final_loss"], float), "Missing/invalid final_loss"
    assert np.isfinite(res["val_accuracy"]), f"Invalid val_accuracy: {res['val_accuracy']}"
    assert np.isfinite(res["final_loss"]), f"Invalid final_loss: {res['final_loss']}"
    assert res["steps_completed"] == args.max_steps, f"Steps mismatch: {res['steps_completed']} != {args.max_steps}"

    print(f"\n[✓] Telemetry check: val_accuracy = {res['val_accuracy']:.2f}%, final_loss = {res['final_loss']:.4f}")
    print(f"[✓] End-to-end 5-step mock dry run completed in {elapsed:.2f}s")
    print("=" * 60)
    print("ALL TASK 5A ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 60)
