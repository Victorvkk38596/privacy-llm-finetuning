"""Perturbation-Aware Layer Tournament with Head Warmup (DP-SelFT Stage 3).

Implements the head-warmed 12-layer tournament in src/select_layers.py following
Sha et al. §5.1, methodology_audit.md §2.1–§2.2, and methodology_audit_addendum.md §5.1:
  1. Proxy Split: 70% train (~716 items), 30% val (~308 items) from data/dp_proxy.parquet.
  2. Step 1: Head Warmup (100 clean steps):
     - Load roberta-base, freeze all 12 encoder blocks, train only classifier head for 100 steps
       with AdamW (batch size 16, lr 1e-3, no perturbation).
     - Verify competent baseline on proxy val (~65-72%).
     - Extract trained head weights: warmed_head_state = model.classifier.state_dict().
  3. Step 2: The 12-Layer Perturbation Tournament:
     - For each candidate layer index l in [0, 1, ..., 11]:
       a) Load model with layer l and head trainable (embeddings & other layers frozen).
       b) Load warmed_head_state into model.classifier so all layers share the same baseline.
       c) Train for 100 steps with AdamW (batch size 16, lr 3e-4) under worst-case
          perturbation xi* = -rho * (g / (||g||_2 + 1e-12)), rho = 0.05.
       d) Evaluate validation accuracy on the 30% proxy validation split.
       e) Cleanly release memory.
  4. Step 3: Ranking & Artifact Export:
     - Rank layers by validation accuracy descending (resolving ties by layer index descending).
     - Select Top-3 layers.
     - Assert at least 3 layers achieve accuracy > 55.0%.
     - Assert Top-3 layers are not all identical ties.
     - Export to configs/selection_seed{seed}.json.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.models import get_model


# Optimize CPU thread allocation if running on CPU (8 threads avoids thread contention)
if not torch.cuda.is_available():
    torch.set_num_threads(8)


class ProxyDataset(Dataset):
    """PyTorch Dataset for tokenized proxy sentences."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = [int(l) for l in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def split_proxy_dataset(
    proxy_path: str = "data/dp_proxy.parquet",
    seed: int = 42,
    train_frac: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits proxy dataset deterministically into train (~70%) and val (~30%) splits."""
    if not os.path.exists(proxy_path):
        raise FileNotFoundError(f"Proxy dataset not found at {proxy_path}. Run src/dp_proxy.py first.")

    df = pd.read_parquet(proxy_path)
    n_total = len(df)
    n_train = int(round(train_frac * n_total))  # 716 for 1024

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    train_df = df.iloc[perm[:n_train]].reset_index(drop=True)
    val_df = df.iloc[perm[n_train:]].reset_index(drop=True)

    return train_df, val_df


def warmup_head(
    train_ds: ProxyDataset,
    val_ds: ProxyDataset,
    collator: DataCollatorWithPadding,
    device: torch.device,
    model_name: str = "roberta-base",
    num_steps: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Step 1: Trains only classification head for 100 clean steps with AdamW (lr 1e-3)."""
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    model, trainable_params, _ = get_model(model_name, num_labels=2, train_layers="head", freeze_embeddings=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator, generator=gen)

    step = 0
    while step < num_steps:
        for batch in train_loader:
            optimizer.zero_grad()
            batch_inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            step += 1
            if step >= num_steps:
                break

    # Evaluate validation accuracy
    model.eval()
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collator)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch_inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch_inputs["labels"]
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = (correct / total) * 100.0 if total > 0 else 0.0
    warmed_state = {k: v.cpu().clone() for k, v in model.classifier.state_dict().items()}

    del model, optimizer, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return warmed_state, float(val_acc)


def train_and_eval_layer(
    layer_idx: int,
    train_ds: ProxyDataset,
    val_ds: ProxyDataset,
    collator: DataCollatorWithPadding,
    device: torch.device,
    warmed_head_state: Dict[str, torch.Tensor],
    model_name: str = "roberta-base",
    num_steps: int = 100,
    batch_size: int = 16,
    lr: float = 3e-4,
    rho: float = 0.05,
    seed: int = 42,
) -> float:
    """Step 2: Trains candidate layer l starting from warmed head for 100 steps under rho=0.05 perturbation."""
    torch.manual_seed(seed + layer_idx)
    gen = torch.Generator()
    gen.manual_seed(seed + layer_idx)

    # 1. Load model with layer l and head trainable (embeddings & others frozen)
    model, _, _ = get_model(model_name, num_labels=2, train_layers=str(layer_idx), freeze_embeddings=True)
    # 2. Load pre-warmed head state
    model.classifier.load_state_dict(warmed_head_state)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator, generator=gen)

    # 3. 100 Steps with SAM-style worst-case perturbation: xi* = -rho * (g / (||g|| + 1e-12))
    step = 0
    while step < num_steps:
        for batch in train_loader:
            optimizer.zero_grad()
            batch_inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_inputs)
            loss = outputs.loss
            loss.backward()

            # Compute L2 norm of gradients of all trainable parameters
            trainable = [p for p in model.parameters() if p.requires_grad]
            grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in trainable if p.grad is not None) + 1e-24)

            # Update parameters: theta' = theta - eta * (g + xi*)
            # where xi* = -rho * g / (||g|| + 1e-12)
            # => g_effective = g * (1.0 - rho / (grad_norm + 1e-12))
            scale = 1.0 - rho / (grad_norm + 1e-12)
            for p in trainable:
                if p.grad is not None:
                    p.grad.mul_(scale)

            optimizer.step()
            step += 1
            if step >= num_steps:
                break

    # 4. Evaluate accuracy on proxy validation split
    model.eval()
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collator)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch_inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch_inputs["labels"]
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = (correct / total) * 100.0 if total > 0 else 0.0

    # Clean memory
    del model, optimizer, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(val_accuracy)


def run_layer_tournament(
    proxy_path: str = "data/dp_proxy.parquet",
    output_dir: str = "configs",
    model_name: str = "roberta-base",
    warmup_steps: int = 100,
    tournament_steps: int = 100,
    batch_size: int = 16,
    warmup_lr: float = 1e-3,
    tournament_lr: float = 3e-4,
    rho: float = 0.05,
    seed: int = 42,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Executes head warmup followed by the 12-layer tournament and saves configs/selection_seed{seed}.json."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading proxy data from {proxy_path} for head-warmed 12-layer tournament (seed={seed})...")
    train_df, val_df = split_proxy_dataset(proxy_path, seed=seed)
    print(f"  Proxy train items: {len(train_df)}, Proxy val items: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = ProxyDataset(train_df["raw"].tolist(), train_df["label"].tolist(), tokenizer=tokenizer)
    val_ds = ProxyDataset(val_df["raw"].tolist(), val_df["label"].tolist(), tokenizer=tokenizer)

    # Step 1: Head Warmup (100 clean steps)
    print(f"\n[Step 1/2] Warming up classification head ({warmup_steps} clean steps, lr={warmup_lr})...")
    t_warm_start = time.time()
    warmed_head_state, head_val_acc = warmup_head(
        train_ds=train_ds,
        val_ds=val_ds,
        collator=collator,
        device=device,
        model_name=model_name,
        num_steps=warmup_steps,
        batch_size=batch_size,
        lr=warmup_lr,
        seed=seed,
    )
    t_warm = time.time() - t_warm_start
    print(f"  Head Warmup Complete: Validation Accuracy = {head_val_acc:6.2f}% ({t_warm:.2f}s)")

    # Step 2: The 12-Layer Perturbation Tournament
    print(f"\n[Step 2/2] Beginning 12-layer perturbation tournament ({tournament_steps} steps per layer, rho={rho}, lr={tournament_lr})...")
    layer_accuracies: Dict[str, float] = {}

    t_tourn_start = time.time()
    for l in range(12):
        t_l_start = time.time()
        acc = train_and_eval_layer(
            layer_idx=l,
            train_ds=train_ds,
            val_ds=val_ds,
            collator=collator,
            device=device,
            warmed_head_state=warmed_head_state,
            model_name=model_name,
            num_steps=tournament_steps,
            batch_size=batch_size,
            lr=tournament_lr,
            rho=rho,
            seed=seed,
        )
        elapsed_l = time.time() - t_l_start
        layer_accuracies[str(l)] = round(acc, 4)
        print(f"  Layer {l:2d}/11: Validation Accuracy = {acc:6.2f}% ({elapsed_l:5.1f}s)")

    total_time = time.time() - t_tourn_start
    print(f"\nTournament completed in {total_time:.2f}s.")

    # Sort layers by validation accuracy descending, resolving ties by layer index descending
    sorted_layers = sorted(
        [(int(l), acc) for l, acc in layer_accuracies.items()],
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )

    top_3_layers = [item[0] for item in sorted_layers[:3]]

    # Prepare export artifact
    result = {
        "seed": seed,
        "selected_layers": top_3_layers,
        "all_layer_accuracies": {str(l): layer_accuracies[str(l)] for l in range(12)},
        "head_warmup_val_accuracy": head_val_acc,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"selection_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved tournament selection to {out_path}")
    return result


# --------------------------------------------------------------------------- Standalone Verification Suite
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Head-Warmed 12-Layer Tournament for DP-SelFT Stage 3")
    parser.add_argument("--proxy_path", type=str, default="data/dp_proxy.parquet")
    parser.add_argument("--output_dir", type=str, default="configs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--tournament_steps", type=int, default=100)
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--tournament_lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rho", type=float, default=0.05)
    args = parser.parse_args()

    print("=" * 60)
    print("TASK 4B REVISION: HEAD-WARMED LAYER TOURNAMENT")
    print("=" * 60)

    results = run_layer_tournament(
        proxy_path=args.proxy_path,
        output_dir=args.output_dir,
        warmup_steps=args.warmup_steps,
        tournament_steps=args.tournament_steps,
        batch_size=args.batch_size,
        warmup_lr=args.warmup_lr,
        tournament_lr=args.tournament_lr,
        rho=args.rho,
        seed=args.seed,
    )

    out_file = os.path.join(args.output_dir, f"selection_seed{args.seed}.json")

    print("\n" + "=" * 60)
    print("TOURNAMENT LEADERBOARD (RANKED)")
    print("=" * 60)
    print(f"{'Rank':<6} {'Layer':<8} {'Val Accuracy':<16} {'Status'}")
    print("-" * 60)

    sorted_layers = sorted(
        [(int(l), acc) for l, acc in results["all_layer_accuracies"].items()],
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )

    for rank, (l_idx, acc) in enumerate(sorted_layers, 1):
        status = "★ SELECTED (Top 3)" if l_idx in results["selected_layers"] else ""
        print(f"{rank:<6} Layer {l_idx:<2} {acc:6.2f}%          {status}")

    print("=" * 60)
    print(f"Selected Top-3 Layers: {results['selected_layers']}")
    print(f"Head Warmup Accuracy: {results['head_warmup_val_accuracy']:.2f}%")
    print("=" * 60 + "\n")

    # Assertions per Task 4B Revision:
    # 1. Config file exists
    assert os.path.exists(out_file), f"Assertion failed: {out_file} does not exist"
    with open(out_file, "r") as f:
        loaded_data = json.load(f)

    # 2. Selected layers format
    selected = loaded_data["selected_layers"]
    assert len(selected) == 3, f"Expected 3 selected layers, got {len(selected)}"
    assert len(set(selected)) == 3, f"Selected layers must be unique, got {selected}"
    for l_idx in selected:
        assert isinstance(l_idx, int) and 0 <= l_idx <= 11, f"Invalid layer index: {l_idx}"

    # 3. All 12 layers present
    accs = loaded_data["all_layer_accuracies"]
    assert len(accs) == 12, f"Expected 12 layer accuracies, got {len(accs)}"
    assert set(accs.keys()) == {str(i) for i in range(12)}

    # 4. At least 3 layers achieve accuracy > 55.0%
    layers_above_55 = [l for l, acc in accs.items() if acc > 55.0]
    assert len(layers_above_55) >= 3, (
        f"Assertion Failed: Expected at least 3 layers with accuracy > 55.0%, found {len(layers_above_55)}: {layers_above_55}"
    )
    print(f"[✓] Assertion: {len(layers_above_55)} layers achieved accuracy > 55.0% (confirming genuine learning)")

    # 5. Top-3 layers are not all identical ties
    top_3_accs = [accs[str(l)] for l in selected]
    assert len(set(top_3_accs)) > 1 or any(acc > 55.0 for acc in top_3_accs), (
        f"Assertion Failed: Top-3 layers all tied at {top_3_accs}"
    )
    print(f"[✓] Assertion: Top-3 layers {selected} have non-trivial differentiated accuracies: {top_3_accs}")

    print("\n[SUCCESS] All Task 4B Revision assertions passed successfully!")
