import argparse
import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from src.models import get_model
from src.audit import run_mia_audit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--train_layers", type=str, default="all", help="'all', 'head', or '9,10,11'")
    parser.add_argument("--use_dp", action="store_true", help="Enable DP-SGD")
    parser.add_argument("--target_epsilon", type=float, default=5.0)
    parser.add_argument("--target_delta", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_train_samples", type=int, default=1024, help="Subsample as in DP-SelFT")
    parser.add_argument("--exp_name", type=str, default="test_run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")

    # 1. Dataset & Tokenization (SST-2)
    print("\n[1/5] Loading and tokenizing SST-2...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw_dataset = load_dataset("nyu-mll/glue", "sst2")

    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=128)

    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
    tokenized_dataset.set_format("torch")

    # Subsample training data
    n_train = min(args.max_train_samples, len(tokenized_dataset["train"]))
    train_dataset = tokenized_dataset["train"].select(range(n_train))
    val_dataset = tokenized_dataset["validation"]

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # 2. Model Setup
    print("\n[2/5] Initializing Model...")
    model, trainable_params, fraction = get_model(args.model_name, num_labels=2, train_layers=args.train_layers)
    model.to(device)
    model.train()

    # 3. Privacy Engine (Opacus)
    privacy_engine = None
    if args.use_dp:
        print("\n[3/5] Setting up Differential Privacy (Opacus)...")
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            model.to(device)

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,
        )
        print(f"PrivacyEngine active: Target eps={args.target_epsilon}, delta={args.target_delta}")
    else:
        print("\n[3/5] Standard optimization (No DP)...")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # 4. Training Loop
    print("\n[4/5] Starting Training...")
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        eps_str = f" | Epsilon spent: {privacy_engine.get_epsilon(args.target_delta):.2f}" if args.use_dp else ""
        print(f"  Epoch {epoch + 1}/{args.epochs} - Training Loss: {avg_loss:.4f}{eps_str}")

    # 5. Evaluate Utility & Privacy
    print("\n[5/5] Evaluating Utility & Auditing Privacy...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = (correct / total) * 100
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Run MIA audit
    raw_model = model._module if args.use_dp else model
    audit_train = train_dataset.select(range(min(200, len(train_dataset))))
    audit_test = val_dataset.select(range(min(200, len(val_dataset))))
    mia_auc = run_mia_audit(raw_model, audit_train, audit_test, device, collate_fn=collator)

    # 6. Save to CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/summary.csv"
    record = {
        "exp_name": args.exp_name,
        "train_layers": args.train_layers,
        "trainable_params": trainable_params,
        "param_fraction_%": round(fraction, 2),
        "use_dp": args.use_dp,
        "target_epsilon": args.target_epsilon if args.use_dp else "None",
        "val_accuracy": round(val_accuracy, 2),
        "mia_auc": round(mia_auc, 4),
    }

    df = pd.DataFrame([record])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)

    print(f"\n[✓] Results logged successfully to: {csv_path}\n")

if __name__ == "__main__":
    main()
