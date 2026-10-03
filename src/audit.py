import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

def compute_sample_losses(model, dataset, device, batch_size=32, collate_fn=None):
    """Computes cross-entropy loss for each individual sample."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    losses = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            losses.extend(loss.cpu().tolist())

    return np.array(losses)

def run_mia_audit(model, train_subset, test_subset, device, collate_fn=None):
    """
    Evaluates empirical privacy leakage via Membership Inference.
    Members (train) = 1, Non-members (test) = 0.
    Attacker score = -loss (Lower loss indicates training member).
    AUC = 0.50 means random guessing (Completely Safe).
    AUC = 1.00 means total privacy leak (Unsafe).
    """
    print("\n[+] Running Membership Inference Attack (MIA) Audit...")
    train_losses = compute_sample_losses(model, train_subset, device, collate_fn=collate_fn)
    test_losses = compute_sample_losses(model, test_subset, device, collate_fn=collate_fn)

    y_true = np.concatenate([np.ones_like(train_losses), np.zeros_like(test_losses)])
    # Attacker predicts member if loss is low -> score = -loss
    scores = np.concatenate([-train_losses, -test_losses])

    mia_auc = roc_auc_score(y_true, scores)
    print(f"[+] Audit Finished -> Empirical MIA AUC: {mia_auc:.4f}")
    return mia_auc
