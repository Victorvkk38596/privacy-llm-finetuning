"""MIA evaluation on cached logits and Offline RMIA engine.

Implements methodology_audit.md §1.3–§1.5 and methodology_audit_addendum.md §2.1–§2.5, §4.1:
  1. Forward passes happen once per model; logits are extracted and cached as compact .npy files.
  2. Attack Suite (Vectorized NumPy):
     - Naive loss threshold baseline: -L_target(x)
     - Continuous calibrated ratio (Primary RMIA): Ratio_x = p_target(x) / (1/2[(1+alpha)p_ref(x) + (1-alpha)] + 1e-8)
     - Offline RMIA quantile form (Marek Alg. 1): exact O((|X| + |Z|) log |Z|) via searchsorted
     - Marek Alg. 1 loss form (alpha=0, floor 1e-6)
  3. Evaluates ROC-AUC, TPR at 5% FPR, TPR at 1% FPR, calibration gap, and generalization gap.

Score convention everywhere: higher score => more likely a training member.
"""
import os
import re
from collections.abc import Mapping
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
LOSS_FLOOR = 1e-6
EPS = 1e-12


# --------------------------------------------------------------------------- logit extraction & caching
def extract_logits(model, dataloader, device) -> np.ndarray:
    """Extracts raw model logits as a float32 numpy array of shape (N, C).
    
    Evaluates in eval mode without gradient tracking. Restores original training state.
    """
    was_training = model.training
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, Mapping):
                kwargs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
                out = model(**kwargs)
            elif isinstance(batch, (tuple, list)):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 1 else None
                if attention_mask is not None:
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    out = model(input_ids=input_ids)
            else:
                out = model(batch.to(device))
            all_logits.append(out.logits.detach().float().cpu().numpy())
    model.train(was_training)
    logits = np.concatenate(all_logits, axis=0).astype(np.float32)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits encountered during extraction")
    return logits


def compute_logits(model, dataset, device, batch_size=64, collate_fn=None):
    """Returns (logits float32 (n, C), labels int64 (n,)) in dataset order. Preserved for pipeline compatibility."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    was_training = model.training
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(out.logits.detach().float().cpu().numpy())
            lbls = batch["labels"] if "labels" in batch else batch["label"]
            all_labels.append(lbls.detach().cpu().numpy())
    model.train(was_training)
    logits = np.concatenate(all_logits, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits encountered")
    return logits, labels


def save_logits(logits: np.ndarray, filepath: str) -> None:
    """Saves logits array as compact .npy file (~40 KB per split for SST-2)."""
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    np.save(filepath, np.asarray(logits, dtype=np.float32))


def load_logits(filepath: str) -> np.ndarray:
    """Loads logits array from .npy file as float32."""
    return np.load(filepath).astype(np.float32)


# --------------------------------------------------------------------------- derivations from logits
def compute_prob_true(logits: np.ndarray, labels: np.ndarray = None) -> np.ndarray:
    """Computes true-class probability p(y|x) = softmax(logits)[y] in float64.
    
    If labels is None, returns max predicted probability across classes.
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    p_all = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64)
        p_true = p_all[np.arange(len(labels)), labels]
    else:
        p_true = np.max(p_all, axis=-1)
    return np.clip(p_true, EPS, 1.0)


def compute_cross_entropy_loss(probs: np.ndarray) -> np.ndarray:
    """Cross-entropy loss: L = -log(p(y|x) + 10^-12)."""
    p = np.asarray(probs, dtype=np.float64)
    return -np.log(np.clip(p, EPS, 1.0))


def compute_scaled_logit(probs: np.ndarray) -> np.ndarray:
    """Scaled logit (stable LiRA form): log(p / (1 - p + 10^-12))."""
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p + EPS))


def log_p_true(logits, labels):
    """log p(y|x) in float64 (stable log-softmax). Preserved for backwards compatibility."""
    return np.log(compute_prob_true(logits, labels))


def sample_losses(logits, labels):
    """Per-sample cross entropy losses. Preserved for backwards compatibility."""
    return compute_cross_entropy_loss(compute_prob_true(logits, labels))


# --------------------------------------------------------------------------- ROC and TPR metrics
def auc(scores_in, scores_out) -> float:
    """Computes Mann-Whitney ROC-AUC for membership classification."""
    y = np.concatenate([np.ones(len(scores_in)), np.zeros(len(scores_out))])
    s = np.concatenate([scores_in, scores_out])
    return float(roc_auc_score(y, s))


def tpr_at_fpr(scores_in, scores_out, fpr_target: float) -> float:
    """Computes True Positive Rate (TPR) at a fixed False Positive Rate (FPR) ceiling."""
    y = np.concatenate([np.ones(len(scores_in)), np.zeros(len(scores_out))])
    s = np.concatenate([scores_in, scores_out])
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.where(fpr <= fpr_target)[0]
    return float(np.max(tpr[idx])) if len(idx) > 0 else 0.0


# --------------------------------------------------------------------------- attack suite
def score_naive_loss(loss: np.ndarray) -> np.ndarray:
    """Attack 1: Naive Loss Threshold (higher score = lower loss => member)."""
    return -np.asarray(loss, dtype=np.float64)


def compute_calibrated_ratio(p_target: np.ndarray, p_ref: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Attack 2: Continuous Calibrated Ratio (Primary RMIA).
    
    p_tilde(x) = 1/2 * [(1 + alpha) * p_ref(x) + (1 - alpha)]
    Ratio_x = p_target(x) / (p_tilde(x) + 1e-8)
    """
    pt = np.asarray(p_target, dtype=np.float64)
    pr = np.asarray(p_ref, dtype=np.float64)
    p_tilde = 0.5 * ((1.0 + alpha) * pr + (1.0 - alpha))
    return pt / (p_tilde + 1e-8)


def score_rmia_quantile(ratio_x: np.ndarray, ratio_z: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Attack 3: Offline RMIA Quantile Form (Marek Alg. 1).
    
    Score_quant(x) = (1 / |Z|) * sum_{z in Z} 1( Ratio_x / Ratio_z >= gamma )
                   = (1 / |Z|) * sum_{z in Z} 1( Ratio_z <= Ratio_x / gamma )
    Implemented in O((|X| + |Z|) log |Z|) via np.sort and np.searchsorted.
    """
    rz_sorted = np.sort(np.asarray(ratio_z, dtype=np.float64))
    rx = np.asarray(ratio_x, dtype=np.float64)
    threshold = rx / max(gamma, 1e-12)
    counts = np.searchsorted(rz_sorted, threshold, side="right")
    return counts / len(rz_sorted)


def score_rmia_alg1_loss(
    loss_target_x: np.ndarray,
    loss_ref_x: np.ndarray,
    loss_target_z: np.ndarray,
    loss_ref_z: np.ndarray,
    alpha: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Attack 4: Marek Alg. 1 Loss Form with alpha=0 and floor 1e-6.
    
    Ratio_x = L_target(x) / (1/2[(1+alpha)L_ref(x) + (1-alpha)])
    Score(x) = (1 / |Z|) * sum_{z in Z} 1( Ratio_x / Ratio_z < gamma )
    """
    def _loss_ratio(lt, lr):
        lt_c = np.maximum(np.asarray(lt, dtype=np.float64), LOSS_FLOOR)
        lr_c = np.maximum(np.asarray(lr, dtype=np.float64), LOSS_FLOOR)
        l_tilde = np.maximum(0.5 * ((1.0 + alpha) * lr_c + (1.0 - alpha)), LOSS_FLOOR)
        return lt_c / l_tilde

    rx = _loss_ratio(loss_target_x, loss_ref_x)
    lt_z = np.maximum(np.asarray(loss_target_z, dtype=np.float64), LOSS_FLOOR)
    lr_z = np.maximum(np.asarray(loss_ref_z, dtype=np.float64), LOSS_FLOOR)
    rz_sorted = np.sort(lt_z / lr_z)

    threshold = rx / max(gamma, 1e-12)
    counts = len(rz_sorted) - np.searchsorted(rz_sorted, threshold, side="right")
    return counts / len(rz_sorted)


# --------------------------------------------------------------------------- evaluation engine
def evaluate_attacks(
    target_logits,
    ref_logits,
    member_labels,
    nonmember_labels,
    pop_target_logits,
    pop_ref_logits,
    alpha: float = 0.0,
    pop_labels: np.ndarray = None,
    gamma: float = 1.0,
) -> dict:
    """Evaluates the complete attack suite across target and reference logits.
    
    Inputs can be 2-tuples (in_logits, out_logits) or concatenated arrays of shape (n_in + n_out, C).
    
    Returns dict with keys:
      - 'loss_auc': Naive loss ROC-AUC.
      - 'rmia_auc': Primary continuous calibrated ratio ROC-AUC.
      - 'rmia_quant_auc': Alg. 1 quantile form ROC-AUC.
      - 'rmia_loss_form_auc': Alg. 1 loss form ROC-AUC.
      - 'rmia_tpr_at_5fpr': True Positive Rate at FPR = 5.0%.
      - 'rmia_tpr_at_1fpr': True Positive Rate at FPR = 1.0%.
      - 'calibration_gap': rmia_auc - loss_auc.
      - 'generalization_gap': mean(loss_OUT) - mean(loss_IN).
    """
    n_in = len(member_labels)
    n_out = len(nonmember_labels)

    def _split(obj):
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            return obj[0], obj[1]
        elif isinstance(obj, dict):
            if "in" in obj and "out" in obj:
                return obj["in"], obj["out"]
            elif "member" in obj and "nonmember" in obj:
                return obj["member"], obj["nonmember"]
        arr = np.asarray(obj)
        if len(arr) == n_in + n_out:
            return arr[:n_in], arr[n_in:]
        raise ValueError(f"Expected array of length {n_in + n_out} or 2-element tuple, got {arr.shape}")

    tgt_in, tgt_out = _split(target_logits)
    ref_in, ref_out = _split(ref_logits)

    p_tgt_in = compute_prob_true(tgt_in, member_labels)
    p_tgt_out = compute_prob_true(tgt_out, nonmember_labels)
    p_ref_in = compute_prob_true(ref_in, member_labels)
    p_ref_out = compute_prob_true(ref_out, nonmember_labels)

    p_tgt_pop = compute_prob_true(pop_target_logits, pop_labels)
    p_ref_pop = compute_prob_true(pop_ref_logits, pop_labels)

    loss_tgt_in = compute_cross_entropy_loss(p_tgt_in)
    loss_tgt_out = compute_cross_entropy_loss(p_tgt_out)
    loss_ref_in = compute_cross_entropy_loss(p_ref_in)
    loss_ref_out = compute_cross_entropy_loss(p_ref_out)

    loss_tgt_pop = compute_cross_entropy_loss(p_tgt_pop)
    loss_ref_pop = compute_cross_entropy_loss(p_ref_pop)

    # Attack 1: Naive Loss Threshold
    loss_auc = auc(score_naive_loss(loss_tgt_in), score_naive_loss(loss_tgt_out))

    # Attack 2: Continuous Calibrated Ratio (Primary RMIA)
    rx_in = compute_calibrated_ratio(p_tgt_in, p_ref_in, alpha=alpha)
    rx_out = compute_calibrated_ratio(p_tgt_out, p_ref_out, alpha=alpha)
    rmia_auc = auc(rx_in, rx_out)

    # Attack 3: Offline RMIA Quantile Form
    rz_pop = compute_calibrated_ratio(p_tgt_pop, p_ref_pop, alpha=alpha)
    sq_in = score_rmia_quantile(rx_in, rz_pop, gamma=gamma)
    sq_out = score_rmia_quantile(rx_out, rz_pop, gamma=gamma)
    rmia_quant_auc = auc(sq_in, sq_out)

    # Attack 4: Marek Alg. 1 Loss Form
    score_alg1_in = score_rmia_alg1_loss(loss_tgt_in, loss_ref_in, loss_tgt_pop, loss_ref_pop, alpha=0.0, gamma=gamma)
    score_alg1_out = score_rmia_alg1_loss(loss_tgt_out, loss_ref_out, loss_tgt_pop, loss_ref_pop, alpha=0.0, gamma=gamma)
    rmia_loss_form_auc = auc(score_alg1_in, score_alg1_out)

    # Low-FPR TPRs on primary continuous ratio score
    rmia_tpr_at_5fpr = tpr_at_fpr(rx_in, rx_out, 0.05)
    rmia_tpr_at_1fpr = tpr_at_fpr(rx_in, rx_out, 0.01)

    return {
        "loss_auc": float(loss_auc),
        "rmia_auc": float(rmia_auc),
        "rmia_quant_auc": float(rmia_quant_auc),
        "rmia_loss_form_auc": float(rmia_loss_form_auc),
        "rmia_tpr_at_5fpr": float(rmia_tpr_at_5fpr),
        "rmia_tpr_at_1fpr": float(rmia_tpr_at_1fpr),
        "calibration_gap": float(rmia_auc - loss_auc),
        "generalization_gap": float(np.mean(loss_tgt_out) - np.mean(loss_tgt_in)),
    }


# --------------------------------------------------------------------------- backwards compatibility
def _log_denominator(logp_ref, alpha):
    p_ref = np.exp(logp_ref)
    return np.log(0.5 * ((1.0 + alpha) * p_ref + (1.0 - alpha)))


def log_ratio_prob(logp_target, logp_ref, alpha):
    return logp_target - _log_denominator(logp_ref, alpha)


def rmia_quantile(logratio_x, logratio_z, gamma=1.0):
    zs = np.sort(logratio_z)
    return np.searchsorted(zs, logratio_x - np.log(gamma), side="right") / len(zs)


def rmia_alg1_loss(loss_target_x, loss_ref_x, loss_target_z, loss_ref_z, alpha=0.0, gamma=1.0):
    return score_rmia_alg1_loss(loss_target_x, loss_ref_x, loss_target_z, loss_ref_z, alpha=alpha, gamma=gamma)


def evaluate_target(target, ref, member_split, alpha_grid=ALPHA_GRID):
    """Legacy evaluation helper on model cache dictionary."""
    def stats(model_cache, split):
        d = model_cache[split]
        return log_p_true(d["logits"], d["labels"])

    lp_t_in, lp_t_out, lp_t_pop = (stats(target, s) for s in (member_split, "out", "pop"))
    lp_r_in, lp_r_out, lp_r_pop = (stats(ref, s) for s in (member_split, "out", "pop"))
    L_t_in, L_t_out, L_t_pop = -lp_t_in, -lp_t_out, -lp_t_pop
    L_r_in, L_r_out, L_r_pop = -lp_r_in, -lp_r_out, -lp_r_pop

    m = {"n_audit_in": len(lp_t_in), "n_audit_out": len(lp_t_out)}
    m["auc_loss"] = auc(lp_t_in, lp_t_out)
    m["gen_gap"] = float(L_t_out.mean() - L_t_in.mean())
    for a in alpha_grid:
        lr_in = log_ratio_prob(lp_t_in, lp_r_in, a)
        lr_out = log_ratio_prob(lp_t_out, lp_r_out, a)
        m[f"auc_ratio_a{a:g}"] = auc(lr_in, lr_out)
    for a in (0.0, 1.0):
        lr_in = log_ratio_prob(lp_t_in, lp_r_in, a)
        lr_out = log_ratio_prob(lp_t_out, lp_r_out, a)
        lr_pop = log_ratio_prob(lp_t_pop, lp_r_pop, a)
        m[f"auc_rmia_q_a{a:g}"] = auc(rmia_quantile(lr_in, lr_pop), rmia_quantile(lr_out, lr_pop))
        m[f"tpr5_ratio_a{a:g}"] = tpr_at_fpr(lr_in, lr_out, 0.05)
        m[f"tpr1_ratio_a{a:g}"] = tpr_at_fpr(lr_in, lr_out, 0.01)
    m["auc_rmia_alg1_loss"] = auc(
        rmia_alg1_loss(L_t_in, L_r_in, L_t_pop, L_r_pop),
        rmia_alg1_loss(L_t_out, L_r_out, L_t_pop, L_r_pop))
    v = target["val"]
    m["val_accuracy"] = float((v["logits"].argmax(1) == v["labels"]).mean() * 100)
    return m


def run_mia_audit(model, train_subset, test_subset, device, collate_fn=None):
    """Legacy reference-free loss-threshold AUC."""
    lt, yt = compute_logits(model, train_subset, device, collate_fn=collate_fn)
    lo, yo = compute_logits(model, test_subset, device, collate_fn=collate_fn)
    return auc(log_p_true(lt, yt), log_p_true(lo, yo))


# --------------------------------------------------------------------------- verification suite
if __name__ == "__main__":
    import time
    import tempfile

    print("Executing Task 3 Verification Suite in src/audit.py...\n")

    # 1. Generate synthetic logits with artificial memorization signal
    np.random.seed(42)
    n_in = 1024
    n_out = 1024
    n_pop = 1024

    member_labels = np.random.binomial(1, 0.5, size=n_in)
    nonmember_labels = np.random.binomial(1, 0.5, size=n_out)
    pop_labels = np.random.binomial(1, 0.5, size=n_pop)

    ref_in = np.random.normal(0, 1.0, size=(n_in, 2)).astype(np.float32)
    ref_out = np.random.normal(0, 1.0, size=(n_out, 2)).astype(np.float32)
    ref_pop = np.random.normal(0, 1.0, size=(n_pop, 2)).astype(np.float32)

    tgt_in = ref_in.copy() + np.random.normal(0, 0.5, size=(n_in, 2)).astype(np.float32)
    tgt_out = ref_out.copy() + np.random.normal(0, 0.5, size=(n_out, 2)).astype(np.float32)
    tgt_pop = ref_pop.copy() + np.random.normal(0, 0.5, size=(n_pop, 2)).astype(np.float32)

    # Artificial memorization signal for members
    for i, y in enumerate(member_labels):
        tgt_in[i, y] += 1.8
    for i, y in enumerate(nonmember_labels):
        tgt_out[i, y] += 0.2

    # 2. Timing and attack execution
    t_start = time.perf_counter()
    metrics = evaluate_attacks(
        target_logits=(tgt_in, tgt_out),
        ref_logits=(ref_in, ref_out),
        member_labels=member_labels,
        nonmember_labels=nonmember_labels,
        pop_target_logits=tgt_pop,
        pop_ref_logits=ref_pop,
        alpha=0.0,
        pop_labels=pop_labels,
        gamma=1.0,
    )
    scoring_time = time.perf_counter() - t_start

    print("=" * 60)
    print("TASK 3 METRICS EVALUATION")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:22s}: {v:.6f}")
    print(f"  {'scoring_time_sec':22s}: {scoring_time:.6f}s")
    print("=" * 60 + "\n")

    # Assertion a: Continuous ratio AUC and Quantile RMIA AUC agree to within tie resolution (< 1e-3)
    auc_diff = abs(metrics["rmia_auc"] - metrics["rmia_quant_auc"])
    assert auc_diff < 1e-3, f"Assertion (a) Failed: |rmia_auc - rmia_quant_auc| = {auc_diff:.6e} >= 1e-3"
    print(f"[✓] Assertion (a) Passed: |rmia_auc - rmia_quant_auc| = {auc_diff:.6e} < 10^-3 (Theorem §2.1 validated)")

    # Assertion b: Scoring executes in under 0.1s for 2048 audit + 1024 pop items
    assert scoring_time < 0.1, f"Assertion (b) Failed: scoring time {scoring_time:.4f}s >= 0.1s"
    print(f"[✓] Assertion (b) Passed: Vectorized scoring executed in {scoring_time:.6f}s < 0.1s")

    # Assertion c: All return dictionary keys match requested schema
    required_keys = {
        "loss_auc",
        "rmia_auc",
        "rmia_quant_auc",
        "rmia_tpr_at_5fpr",
        "rmia_tpr_at_1fpr",
        "calibration_gap",
        "generalization_gap",
    }
    assert required_keys.issubset(metrics.keys()), (
        f"Assertion (c) Failed: missing keys {required_keys - set(metrics.keys())}"
    )
    print(f"[✓] Assertion (c) Passed: All {len(required_keys)} required metric keys are present in schema")

    # Assertion d: Disk save/load roundtrips with zero numerical error
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_f:
        tmp_path = tmp_f.name
    try:
        save_logits(tgt_in, tmp_path)
        loaded_logits = load_logits(tmp_path)
        assert np.array_equal(tgt_in, loaded_logits), "Assertion (d) Failed: arrays not identical"
        max_abs_err = float(np.max(np.abs(tgt_in - loaded_logits)))
        assert max_abs_err == 0.0, f"Assertion (d) Failed: max error {max_abs_err} != 0.0"
        file_size_kb = os.path.getsize(tmp_path) / 1024.0
        print(f"[✓] Assertion (d) Passed: Disk save/load roundtrip zero error (max abs error = {max_abs_err}, size = {file_size_kb:.1f} KB)")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\n" + "=" * 60)
    print("ALL 4 TASK 3 ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 60)

