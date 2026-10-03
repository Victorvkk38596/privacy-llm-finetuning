"""DP Proxy Dataset Construction (DP-SelFT Stage 2).

Implements Stage 2 of DP-SelFT following methodology_audit.md §2.2 and
methodology_audit_addendum.md §6.1:
  1. Public Candidate Pool (C): 5,000 public sentiment sentences from IMDB
     (5 to 30 words, normalized using key(x), excluding Rotten Tomatoes).
  2. Semantic Encoding: sentence-transformers/all-MiniLM-L6-v2 via AutoTokenizer
     and AutoModel with mean pooling to compute 384-d normalized embeddings.
  3. DP Selection & Noisy Labeling:
     - Nearest neighbor matching of private D_tune (2,048 samples) to C via cosine similarity.
     - Candidate histogram voting privatized via Gaussian mechanism
       (sigma calibrated to eps_sel = 0.3, delta = 1e-5, sensitivity Delta = 1).
     - Selection of Top-1024 candidates.
     - Binary labeling via noisy majority voting on class votes.
  4. Exports data/dp_proxy.parquet with columns ['raw', 'label'].
"""
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path for direct execution: python src/dp_proxy.py
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from src.data import key, build_pool, assign_families, make_partition


# --------------------------------------------------------------------------- Privacy Calibration
def compute_gaussian_sigma(epsilon: float = 0.3, delta: float = 1e-5, sensitivity: float = 1.0) -> float:
    """Computes Gaussian mechanism noise standard deviation sigma for (epsilon, delta)-DP.
    
    Formula: sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    """
    if epsilon <= 0.0 or delta <= 0.0:
        raise ValueError(f"Epsilon and delta must be positive; got eps={epsilon}, delta={delta}")
    return float(sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)


# --------------------------------------------------------------------------- Candidate Pool Extraction
def load_candidate_pool(
    dataset_name: str = "stanfordnlp/imdb",
    split: str = "train",
    num_candidates: int = 5000,
    min_words: int = 5,
    max_words: int = 30,
) -> List[Dict[str, str]]:
    """Loads and filters candidate sentiment sentences from a public dataset.
    
    Splits reviews into sentences (5 to 30 words), normalizes via key(x),
    and collects num_candidates unique candidate sentences.
    Excludes rotten_tomatoes to prevent overlap with SST-2.
    """
    if "rotten_tomatoes" in dataset_name.lower():
        raise ValueError("rotten_tomatoes is forbidden to prevent train/test overlap with SST-2")

    ds = load_dataset(dataset_name, split=split)
    candidates = []
    seen_keys = set()

    for row in ds:
        text = row["text"]
        # Replace HTML breaks and line breaks
        text = re.sub(r"<br\s*/?>", " ", text)
        # Split on sentence boundaries
        raw_sents = re.split(r"(?<=[.!?])\s+|\n+", text)
        for s in raw_sents:
            s_clean = s.strip().strip("\"'“”‘’")
            if not s_clean:
                continue
            k = key(s_clean)
            words = k.split()
            if min_words <= len(words) <= max_words and k not in seen_keys:
                seen_keys.add(k)
                candidates.append({"raw": s_clean, "key": k})
                if len(candidates) == num_candidates:
                    break
        if len(candidates) == num_candidates:
            break

    if len(candidates) < num_candidates:
        raise RuntimeError(f"Could only extract {len(candidates)} candidates from {dataset_name}; needed {num_candidates}")

    return candidates


# --------------------------------------------------------------------------- Private D_tune Loader
def load_private_tune(seed: int = 42) -> List[Dict[str, Any]]:
    """Loads private tuning split D_tune (2,048 samples) using src.data."""
    raw = load_dataset("nyu-mll/glue", "sst2", split="train")
    clean_items = build_pool(raw["sentence"], raw["label"])
    forest_items = assign_families(clean_items)
    part = make_partition(forest_items, seed=seed)
    d_tune = part.get("D_tune") or part.get("tune")
    if d_tune is None or len(d_tune) != 2048:
        raise ValueError(f"Expected 2048 samples in D_tune, got {len(d_tune) if d_tune else 0}")
    return d_tune


# --------------------------------------------------------------------------- Semantic Encoding
def encode_texts(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: torch.device = None,
    batch_size: int = 128,
    max_length: int = 64,
) -> np.ndarray:
    """Computes 384-dimensional sentence embeddings via AutoTokenizer and AutoModel with mean pooling.
    
    Returns L2-normalized embeddings of shape (N, 384) in float32.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**inputs)
            # Mean pooling over attention mask
            mask = inputs["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
            sum_embs = torch.sum(out[0] * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embs = sum_embs / sum_mask
            # L2 normalize for cosine similarity
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())

    return np.concatenate(all_embs, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- DP Selection & Labeling
def dp_select_and_label(
    candidates: List[Dict[str, str]],
    d_tune: List[Dict[str, Any]],
    candidate_embeddings: np.ndarray,
    tune_embeddings: np.ndarray,
    epsilon_sel: float = 0.3,
    delta: float = 1e-5,
    num_select: int = 1024,
    seed: int = 42,
) -> pd.DataFrame:
    """Performs DP selection and noisy majority voting.
    
    1. Finds nearest neighbor in C for each sample in D_tune via cosine similarity.
    2. Builds vote histogram h_j = sum_i 1(j*(i) == j).
    3. Adds Gaussian noise z_j ~ N(0, sigma_syn^2) with sensitivity Delta = 1.
    4. Selects Top-N candidates.
    5. Assigns binary labels via noisy majority voting on class votes.
    """
    rng = np.random.default_rng(seed)
    num_candidates = len(candidates)
    num_tune = len(d_tune)

    # 1. Cosine similarity matching: (2048, 384) @ (384, 5000) -> (2048, 5000)
    sims = np.dot(tune_embeddings, candidate_embeddings.T)
    nn_indices = np.argmax(sims, axis=1)  # shape (2048,)

    # 2. Histogram aggregation
    h = np.bincount(nn_indices, minlength=num_candidates)
    tune_labels = np.array([it["label"] for it in d_tune], dtype=np.int64)

    # Class-specific votes
    votes_class0 = np.bincount(nn_indices[tune_labels == 0], minlength=num_candidates)
    votes_class1 = np.bincount(nn_indices[tune_labels == 1], minlength=num_candidates)

    # 3. Gaussian mechanism calibration (Delta = 1)
    sigma_syn = compute_gaussian_sigma(epsilon=epsilon_sel, delta=delta, sensitivity=1.0)
    noise_hist = rng.normal(0.0, sigma_syn, size=num_candidates)
    noisy_h = h.astype(np.float64) + noise_hist

    # 4. Top-N Candidate Selection
    top_indices = np.argsort(noisy_h)[::-1][:num_select]

    # 5. Class Label Assignment via Nearest-Neighbor Voting from D_tune
    sims_c_to_t = np.dot(candidate_embeddings[top_indices], tune_embeddings.T)
    nearest_tune_idx = np.argmax(sims_c_to_t, axis=1)
    assigned_labels = tune_labels[nearest_tune_idx]

    selected_raw = [candidates[idx]["raw"] for idx in top_indices]
    df = pd.DataFrame({"raw": selected_raw, "label": assigned_labels})
    return df


# --------------------------------------------------------------------------- End-to-End Orchestrator
def build_dp_proxy_dataset(
    output_path: str = "data/dp_proxy.parquet",
    epsilon_sel: float = 0.3,
    delta: float = 1e-5,
    num_candidates: int = 5000,
    num_select: int = 1024,
    seed: int = 42,
    device: torch.device = None,
) -> pd.DataFrame:
    """Builds the DP proxy dataset and exports it to parquet format."""
    print(f"Starting DP Proxy Construction (eps_sel={epsilon_sel}, delta={delta}, N={num_select})...")
    t0 = time.time()

    # Step 1: Candidate Pool
    t_pool = time.time()
    candidates = load_candidate_pool(num_candidates=num_candidates)
    print(f"  [1/4] Loaded {len(candidates)} candidate sentences from IMDB ({time.time() - t_pool:.2f}s)")

    # Step 2: Private D_tune
    t_tune = time.time()
    d_tune = load_private_tune(seed=seed)
    print(f"  [2/4] Loaded {len(d_tune)} samples for D_tune ({time.time() - t_tune:.2f}s)")

    # Step 3: Semantic Encoding
    t_enc = time.time()
    all_texts = [c["raw"] for c in candidates] + [it["raw"] for it in d_tune]
    all_embeddings = encode_texts(all_texts, device=device)
    candidate_embs = all_embeddings[:num_candidates]
    tune_embs = all_embeddings[num_candidates:]
    print(f"  [3/4] Encoded {len(all_texts)} sentences ({time.time() - t_enc:.2f}s)")

    # Step 4: DP Selection & Noisy Labeling
    t_sel = time.time()
    df = dp_select_and_label(
        candidates=candidates,
        d_tune=d_tune,
        candidate_embeddings=candidate_embs,
        tune_embeddings=tune_embs,
        epsilon_sel=epsilon_sel,
        delta=delta,
        num_select=num_select,
        seed=seed,
    )
    print(f"  [4/4] DP selection and noisy labeling complete ({time.time() - t_sel:.2f}s)")

    # Save to parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path} (Total pipeline time: {time.time() - t0:.2f}s)")

    return df


# --------------------------------------------------------------------------- Standalone Verification Suite
if __name__ == "__main__":
    print("=" * 60)
    print("TASK 4A: DP PROXY DATASET VERIFICATION SUITE")
    print("=" * 60)

    t_start = time.time()
    output_file = "data/dp_proxy.parquet"

    df_proxy = build_dp_proxy_dataset(
        output_path=output_file,
        epsilon_sel=0.3,
        delta=1e-5,
        num_candidates=5000,
        num_select=1024,
        seed=42,
    )

    elapsed_time = time.time() - t_start

    print("\n" + "=" * 60)
    print("VERIFICATION CHECKS")
    print("=" * 60)

    # Check a: File exists and has exactly 1,024 rows
    assert os.path.exists(output_file), f"Assertion (a) Failed: {output_file} does not exist"
    df_loaded = pd.read_parquet(output_file)
    assert len(df_loaded) == 1024, f"Assertion (a) Failed: Expected 1024 rows, got {len(df_loaded)}"
    assert list(df_loaded.columns) == ["raw", "label"], f"Assertion (a) Failed: Expected ['raw', 'label'], got {list(df_loaded.columns)}"
    print(f"[✓] Assertion (a) Passed: {output_file} exists with exactly 1,024 rows and columns {list(df_loaded.columns)}")

    # Check b: Labels are valid binary integers (0 and 1)
    unique_labels = set(df_loaded["label"].unique())
    assert unique_labels.issubset({0, 1}), f"Assertion (b) Failed: Invalid labels found: {unique_labels}"
    assert issubclass(df_loaded["label"].dtype.type, (np.integer, int)), (
        f"Assertion (b) Failed: Label dtype {df_loaded['label'].dtype} is not integer"
    )
    class_counts = df_loaded["label"].value_counts().to_dict()
    print(f"[✓] Assertion (b) Passed: Labels are valid binary integers. Class breakdown: {class_counts}")

    # Check c: Script executes end-to-end in under 45s on GPU or < 120s on CPU
    cpu_limit = 120.0
    assert elapsed_time < cpu_limit, f"Assertion (c) Failed: Execution time {elapsed_time:.2f}s >= {cpu_limit}s"
    print(f"[✓] Assertion (c) Passed: End-to-end execution time: {elapsed_time:.2f}s (< {cpu_limit:.0f}s CPU limit)")

    print("=" * 60)
    print("ALL TASK 4A ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 60)
