# TASK_01: Core Grid Execution & Location / Selection Study

**Specifications:** `docs/ARCHITECTURE.md`, `methodology_audit.md` §2–§4, and `methodology_audit_addendum.md` §3–§5.

This document formalizes the execution commands, configs, and acceptance criteria for the core grid of 8 experimental configurations under the swap-pair protocol.

---

## 1. Locked Experiment Configurations

All experiments are standardized on:
- Backbone: `roberta-base` (embeddings strictly frozen, 124,647,170 total parameters)
- Schedule: `max_steps = 1000` (replaces epoch loops)
- Batch sizes: `logical_batch_size = 64`, `physical_batch_size = 16` via Opacus `BatchMemoryManager`
- Optimizer: AdamW, clipping norm $C = 1.0$, 5% linear warmup
- Learning rates: $\eta = 3\times 10^{-5}$ for clean; $\eta = 3\times 10^{-4}$ for DP-SGD
- Differential Privacy: Target $\delta = 10^{-5}$ ($< 1/N$ with $N = 1024$)
- Partitions: Deterministic 6-strata hash partitioning with F1 family policy (`D_A`, `D_B`, `D_out`, `D_pop`, `D_tune`)

| Config File | Trainable Layers | $\epsilon_{\text{target}}$ | $\epsilon_{\text{ft}}$ | $\epsilon_{\text{sel}}$ | Description |
|---|---|---|---|---|---|
| [`configs/baseline_full_clean.yaml`](file:///home/vasudev-krishna/privacy-llm-finetuning/configs/baseline_full_clean.yaml) | `"all"` (12 blocks + head) | $\infty$ (Clean) | - | - | Non-private full fine-tuning baseline |
| [`configs/baseline_full_dp.yaml`](file:///home/vasudev-krishna/privacy-llm-finetuning/configs/baseline_full_dp.yaml) | `"all"` (12 blocks + head) | $5.0$ | $5.0$ | - | DP full fine-tuning baseline ($85.6$M params) |
| [`configs/head_only_dp.yaml`](file:///home/vasudev-krishna/privacy-llm-finetuning/configs/head_only_dp.yaml) | `"head"` | $5.0$ | $5.0$ | - | Classifier head only ($592$k params) |
| [`configs/selective_early_k3.yaml`](file:///home/vasudev-krishna/privacy-llm-finetuning/configs/selective_early_k3.yaml) | `"0,1,2"` | $5.0$ | $5.0$ | - | Early encoder layers ($21.9$M params) |
| [`configs/selective_mid_k3.yaml`](file:///home/vasudev-krishna/privacy-llm-finetuning/configs/selective_mid_k3.yaml) | `"4,5,6"` | $5.0$ | $5.0$ | - | Mid encoder layers ($21.9$M params) |
| [`configs/selective_late_k3.yaml`](file:///home/vasudev-krishna/privacy-llm-finetuning/configs/selective_late_k3.yaml) | `"9,10,11"` | $5.0$ | $5.0$ | - | Late encoder layers ($21.9$M params) |
| [`configs/selective_dpselft_k3.yaml`](file:///home/vasudev-krishna/privacy-llm-finetuning/configs/selective_dpselft_k3.yaml) | `"3,6,4"` | $5.0$ | $4.7$ | $0.3$ | DP-SelFT perturbation-optimal layers |
| [`configs/control_H_prime.yaml`](file:///home/vasudev-krishna/privacy-llm-finetuning/configs/control_H_prime.yaml) | `"9,10,11"` | $4.7$ | $4.7$ | - | Selection-matched control $H'$ ($\sigma$ matched to DP-SelFT) |

---

## 2. Execution Commands

### A. Pre-flight Dry Run (Verification on CPU)
```bash
python run_experiment.py --dry_run --train_layers head --max_steps 5
```

### B. Running Single Configs
```bash
# Clean baseline
python run_experiment.py --config configs/baseline_full_clean.yaml

# DP Head-only baseline
python run_experiment.py --config configs/head_only_dp.yaml

# Location Study (Early, Mid, Late k=3)
python run_experiment.py --config configs/selective_early_k3.yaml
python run_experiment.py --config configs/selective_mid_k3.yaml
python run_experiment.py --config configs/selective_late_k3.yaml

# DP-SelFT vs Selection-Matched Control H'
python run_experiment.py --config configs/selective_dpselft_k3.yaml
python run_experiment.py --config configs/control_H_prime.yaml

# Full DP baseline (requires GPU)
python run_experiment.py --config configs/baseline_full_dp.yaml
```

### C. Sequential Grid Sweep
```bash
for cfg in baseline_full_clean head_only_dp \
           selective_early_k3 selective_mid_k3 selective_late_k3 \
           selective_dpselft_k3 control_H_prime baseline_full_dp; do
    echo "=== Running $cfg ==="
    python run_experiment.py --config configs/$cfg.yaml 2>&1 | tee logs_$cfg.txt
done
```

---

## 3. Acceptance Gates & Verification

Each experiment run executes the swap-pair protocol ($M_A$ trained on $D_A$, $M_B$ trained on $D_B$), extracts logits on all 5 splits ($D_A, D_B, D_{\text{out}}, D_{\text{pop}}, \text{validation}$), performs bidirectional cross-audits on CPU, and appends two rows to `results/summary.csv`.

| Gate | Description | Condition |
|---|---|---|
| **A1 Schema** | Header integrity | Exactly 22 columns defined in `CSV_COLUMNS`. |
| **A2 Trainable Params** | Parameter counts | `head` = 592,130; $k=3$ layers = 21,856,766; `"all"` = 85,646,594. |
| **A3 Budget** | DP guarantee | `epsilon_spent` $\le$ `target_epsilon` and `noise_multiplier` > 0 for all DP runs. |
| **A4 Clean Parity** | Non-DP fields | `target_epsilon` is `None`, `epsilon_spent` is empty for clean run. |
| **A5 Clean Utility** | Clean convergence | `val_accuracy` > 85.0% on standard SST-2 validation set. |
| **A6 Audit Dimension** | Sample counts | `n_audit_in` = `n_audit_out` = 1,024. |
| **A7 Numerical Stability** | Convergence | No NaN/Inf values; `final_train_loss` finite. |
