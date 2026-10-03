# TASK_01: Baselines and Location Study (ε = ∞ / 5.0, k = 3)

**Spec:** `docs/ARCHITECTURE.md` (read §2 invariants and §3 data contract first).
**Goal:** Produce the six runs below for seed 42 and verify they pass the validity gates. Do **not** tune hyperparameters on the validation set.

## 0. Setup

```bash
git checkout -b task01-baselines-location
python -m venv venv && source venv/bin/activate   # skip if venv exists
pip install -r requirements.txt
```

## 1. Pre-flight checks (must pass before any full run)

```bash
# 1a. Smoke test (about 1 minute). Writes a row; it is removed in step 1c.
python run_experiment.py --config configs/head_only_dp.yaml \
    --max_train_samples 64 --audit_size 32 --epochs 1 --exp_name smoke_preflight

# 1b. Parameter-count oracle (ARCHITECTURE I-M4): expect 592130 + k*7087872
python - <<'EOF'
from src.models import get_model
for tl, exp in [("head",592130),("0,1,2",592130+3*7087872),("all",124647170)]:
    _, n, f = get_model("roberta-base", 2, tl); assert n == exp, (tl, n, exp)
print("param counts OK")
EOF

# 1c. Remove smoke rows so they never enter results
python - <<'EOF'
import pandas as pd
df = pd.read_csv("results/summary.csv"); df = df[~df.exp_name.str.startswith("smoke")]
df.to_csv("results/summary.csv", index=False)
EOF
```

Note: the first run with the new code archives the old 2-row file as `results/summary_legacy.csv` (header mismatch). This is expected. Do not delete it and do not use it in analysis.

## 2. Runs

GPU required for the full-FT DP run. Use Kaggle T4 if the local machine has no GPU.
If `baseline_full_dp` runs out of memory, lower `--batch_size` to 16 **for all six runs** and re-run all of them. Do not change it for one run only.

```bash
for c in baseline_full_clean baseline_full_dp \
         selective_early_k3 selective_mid_k3 selective_late_k3 head_only_dp; do
  python run_experiment.py --config configs/$c.yaml 2>&1 | tee logs_$c.txt
done
```

Optionally repeat with `--seed 43 --exp_name <name>_s43` and `--seed 44 --exp_name <name>_s44` (ARCHITECTURE G5 wants ≥ 3 seeds for claims).

## 3. Acceptance gates

Inspect `results/summary.csv`.

| Gate | Run(s) | Condition |
|---|---|---|
| A1 schema | all | Header equals the 22 columns in `CSV_COLUMNS`. One row per run. |
| A2 counts | all | `trainable_params` = 124,647,170 (full), 592,130 (head), 21,856,766 (k3 sets). |
| A3 budget | the 4 DP runs with layers selected plus `baseline_full_dp` | `epsilon_spent` ≤ 5.0 and `noise_multiplier` > 0. |
| A4 no-DP fields | `baseline_full_clean` | `target_epsilon == None`, `epsilon_spent` empty. |
| A5 learned (clean) | `baseline_full_clean` | `val_accuracy` > 85. |
| A6 learned (unfrozen layers) | `baseline_full_dp`, `selective_{early,mid,late}_k3` | `val_accuracy` > 75. |
| A7 learned (head) | `head_only_dp` | `val_accuracy` > 60. |
| A8 audit size | all | `n_audit_in == n_audit_out == 500`. |
| A9 finite | all | No NaN or inf. `final_train_loss` < 0.69 (below chance-level loss). |

`mia_auc` has no pass/fail threshold, since it is the measured quantity. Standard error at n = 500 per class is about 0.016 (ARCHITECTURE §1.2.2).

**If a gate fails:** do not edit configs to chase the gate. Commit the logs, write the failing numbers in `results/TASK_01_REPORT.md`, and stop. The most likely cause is the learning rate (5e-5, untuned) or too few epochs for DP, and a change to it needs sign-off because it applies to all configs.

## 4. Report

Create `results/TASK_01_REPORT.md` with:
1. A table of the six rows (accuracy, `mia_auc`, `epsilon_spent`, `noise_multiplier`).
2. Early vs. mid vs. late: equal `d_Λ`, so differences isolate **location**. State whether the gap exceeds ~2·SE.
3. Which gates passed or failed.

## 5. Git

```bash
git add configs/ run_experiment.py requirements.txt results/summary.csv results/summary_legacy.csv results/TASK_01_REPORT.md
git commit -m "TASK_01: baselines and location study (seed 42)"
git push -u origin task01-baselines-location
```

Do not commit `venv/` or `logs_*.txt` unless asked. Open a PR; do not merge it yourself.
