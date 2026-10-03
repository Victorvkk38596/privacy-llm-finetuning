# Research Project: Parameter Updates vs. Empirical Privacy Leakage in LLM Fine-Tuning

## Core Objective
Investigate how the **fraction** (how many) and **location** (where: early, middle, late layers) of trainable parameters affect **empirical privacy leakage** (MIA ROC-AUC) when fine-tuning language models under Differential Privacy (DP-SGD / DP-AdamW).

## Academic Context & Papers
1. **Marek et al. (ICLR 2026):** Benchmarks empirical privacy risks of DP adaptations using Membership Inference Attacks (RMIA) and canary extraction. Key finding: theoretical epsilon does not guarantee empirical safety; adaptation structure and data distribution heavily govern leakage.
2. **Sha et al. (DP-SelFT, 2026):** Demonstrates that selectively fine-tuning a subset of transformer layers under DP improves utility because the noise energy scales linearly with the trainable parameter dimension ($d_\Lambda \sigma^2 C^2$).
3. **Our Gap:** DP-SelFT only measured utility (accuracy); Marek et al. only tested coarse baselines. We systematically map the privacy-utility Pareto frontier across parameter fractions and locations.

## Technical Architecture
- **Language / Framework:** Python, PyTorch, Hugging Face `transformers`, `datasets`, `opacus`, `scikit-learn`.
- **Base Model:** `roberta-base` (12 layers, ~125M parameters).
- **Task:** SST-2 sentiment classification (binary).
- **Compute:** Local Linux ThinkBook for development/dry runs; Kaggle GPUs (T4) dispatched headlessly via `kaggle` CLI.
- **Repository Structure:**
  - `src/models.py`: Layer freezing logic (`all`, `head`, or comma-separated layer indices).
  - `src/audit.py`: MIA evaluation (per-sample loss on members vs non-members -> ROC-AUC).
  - `run_experiment.py`: Main execution script.

## Immediate Task 1 (Bug Fix)
`run_experiment.py` currently crashes on `load_dataset("glue", "sst2")` with:
`huggingface_hub.errors.HfUriError: Repository id must be 'namespace/name', got 'glue'.`
- **Fix:** Update dataset loading to use the namespaced repository `nyu-mll/glue` (or `load_dataset("nyu-mll/glue", "sst2")`, or `load_dataset("sst2")`) and ensure tokenization is compatible.
- Verify by running the 16-sample local smoke test:
  `python run_experiment.py --train_layers head --max_train_samples 16 --epochs 1 --batch_size 8 --exp_name smoke_test`

## Subsequent Tasks (Roadmap)
1. **Kaggle Dispatch Setup:**
   - Create `kaggle_runner/kernel-metadata.json` and `kaggle_runner/run_on_kaggle.py`.
   - Ensure the script clones the GitHub repo, runs an experiment with GPU enabled, and outputs `results/summary.csv`.
   - Verify headless dispatch using `kaggle kernels push -p kaggle_runner`.
2. **Execute Experimental Grid:**
   - Baselines: Full FT ($\epsilon=\infty$), Full FT ($\epsilon=5$), Head Only ($\epsilon=5$).
   - Location Study: Early (0,1,2), Mid (4,5,6), Late (9,10,11) at $\epsilon=5$.
   - Fraction Study: 1 layer vs 3 layers vs 6 layers vs Full.
3. **Analysis:**
   - Generate plots comparing Parameter Fraction vs. MIA AUC, Location vs. MIA AUC, and the Accuracy vs. MIA AUC Pareto curve.