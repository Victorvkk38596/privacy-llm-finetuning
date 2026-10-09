
# Privacy-Preserving LLM Fine-Tuning

Investigating the privacy–utility trade-off of selective fine-tuning using Differentially Private Stochastic Gradient Descent (DP-SGD).

This project explores whether fine-tuning selected layers of a pretrained RoBERTa model can maintain classification performance while improving privacy under differential privacy constraints.

The experiments use the SST-2 sentiment classification dataset from GLUE and evaluate both formal differential privacy guarantees and empirical privacy leakage through membership inference attacks.

## Research Objectives

- Compare full fine-tuning with selective layer fine-tuning.
- Evaluate the impact of trainable parameter count on model utility and privacy.
- Investigate how Transformer layer location affects performance.
- Implement DP-SGD for differentially private model training.
- Explore DP-SelFT-style private layer selection.
- Evaluate privacy leakage using Membership Inference Attacks (MIA) and RMIA-style methods.
- Analyze the trade-off between classification accuracy and privacy.

## Key Features

- **RoBERTa-base:** Pretrained Transformer model for binary sentiment classification.
- **Selective fine-tuning:** Train specific Transformer blocks while freezing the remaining backbone.
- **Differential privacy:** Gradient clipping and Gaussian noise through DP-SGD.
- **Private layer selection:** DP-SelFT-style selection using a differentially private proxy dataset.
- **Privacy auditing:** Loss-based membership inference, ROC-AUC, TPR/FPR metrics, and RMIA-style calibrated scores.
- **Experiment tracking:** YAML configurations, cached logits, manifests, and CSV summaries.
- **Data leakage prevention:** Text normalization, duplicate detection, conflicting-label removal, and family-aware data splitting.

## Methodology

### 1. Model Architecture

The project uses `roberta-base`, which contains 12 Transformer encoder blocks.

The embedding layer is frozen, and the classifier remains trainable in every configuration.

### 2. Experimental Configurations

| Experiment | Trainable Transformer Blocks | Purpose |
|---|---|---|
| Full Clean | All 12 blocks | Non-private utility baseline |
| Full DP | All 12 blocks | Differentially private baseline |
| Head DP | Classifier only | Evaluate parameter reduction |
| Early-3 | 0, 1, 2 | Investigate early layers |
| Mid-3 | 4, 5, 6 | Investigate middle layers |
| Late-3 | 9, 10, 11 | Investigate late layers |
| DP-SelFT | 3, 6, 4 | Investigate private layer selection |
| H' Control | 9, 10, 11 | Matched privacy-budget control |

### 3. Differentially Private Training

DP-SGD modifies standard gradient-based training through the following steps:

1. Compute per-example gradients.
2. Clip gradients to a predefined norm.
3. Aggregate the clipped gradients.
4. Add calibrated Gaussian noise.
5. Update model parameters.

The main experiments target:

- Privacy budget: `epsilon = 5.0`
- Failure probability: `delta = 1e-5`

The DP-SelFT branch allocates part of the privacy budget to layer selection and the remainder to final fine-tuning. The actual guarantee depends on the implementation and privacy accounting.

### 4. Private Layer Selection

The DP-SelFT-style pipeline uses public IMDb candidate examples and semantic embeddings to construct a differentially private proxy dataset.

The selection process consists of:

1. Embedding public candidate examples.
2. Matching private tuning examples to candidate examples.
3. Building a candidate vote histogram.
4. Adding Gaussian noise to the histogram.
5. Constructing a proxy dataset from selected candidates.
6. Evaluating candidate layers on the proxy dataset.

The project walkthrough identifies layers 3, 6, and 4 as the selected set in its described layer tournament. This selection outcome must be validated through final experiments.

## Privacy Evaluation

### Utility Metrics

- Validation accuracy
- Classification performance across experimental configurations

### Privacy Metrics

- **Membership Inference AUC:** Measures how well an attack distinguishes members from non-members. An AUC near 0.5 indicates random-ranking performance.
- **TPR at 1% FPR:** Measures membership detection at a low false-positive rate.
- **Loss-based membership scores:** Use model loss to estimate whether an example was part of training.
- **RMIA-style calibrated scores:** Compare target-model behavior with reference models and population examples.
- **Differential privacy accounting:** Tracks the formal privacy guarantee using epsilon, delta, and the training configuration.

Formal differential privacy and empirical attack resistance are complementary measurements and should be reported separately.

## Project Structure

```text
privacy-llm-finetuning/
├── configs/
│   ├── baseline_full_clean.yaml
│   ├── baseline_full_dp.yaml
│   ├── head_only_dp.yaml
│   ├── selective_early_k3.yaml
│   ├── selective_mid_k3.yaml
│   ├── selective_late_k3.yaml
│   ├── selective_dpselft_k3.yaml
│   ├── control_H_prime.yaml
│   └── selection_seed42.json
├── data/
│   └── dp_proxy.parquet
├── src/
│   ├── data.py
│   ├── models.py
│   ├── dp_proxy.py
│   ├── select_layers.py
│   ├── train.py
│   └── audit.py
├── run_experiment.py
├── results/
│   ├── logits/
│   └── summary.csv
└── manifests/
    └── manifest_seed42.json
```

*This structure reflects the supplied project walkthrough. Verify file names against the current repository.*

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd privacy-llm-finetuning
```

Set up a Python environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies using the repository's dependency file, if available:

```bash
pip install -r requirements.txt
```

The supplied walkthrough does not specify a verified dependency list. Confirm that the repository contains `requirements.txt` or use its actual environment configuration before running this command.

## Running Experiments

Run an experiment using a YAML configuration:

```bash
python run_experiment.py --config configs/selective_late_k3.yaml
```

Other configurations include:

```bash
python run_experiment.py --config configs/baseline_full_clean.yaml
python run_experiment.py --config configs/baseline_full_dp.yaml
python run_experiment.py --config configs/head_only_dp.yaml
python run_experiment.py --config configs/selective_early_k3.yaml
python run_experiment.py --config configs/selective_mid_k3.yaml
python run_experiment.py --config configs/selective_dpselft_k3.yaml
python run_experiment.py --config configs/control_H_prime.yaml
```

Run these commands from the repository root. They assume the listed configuration files exist in the current checkout.

## Execution Pipeline

The standard experiment follows this workflow:

1. Load the experiment configuration.
2. Prepare and validate dataset splits.
3. Initialize the RoBERTa model.
4. Freeze unselected layers and enable the selected parameters.
5. Train using standard optimization or DP-SGD.
6. Evaluate the model and cache logits.
7. Repeat training for the independent reference model.
8. Run membership inference and RMIA-style privacy audits.
9. Save experiment metrics and metadata.

The output summary is written to:

```text
results/summary.csv
```

Cached logits are stored under:

```text
results/logits/
```

Exact output names depend on the current implementation and configuration.

## Interpreting Results

- Higher validation accuracy indicates better task utility.
- Membership inference AUC closer to 0.5 indicates weaker discrimination by the evaluated attack.
- Lower TPR at a fixed FPR generally indicates less membership signal at that operating point.
- The clipping fraction indicates how frequently per-example gradients exceed the clipping threshold.
- Privacy metrics must be interpreted alongside the accountant settings and training configuration.

The H' control is particularly important because it matches the DP-SelFT fine-tuning privacy budget, helping distinguish the effect of layer selection from the effect of the privacy budget.

## Reproducibility

For meaningful comparisons, record:

- Random seeds
- Dataset versions and split sizes
- Model configuration and selected layers
- Batch sizes and training steps
- Learning rate and optimizer settings
- Gradient clipping norm
- Noise multiplier and privacy accountant settings
- Selection-stage and fine-tuning privacy budgets
- Validation metrics and membership inference results
- Software versions and run manifests

Smoke tests should be treated as pipeline checks rather than final scientific results. Confirm that the training steps, data counts, logits, privacy accounting, and result fields are valid before interpreting an experiment.

## Limitations

- Selective fine-tuning does not automatically guarantee stronger privacy.
- A low membership inference AUC does not establish formal differential privacy.
- Layer-selection results require validation using matched final experiments.
- Privacy guarantees depend on correct implementation and accounting.
- Results may vary with random seeds, data splits, hyperparameters, and computational settings.

## Research Question

Can a small, carefully selected subset of Transformer layers preserve useful learning signal while reducing the cost of differentially private optimization?

This project investigates that question by combining selective fine-tuning, DP-SGD, private layer selection, and empirical privacy auditing.

## Acknowledgements

This project builds on research and open-source tools for Transformer fine-tuning, differential privacy, and privacy auditing, including RoBERTa, DP-SGD, and Opacus.

