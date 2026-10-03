# ARCHITECTURE: Fraction and Location of Trainable Parameters vs. Empirical Privacy Leakage

**Status:** Phase 1 (theory, module contracts, data contract). No task cards.
**Version:** 1.0
**Related work:**
- [M26] Marek et al., *Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models*, ICLR 2026.
- [S26] Sha et al., *DP-SelFT: Differentially Private Selective Fine-Tuning for Large Language Models*, 2026.

**Research question.** How do the **fraction** (how many) and **location** (early, mid, late layers) of trainable parameters affect empirical privacy leakage (MIA ROC-AUC) and downstream utility in LLM fine-tuning?

> **Provenance (confirmed against the papers).**
> - [S26]: Theorem 1 (informal, §4.4, p. 8) and Theorem F.1 (formal, App. F.1, p. 18). Noise `z_Λ ~ N(0, σ²C² I_{d_Λ})` (Eq. 27, 39). The expected one-step degradation contains the noise-damage term `(βη²/2)·d_Λ·σ²·C²` (Eq. 40/44). Signal retention ratio `s(Λ) = ‖∇_{θ_Λ}F_pri‖² / ‖∇_θ F_pri‖²` (Eq. 43). This is the quantity written `ρ_Λ` below, so **`ρ_Λ ≡ s(Λ)`**.
> - [M26]: the attack is **offline RMIA**, a likelihood-ratio test (App. A.2, Eq. 5-6, p. 20). Canary exposure is Eq. 2 (p. 3) and Eq. 7 (p. 20).
> - Section 1.1 gives a self-contained derivation with a slightly simplified (idealized, unclipped) statement. Section 1.2 is an *adaptation* of [M26]. Our implementation is simpler than offline RMIA, and §1.2.3 says so.

---

## 0. Experimental Setting (fixed across all runs)

| Item | Value | Source in code |
|---|---|---|
| Backbone | `roberta-base` (L = 12 encoder blocks, hidden 768) | `--model_name` |
| Task | SST-2 (binary sentiment), GLUE | `load_dataset("nyu-mll/glue","sst2")` |
| Train set | first `N = --max_train_samples` (default 1024) examples of `train` | `select(range(n_train))` |
| Utility set | full SST-2 `validation` (872 examples) | `val_loader` |
| DP engine | Opacus `PrivacyEngine.make_private_with_epsilon` (DP-SGD, Poisson sampling, RDP/PRV accountant) | `run_experiment.py` |
| Optimizer | AdamW over `requires_grad` parameters only | `run_experiment.py` |

**Parameter inventory for `roberta-base` + `RobertaForSequenceClassification`.** These numbers are used as test oracles in §2.

| Component | Params |
|---|---|
| Embeddings (word, position, type, LayerNorm) | 39,000,576 (always frozen when selective) |
| One encoder block (attn 2,362,368 + FFN 4,722,432 + 2 LayerNorm 3,072) | **7,087,872** |
| 12 encoder blocks | 85,054,464 |
| Classification head (`classifier.dense` 590,592 + `classifier.out_proj` 1,538) | **592,130** |
| **Total `d`** | **124,647,170** |

The `head`-only configuration is therefore 592,130 / 124,647,170 = 0.475 %, which the CSV rounds to 0.48. This matches the existing rows of `results/summary.csv`.

---

## 1. Theoretical Foundations

### 1.1 One-step DP loss degradation and the layer-selection trade-off ([S26], Thm. 1 / F.1)

#### 1.1.1 Setup

- Parameters `θ ∈ ℝ^d`. Loss `L(θ)` is **β-smooth**: `‖∇L(x) − ∇L(y)‖ ≤ β‖x − y‖`.
- A **selection** `Λ` is a set of trainable coordinates, `|Λ| = d_Λ`.
  - `P_Λ` is the diagonal 0/1 projector onto `Λ`.
  - In this study `Λ` is a union of whole layers: layer index set `ℓ ⊆ {0,…,L−1}`, plus the head.
- `g = ∇L(θ)` is the (idealized) full gradient. `g_Λ = P_Λ g`.
- **Gradient alignment** (selection quality):

  ```
  ρ_Λ = s(Λ) = ‖g_Λ‖² / ‖g‖²  ∈ [0, 1]      ([S26] Eq. 43, signal retention ratio)
  ```

  `ρ_Λ = 1` means all gradient energy lies inside `Λ`. For `Λ = [d]`, `ρ = 1`.
- **DP-SGD step restricted to Λ.** Per-example gradients are clipped to norm `C`, summed, and noised. With clipping inactive (`‖∇ℓ_i‖ ≤ C`):

  ```
  θ⁺ = θ − η ( g_Λ + ξ ),    ξ ~ N(0, σ²C² · I_{d_Λ})  supported on Λ
  ```

  **Convention (§1.1.2).** Here `ξ` is the noise in the *update direction*. Opacus adds `N(0, σ²C²)` to the *sum* of clipped gradients and divides by the expected batch size `B`, so the per-coordinate std is `σC/B`.
  - [S26]'s `(βη²/2)·d_Λ·σ²C²` is recovered by absorbing `1/B` into `η` (η is the step on the summed gradient).
  - In implementation terms, replace `σ²C²` by `σ²C²/B²` when `η` multiplies the *averaged* gradient.
  - Every statement below holds under either convention, as long as it is applied consistently.

#### 1.1.2 Theorem (one-step expected loss, Λ-restricted DP step)

For β-smooth `L`:

```
E_ξ[ L(θ⁺) ]  ≤  L(θ)  −  η ‖g_Λ‖²  +  (βη²/2) ‖g_Λ‖²  +  (βη²/2) · d_Λ · σ² · C²
```

The negative term is the **descent** term. The last term is the **DP noise damage**.

**Proof.** Smoothness gives the quadratic upper bound `L(y) ≤ L(x) + ⟨∇L(x), y−x⟩ + (β/2)‖y−x‖²`. Take `y = θ⁺`, so `y − x = −η(g_Λ + ξ)`.
1. Linear term: `⟨g, −η(g_Λ+ξ)⟩ = −η‖g_Λ‖² − η⟨g_Λ, ξ⟩` (since `⟨g,g_Λ⟩ = ‖g_Λ‖²` and `ξ` is supported on `Λ`). `E[ξ] = 0` removes the cross term.
2. Quadratic term: `(β/2)η² E‖g_Λ+ξ‖² = (β/2)η²(‖g_Λ‖² + E‖ξ‖²)`, using `E[ξ]=0`.
3. `E‖ξ‖² = d_Λ σ² C²`. ∎

For `η ≤ 1/β` the bound simplifies to

```
E[L(θ⁺)] ≤ L(θ) − (η/2)·ρ_Λ·‖g‖²  +  (βη²/2)·d_Λ·σ²·C²
```

**Interpretation.**

| Term | Depends on | Effect of freezing layers |
|---|---|---|
| Descent `(η/2) ρ_Λ ‖g‖²` | alignment `ρ_Λ` (location) | Shrinks as `ρ_Λ` falls below 1 |
| Noise damage `(βη²/2) d_Λ σ² C²` | dimension `d_Λ` (fraction) | Shrinks **linearly** in `d_Λ` |

Noise damage is **independent of which** coordinates are chosen. It depends only on how many. Location matters only through `ρ_Λ`.

#### 1.1.3 When does selection beat full fine-tuning?

Compare the bounds for `Λ` (dimension `d_Λ`) and full (`d`, `ρ=1`), at `η ≤ 1/β`:

```
Bound(Λ) − Bound(full)  =  (η/2)(1 − ρ_Λ)‖g‖²  −  (βη²σ²C²/2)(d − d_Λ)
```

**Selection is predicted to win iff**

```
(1 − ρ_Λ) · ‖g‖²   <   β · η · σ² · C² · (d − d_Λ)          (★)
```

Consequences used to form hypotheses:
- **H1 (fraction):** At larger noise (`σ`, i.e. smaller ε) or larger `d`, the right-hand side of (★) grows, so more aggressive freezing is favored. At ε → ∞ (`σ → 0`), (★) fails for any `ρ_Λ < 1`, so full fine-tuning is favored. Non-private runs should show *no* advantage for selection.
- **H2 (location):** Under (★), the best `Λ` of a given `d_Λ` maximizes `ρ_Λ`. For pre-trained transformers, gradient energy per parameter is expected to be concentrated in later layers and the head. Prediction: `ρ_late ≥ ρ_mid ≥ ρ_early` at equal `d_Λ`, so late-layer selection gives better DP utility.
- **H3 (privacy side):** Freezing reduces the number of coordinates that can memorize training data. Without DP, the **MIA AUC** is expected to rise with `d_Λ`, and location may matter because later layers sit closer to the output logits (the loss attack's signal). Under DP, the formal ε caps AUC. The *empirical* AUC is expected to be near 0.5 regardless of `Λ`, so the Pareto differences come mostly from utility.

**Caveats (must be stated in the paper).**
- The theorem is a one-step, upper-bound statement.
- It ignores clipping bias and multi-step accumulation.
- It assumes `ξ` is isotropic over `Λ` (true for DP-SGD).
- `ρ_Λ` is not measured by the current pipeline. An optional diagnostic is *per-layer gradient norm at initialization*, which gives an estimate of `ρ_ℓ`.
- The adaptive optimizer (AdamW) rescales per-coordinate step sizes. This breaks the plain-SGD form above, and the bound is used as a **qualitative guide**, not a quantitative predictor.

### 1.2 Empirical membership inference formulation (adapted from [M26])

#### 1.2.1 Definitions

- Training set `D_train` (members, `|D_train| = N`).
- Held-out set `D_out` (non-members), **disjoint** from `D_train` and drawn from the same distribution.
- Model `f_θ`. Per-example loss `ℓ(θ; x, y) = −log softmax(f_θ(x))_y`.
- Membership label `m(z) = 1` if `z ∈ D_train`, else 0.

**Likelihood-ratio attack (general form).** For a sample `z`, with `p_in(·)` and `p_out(·)` the distributions of the loss statistic when `z` is in or not in training,

```
Λ_LR(z) = p_in(ℓ(θ;z)) / p_out(ℓ(θ;z))        predict "member" iff Λ_LR(z) > τ
```

**Our instantiation (population-level, loss-threshold).** We use the monotone score

```
s(z) = −ℓ(θ; z)
```

A threshold on `s` is the likelihood-ratio test when `p_in` and `p_out` are such that the ratio is monotone decreasing in loss. This holds for fine-tuned classifiers, whose members have lower loss.

#### 1.2.2 Metric

Sweep `τ` over all thresholds to obtain `(FPR(τ), TPR(τ))`.

```
MIA_AUC = ∫₀¹ TPR(FPR) dFPR  = P( s(z_in) > s(z_out) ) + ½ P( s(z_in) = s(z_out) )
```

where `z_in ~ D_train`, `z_out ~ D_out` are independent draws. This is the **Mann-Whitney / ROC-AUC** estimator, computed with `sklearn.metrics.roc_auc_score`.

- `0.5` means no leakage (attacker is no better than chance).
- `1.0` means total leakage.
- `< 0.5` is possible only through sampling noise or a degenerate model. It is reported as is, never clipped.

**Derived leakage metrics** (computed in analysis, not stored unless added in v2):

```
Adv = 2·|MIA_AUC − 0.5|            (attacker advantage, ∈ [0,1])
```

**Sampling uncertainty.** With `n_in = n_out = 200`, `SE(AUC) ≈ 0.025` near 0.5. Differences in AUC below about 0.05 are **not** significant for a single seed. Claims require multiple seeds (§3, `seed`) or larger audit sets.

#### 1.2.3 Deviations from [M26] (explicit)

| Aspect | [M26] | This study | Impact |
|---|---|---|---|
| Attack strength | Stronger (calibrated/likelihood-ratio style attacks, reference/shadow information as in their benchmark) | Single-model, global loss threshold (Yeom-style) | AUC is a **lower bound** on leakage. Absence of leakage is *not* a proof of privacy. |
| Non-member source | Held-out data from the same pool / canaries | SST-2 `validation[:200]` | Valid only if the validation set is not used for model selection |
| Metric | ROC-AUC (+ low-FPR regime) | ROC-AUC only | Low-FPR TPR (e.g. TPR@1%FPR) is a recommended v2 addition |

### 1.3 Privacy-Utility Pareto frontier

For each configuration `c = (Λ, DP setting, seed)` a run produces a point

```
P(c) = ( u(c), a(c) )     u = val_accuracy (maximize),   a = mia_auc (minimize toward 0.5)
```

Leakage is measured as distance from chance, `λ(c) = |mia_auc(c) − 0.5|`, so that AUC below 0.5 (noise) is not rewarded.

**Dominance.** `c₁ ≻ c₂` iff `u(c₁) ≥ u(c₂)` and `λ(c₁) ≤ λ(c₂)`, with at least one strict inequality.

**Pareto frontier.** `F = { c ∈ 𝒞 : ¬∃ c' ∈ 𝒞, c' ≻ c }`, with `𝒞` the set of **valid** runs (see the validity gates in §3.3).

**Reporting.**
1. Plot `λ` (x) against `u` (y) per configuration family (`early`, `mid`, `late`, `head`, `all`) and per `target_epsilon ∈ {∞ (no DP), …}`.
2. Report the **frontier set** and its **hypervolume** w.r.t. reference point `(λ_ref = 0.5, u_ref = 50)` (majority-class accuracy on SST-2 is about 50 %).
3. Aggregate over seeds as mean ± 95 % CI. Dominance is declared only when the CIs of *both* axes separate.
4. Secondary frontier in `(param_fraction_%, u)` and `(param_fraction_%, λ)` to answer the "how many" part of the question directly.

---

## 2. System Module Boundaries and Invariants

```
run_experiment.py  ──►  src/models.py   (get_model)
        │
        └────────────►  src/audit.py    (run_mia_audit)
        │
        └────────────►  results/summary.csv   (append-only)
```

Dependency rule: `src/*` modules MUST NOT import from `run_experiment.py`, and must not write files.

### 2.1 `src/models.py`: selective layer freezing and parameter counting

**Function:** `get_model(model_name: str, num_labels: int, train_layers: str) -> (model, trainable_params: int, fraction: float)`

**Inputs**

| Name | Type | Domain |
|---|---|---|
| `model_name` | str | HF identifier of a RoBERTa-family checkpoint (current implementation is **RoBERTa-specific**: it accesses `model.roberta.encoder.layer` and `model.classifier`) |
| `num_labels` | int | ≥ 2 (SST-2: 2) |
| `train_layers` | str | Grammar: `"all"` \| `"head"` \| `idx(,idx)*` with `idx ∈ {0,…,L−1}`, no duplicates |

**Outputs**
- `model`: `RobertaForSequenceClassification` with `requires_grad` set per the configuration.
- `trainable_params = Σ numel(p) for p with requires_grad`.
- `fraction = 100 · trainable_params / total_params` (**percent**, not a ratio).

**Canonical configurations** (for L = 12):

| Family | `train_layers` | Trainable |
|---|---|---|
| Baselines | `head`, `all` | 592,130 (0.48 %) / 124,647,170 (100 %) |
| Early | `0,1,2` | head + 3·7,087,872 = 21,856,766 (17.54 %) |
| Mid | `4,5,6` | 21,856,766 (17.54 %) |
| Late | `9,10,11` | 21,856,766 (17.54 %) |
| Fraction sweep (late-anchored) | `11`; `10,11`; `9,10,11`; `8,…,11`; … | head + k·7,087,872 |

Early, mid and late sets have **equal `d_Λ`**, which isolates *location* from *fraction* (the key controlled comparison, per (★)).

**Invariants**
1. **I-M1 (head always trainable).** For every `train_layers`, all `classifier.*` parameters have `requires_grad=True`.
2. **I-M2 (exact freezing).**
   - Under a layer selection `ℓ`, a parameter is trainable iff it is in `classifier.*` or in `encoder.layer[i].*` for `i ∈ ℓ`.
   - Embeddings, the pooler (if present) and all other blocks are frozen.
3. **I-M3 (counts).** `0 < trainable_params ≤ total_params`, and `0 < fraction ≤ 100`.
   - `total_params` is computed **before** freezing, from `model.parameters()`, and is constant for a given `model_name` and `num_labels`.
   - `fraction = 100` iff `train_layers == "all"`.
4. **I-M4 (additivity).** For a layer set `ℓ`:
   `trainable_params = 592,130 + |ℓ| · 7,087,872` (RoBERTa-base). Used as a unit-test oracle.
5. **I-M5 (monotonicity).** `ℓ₁ ⊂ ℓ₂ ⇒ trainable(ℓ₁) < trainable(ℓ₂)`.
6. **I-M6 (parsing safety).**
   - Out-of-range, negative or duplicate indices MUST raise `ValueError` (the current code raises `IndexError` or silently accepts duplicates, a known gap).
   - Whitespace is tolerated.
7. **I-M7 (determinism).** The classification head is randomly initialized. The caller MUST seed (`torch.manual_seed(seed)`) **before** `get_model` so that results are reproducible.
8. **I-M8 (DP compatibility).** Only `requires_grad` parameters receive per-sample gradients, so the optimizer MUST be built from those parameters only. Frozen blocks run in forward mode without storing their per-sample activations for gradients, which is the intended compute and memory saving.

### 2.2 `src/audit.py`: MIA evaluation and ROC-AUC

**Functions**
- `compute_sample_losses(model, dataset, device, batch_size=32, collate_fn=None) -> np.ndarray, shape (n,)`
- `run_mia_audit(model, train_subset, test_subset, device, collate_fn=None) -> float`

**Inputs**
- `model`: a *plain* `nn.Module` (the **unwrapped** module when Opacus's `GradSampleModule` is used).
- `train_subset`: members, a **subset of the actual training set**.
- `test_subset`: non-members, drawn from a split **not used in training or in hyperparameter selection**.
- `collate_fn`: pads batches. Required for variable-length tokenization.

**Outputs:** `mia_auc ∈ [0, 1]`, a float.

**Invariants**
1. **I-A1 (per-sample, unreduced loss).** Loss uses `reduction="none"`, so `len(losses) == len(dataset)` and order matches dataset order (`shuffle=False`).
2. **I-A2 (no side effects).** Evaluation uses `model.eval()` and `torch.no_grad()`. No parameter, gradient or dropout state is changed. Calling the audit twice returns identical AUCs (up to GPU nondeterminism).
3. **I-A3 (label convention).** Members = 1, non-members = 0, score = `−loss`. Higher score means more likely to be a member.
4. **I-A4 (disjointness).** `train_subset ∩ test_subset = ∅`. Members are real training examples, so a **sample-size bound** applies: `n_in ≤ N`.
5. **I-A5 (both classes present).** `n_in ≥ 1` and `n_out ≥ 1`, otherwise `roc_auc_score` raises. Recommended: `n_in = n_out` (balanced).
6. **I-A6 (range and symmetry).** `0 ≤ AUC ≤ 1`. Swapping the labels gives `1 − AUC`. Adding a constant to all losses leaves AUC unchanged (rank-based).
7. **I-A7 (finite values).** Losses are finite. NaN or inf MUST abort the run.
8. **I-A8 (statistical resolution).** With `n = 200` per class, `SE ≈ 0.025`. Conclusions need multi-seed aggregation.
9. **I-A9 (train-mode leakage).** The audit must be run **after** the model is put in eval mode and **after** training ends. Auditing a model that is still in train mode (dropout on) would add noise to the AUC.

### 2.3 `run_experiment.py`: Opacus DP integration, training and evaluation

**Role:** orchestrator. Parses the configuration, builds data and model, optionally wraps with Opacus, trains, evaluates utility and privacy, appends one CSV row.

**Inputs (CLI)**

| Arg | Type | Default | Constraint |
|---|---|---|---|
| `--model_name` | str | `roberta-base` | RoBERTa-family |
| `--train_layers` | str | `all` | grammar in §2.1 |
| `--use_dp` | flag | off | |
| `--target_epsilon` | float | 5.0 | > 0 (used only if `--use_dp`) |
| `--target_delta` | float | 1e-5 | `0 < δ < 1/N`, with `N = n_train` |
| `--max_grad_norm` (C) | float | 1.0 | > 0 |
| `--epochs` | int | 2 | ≥ 1 |
| `--batch_size` | int | 32 | ≥ 1 |
| `--lr` | float | 2e-5 | > 0 |
| `--max_train_samples` | int | 1024 | ≥ 1 |
| `--exp_name` | str | `test_run` | non-empty, no commas |

**Outputs:** one appended row in `results/summary.csv` (§3). Console logs.

**Pipeline and invariants**
1. **I-R1 (data split).** `D_train = train[:N]`, utility set = `validation`. The audit members are a subset of `D_train`. The non-members are from the validation set. **The validation set must not be used for early stopping or tuning.**
2. **I-R2 (model before engine).** `get_model` runs first. Freezing is applied **before** `make_private_with_epsilon`.
3. **I-R3 (DP wrapping).**
   - Only when `use_dp`: `ModuleValidator.fix` then `make_private_with_epsilon`.
   - The optimizer is built from the **trainable** parameters (and rebuilt after `fix` if the module is replaced).
   - `sample_rate = batch_size / N`.
   - The accountant calibrates `σ` such that after `epochs · (1/sample_rate)` steps, `ε(δ) ≤ target_epsilon`.
4. **I-R4 (privacy guarantee).**
   - The guarantee holds for the (ε, δ) DP of the training procedure **with respect to `D_train`** under Poisson sampling.
   - It does not cover the choice of `Λ` or hyperparameters if those were tuned on private data. In our study `Λ` is fixed *a priori*, a documented limitation vs. [S26], which selects layers privately.
   - Realized `ε_spent ≤ target_epsilon` MUST be recorded (see `epsilon_spent`, v2).
5. **I-R5 (clipping).** `max_grad_norm = C` clips per-sample gradients over the **trainable** parameters only. For the same `C`, the clipped fraction is therefore lower for small `Λ`. This is a confound to report.
6. **I-R6 (non-DP parity).** The non-DP run uses the same optimizer, learning rate, epochs, batch size and seed. The only difference is the Opacus wrapper. Otherwise the DP vs. non-DP comparison is invalid.
7. **I-R7 (training loop).** For each epoch: `zero_grad → forward → loss → backward → step`. The loss is the mean cross-entropy over the (logical) batch. Under Poisson sampling, batch sizes vary and empty batches must be handled by Opacus.
8. **I-R8 (evaluation order).** Switch to `model.eval()`, evaluate accuracy on the full validation set, **then** call the audit on `model._module` (the unwrapped module when DP is on).
9. **I-R9 (single writer).** The CSV append happens **once**, at the end of a *successful* run. A failed run MUST NOT write a partial row.
10. **I-R10 (reproducibility).** Every run is a function of `(config, seed)`. Seeds MUST be set for `torch`, `numpy`, `random` and CUDA, and be logged (v2 `seed`).

---

## 3. Data Contract: `results/summary.csv`

**Format.** UTF-8, comma-separated, header on the first line, **append-only**, one row per completed run, `\n` line terminator. The header is written once (when the file does not exist).

### 3.1 Schema v1 (currently emitted by `run_experiment.py`)

| # | Field | Type | Value range | Description |
|---|---|---|---|---|
| 1 | `exp_name` | string | non-empty, no commas, ≤ 128 chars | Human-readable run identifier. Should be unique per `(config, seed)`. |
| 2 | `train_layers` | string | `all` \| `head` \| comma-separated ints in [0, L−1] (e.g. `9,10,11`). **Quote the field when it contains commas.** | Trainable-layer selection (**location**). |
| 3 | `trainable_params` | int64 | 1 ≤ x ≤ 124,647,170 (roberta-base) | Count of parameters with `requires_grad=True`. |
| 4 | `param_fraction_%` | float | (0, 100], 2 decimals | `100·trainable_params/total_params` (**fraction**). Percent, not ratio. |
| 5 | `use_dp` | bool | `True` \| `False` | Whether DP-SGD (Opacus) was active. |
| 6 | `target_epsilon` | float or the literal string `None` | ε > 0, or `None` when `use_dp=False` | Requested ε at `target_delta`. **It is the target, not the spent budget.** |
| 7 | `val_accuracy` | float | [0, 100], 2 decimals | Accuracy (%) on the full SST-2 validation set (872 examples). |
| 8 | `mia_auc` | float | [0, 1], 4 decimals | Loss-based MIA ROC-AUC (§1.2). 0.5 means no leakage. |

**Consistency constraints on v1 rows**
- C1: `use_dp=False ⇔ target_epsilon="None"`.
- C2: `param_fraction_% = round(100·trainable_params/124647170, 2)` for roberta-base.
- C3: `train_layers="all" ⇒ param_fraction_%=100.00`.
- C4: For layer set `ℓ`, `trainable_params = 592130 + |ℓ|·7087872` (I-M4).
- C5: `exp_name` is unique per file. A duplicate indicates an accidental re-run, so keep the latest and flag the rest.

### 3.2 Schema v2 (required additions for publication-grade analysis)

Appended **after** the v1 columns, so v1 readers keep working. Rows written before v2 have these fields empty (`NaN`).

| Field | Type | Range | Description |
|---|---|---|---|
| `seed` | int | ≥ 0 | Global RNG seed. Needed for CIs. |
| `model_name` | string | HF id | Backbone. |
| `n_train` | int | ≥ 1 | Training set size `N` (`--max_train_samples`). |
| `epochs` | int | ≥ 1 | Training epochs. |
| `batch_size` | int | ≥ 1 | Logical batch size. |
| `lr` | float | > 0 | Learning rate. |
| `target_delta` | float | (0, 1/N) or empty | δ for DP. |
| `max_grad_norm` | float | > 0 or empty | Clipping norm `C`. |
| `noise_multiplier` | float | ≥ 0 or empty | Realized `σ` (accountant output). Needed to evaluate (★). |
| `epsilon_spent` | float | (0, target_epsilon] or empty | Realized ε. |
| `final_train_loss` | float | ≥ 0 | Mean loss of the last epoch (a divergence check). |
| `n_audit_in` / `n_audit_out` | int | ≥ 1 | Audit set sizes (to compute SE). |
| `mia_tpr_at_1fpr` | float | [0, 1] | TPR at FPR = 1 % (low-FPR leakage, as in [M26]). |
| `layer_family` | string | `all` \| `head` \| `early` \| `mid` \| `late` \| `custom` | Derived label for grouping. |

### 3.3 Validity gates (rows failing a gate are kept but excluded from Pareto analysis)

| Gate | Condition | Reason |
|---|---|---|
| G1 (learned) | `val_accuracy` > 55 | SST-2 is balanced (about 50 % majority class). The existing rows show `val_accuracy = 49.08`, a **chance-level result**, so the model did not learn. The low learning rate with a head-only model and 2 epochs, or an under-trained head, is the likely cause. |
| G2 (AUC meaningful) | `mia_auc` is measured on a *trained* model | An AUC of about 0.60 on an untrained or chance-level model (row `smoke_test`) reflects initialization artifacts, **not** memorization. |
| G3 (budget) | `epsilon_spent ≤ target_epsilon` | DP guarantee holds. |
| G4 (finite) | No NaN / inf in any numeric field | |
| G5 (replication) | ≥ 3 seeds per configuration for any claim | `SE(AUC) ≈ 0.025` per run. |

> **Observation on the current file.** Both existing rows (`smoke_test`, `kaggle_gpu_head_eps5`) fail G1. They are pipeline smoke tests only and MUST NOT appear in reported results. Phase 2 should address the training-recipe issue (learning rate, epochs and head-only capacity under DP) before running the sweep.

---

## 4. Known Gaps Between Spec and Current Code (to be resolved in later phases)

| ID | Gap | Section |
|---|---|---|
| K1 | `get_model` raises `IndexError` for bad indices and accepts duplicates (I-M6) | 2.1 |
| K2 | `get_model` is RoBERTa-only (hard-coded `model.roberta`, `model.classifier`) | 2.1 |
| K3 | `target_epsilon` is logged, not the realized `epsilon_spent`, and `noise_multiplier` is not logged | 3.2 |
| K4 | No global seeding, so runs are not reproducible (I-R10) | 2.3 |
| K5 | Audit uses 200 + 200 examples (SE ≈ 0.025) | 1.2.2 |
| K6 | The validation set serves as both the utility set and non-member pool. Fine only if never used for tuning | 2.3 |
| K7 | Single global-threshold loss attack, not offline RMIA (App. A.2 of [M26]). Implementing RMIA (reference models) and canary exposure is future work | 1.2.3 |
| K8 | Layers are selected *a priori*, not with DP selection as in [S26] | 2.3 |
| K9 | v1 CSV omits the hyperparameters needed to identify a run | 3.2 |
| K10 | Existing rows show chance-level accuracy (G1) | 3.3 |
