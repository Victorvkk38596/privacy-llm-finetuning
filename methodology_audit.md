# Methodological Audit: Critiques 1–4 (PI Review, pre-implementation)

**Evidence base.** Text extracted from the two PDFs in the repo root:
- Marek et al.: `2606.09401v1.pdf` (§3–4.4, App. A.2, B.2–B.4, Alg. 1).
- Sha et al.: `2605.17432v1.pdf` (§4.2–5.6, App. A–B).

Quotes and numbers below come from those texts. Where I estimate something, I say so.

**Bottom line.**

| # | Critique | Verdict |
|---|---|---|
| 1 | Attack fidelity | **Accept the problem, reject the proposed fix.** Using the pretrained `roberta-base` as the reference is unsound for a classifier. A **shadow-model** reference is required. I also found a more serious, independent flaw: the member and non-member sets come from different distributions. |
| 2 | Not actually testing DP-SelFT | **Accept.** It can be replicated faithfully without an API by using a public candidate pool plus DP histogram voting (ε_sel = 0.3). Two caveats: k = 3 is outside the regime the paper validated, and with equal-size layers DP-SelFT's dimension penalty is constant across layers. |
| 3 | 160 steps is too few | **Accept the conclusion, correct the reasoning.** At fixed (ε, δ, N), the aggregate signal-to-noise ratio is roughly *independent* of T. The real problems are optimization budget, learning rate, and Opacus memory. A single learning rate shared by DP and non-DP runs (my own invariant I-R6) is wrong. |
| 4 | Classification vs. generation | **Accept as a scoped choice.** It matches Paper 2 exactly, and RMIA was originally a classification attack. It must be framed explicitly, and canary-style auditing can partly close the gap. |

**TASK_01 should be put on hold.** Running it as written would produce results that a reviewer would reject on the grounds of Critiques 1 and 3 and of finding N1 below.

---

## Critique 1: Attack fidelity

### 1.1 What Marek et al. actually show (this corrects the critique's premise)

The critique says that naive loss gives "AUC around 0.50" while RMIA gives "0.70–0.90" under DP. That holds **only for RMIA with a *shadow* reference model**, defined as a model "trained in the same way as the target model, but on a different split of the same fine-tuning data" (§3, p. 4). When the **pretrained model** is the reference, their own tables show near-chance results under DP:

| Pythia 1B, ε = 8 | OOD avg (Tab. 2) | IID avg (Tab. 3) |
|---|---|---|
| RMIA (shadow) | 0.73 | 0.78 |
| Reference (pretrained Pythia 1B) | 0.51 | 0.55 |
| RMIA with pretrained models as reference (Fig. 2, RQ4) | AUC axis 0.49–0.54 | — |

RQ4 states: "RMIA's effectiveness rapidly deteriorates as shadow models are trained on different distributions… The MIA performance is close to random for private adaptations with ε = 8" when no in-distribution shadow is available.

### 1.2 Proposal: pretrained `roberta-base` as reference with validation data as population π. **Verdict: unsound, for three reasons.**

1. **The reference has a random classification head.** `RobertaForSequenceClassification` loaded from the checkpoint has an *untrained* `classifier`. Its SST-2 loss is roughly ln 2 plus noise from the random head, and carries no information about how hard each sample is. The "difficulty calibration" that justifies RMIA therefore does not exist.
   - Marek can use the pretrained LM as a reference because next-token loss is meaningful before adaptation. Our cross-entropy over a new label space is not.
   - A random-head reference reduces RMIA to a noisy monotone transform of the naive loss attack.
2. **Even with a meaningful reference, a pretrained reference is the *weak* regime** in Marek (above). Reporting it as "RMIA as in Marek" would misrepresent their method.
3. **The population π is drawn from the wrong distribution.** See finding N1 below.

**Required design: a shadow reference trained by the identical recipe on a disjoint, same-distribution split.** This is Marek's main setting (k = 1 shadow).

**Swap pairing removes wasted compute.** For each (config, seed), train two models with the same recipe: `M_A` on `D_A` and `M_B` on `D_B` (disjoint).
- `M_B` is the reference when auditing `M_A`, and vice versa.
- Every training run is both a target and a shadow, so there is no compute overhead per audited target and each run yields 2 audited targets.

### 1.3 Faithful Algorithm 1, with three details the critique's formula drops

From App. A.2 / Alg. 1 (p. 21), with k = 1 reference θ′:

```
L_OUT(x) = L(x | θ′)
L̃(x)     = ½[(1+α)·L_OUT(x) + (1−α)]          # offline approximation of the IN/OUT mixture
Ratio_x  = L(x | θ) / L̃(x)
Ratio_z  = L(z | θ) / L(z | θ′)               # for z ∈ Z ⊂ π
Score(x) = (1/|Z|) · #{ z ∈ Z : Ratio_x / Ratio_z < γ }
```

1. **α is missing from the critique's formula.** It is the offline correction (Zarifzadeh et al.). Without it, the attack is not "offline RMIA".
2. **The paper is internally inconsistent.** Eq. 6 defines a *difference*, `L(θ|x) − L(θ|z)`, while Alg. 1 uses *ratios*. We implement Alg. 1 and record this choice in ARCHITECTURE.
3. **Classification-specific instability.** Cross-entropy goes to 0 for confident correct predictions, so loss ratios become `0/0`-like and ill-conditioned.
   - Marek moved to losses only because "the adapted RMIA … for LLMs for text generation is based on comparing loss values rather than output probabilities."
   - For classification, the **original Zarifzadeh form with `p_θ(y|x)` is the native, numerically stable choice**.
   - **Recommendation:** primary = original RMIA on `p(y|x)` with the α correction; secondary = Alg. 1 on losses with a floor ε_L = 1e-6 (fidelity to Marek). Both cost the same (§1.4).

**Choosing γ and α (a risk the critique overlooked).** Marek reports the *best AUC over a hyperparameter grid*. That selects on the evaluation labels and is optimistically biased.
- **Pre-register γ = 1.**
- **Choose α on a calibration audit of a disjoint shadow pair** (e.g. seed 0), never on the evaluated targets.
- Report the grid-maximum only as a labelled "oracle attacker" upper bound.

### 1.4 Complexity on a Kaggle T4

Let X = audit set (`n_in + n_out`), Z = population sample.

- **Forward passes:** `2 models × (|X| + |Z|)`. With |X| = 2048 (1024 in + 1024 out) and |Z| = 1024, that is about 6.1k forward passes of `roberta-base` on SST-2-length inputs (mostly < 64 tokens with dynamic padding). **Estimate: well under a minute on a T4** in fp16 and eval mode. This is negligible next to training.
- **Scoring:** a naive implementation is O(|X|·|Z|) = about 2·10⁶ comparisons, already trivial. It can be exact in **O((|X|+|Z|) log |Z|)**: sort the vector `Ratio_z` once, then `searchsorted(γ·…)` for each x. Memory is O(|X| + |Z|).
- **Caching:**
  - Reference-side vectors `L(·|θ′)` over X ∪ Z are reusable for every target that shares that reference.
  - Target-side vectors must be recomputed for each target.
  - Cache per-sample vectors (`loss`, `p_true`) for every (model, split) as `.npy` keyed by run ID. Then every attack (loss, Carlini ratio, both RMIA forms, any α or γ) is post-processing **without re-running models**.

**Revisit the audit size.** n = 500 is arbitrary. Members are bounded by N = 1024, so use **all 1024 members against 1024 non-members**, which gives SE(AUC) ≈ 0.013 near 0.5.
- TPR at 1% FPR then rests on about 10 non-members, which is too noisy. Report **TPR at 5% FPR** as the primary low-FPR metric and TPR at 1% as indicative only.

### 1.5 Report both attacks? **Yes. Report all three, plus derived quantities.**

| Attack | Role |
|---|---|
| Loss threshold (Yeom) | Reference-free baseline, the current implementation |
| Carlini Reference ratio (Marek Eq. 4) with the shadow | Isolates the benefit of calibration alone |
| Offline RMIA (shadow, pre-registered α and γ) | Primary leakage metric |
| **Calibration gap** = AUC_RMIA − AUC_loss | Our own measure of how much the naive attack under-reports leakage, per location and fraction |
| Generalization gap = mean OUT loss − mean IN loss | Mechanistic covariate |

### 1.6 Finding N1 (new and critical): members and non-members come from different distributions

GLUE SST-2 `train` contains **phrase-level fragments** taken from parse trees, many only a few tokens long. `validation` contains **872 full sentences**. Our members come from `train`; our non-members come from `validation`.
- A loss attack on such sets partly detects *distribution* (length, syntactic completeness), not *membership*.
- This plausibly explains `smoke_test`'s AUC = 0.60 on a model at chance accuracy.
- (The phrase vs. sentence structure is well known for GLUE SST-2. Confirm with a length histogram before implementation.)

**Fix.**
- Draw non-members **and** population from held-out parts of `train`, using a seeded partition.
- Exact-deduplicate the text first: SST-2 `train` has repeated phrases, and a duplicate across IN and OUT corrupts labels.
- Use `validation` **only** for utility.
- It is also too small anyway: it cannot supply 1024 non-members plus 1024 population samples.

---

## Critique 2: Are we testing DP-SelFT?

### 2.1 The actual algorithm (App. A–B), and the facts that change the plan

- **Q = singletons.** "Instead of evaluating arbitrary layer subsets, we evaluate each layer independently… temporarily unfreeze only that layer," rank layers, and take the top-k.
  - This is cheap: L = 12 temporary trainings for `roberta-base`.
- **Worst-case, not random, perturbation (Eq. 20).** `ξ* ∈ argmax_{‖ξ‖≤ρ_Λ} L(θ − η(g_Λ + ξ))`, with ρ_Λ matched to the DP noise scale.
  - To first order, `ξ* = −ρ_Λ · ∇L(θ−ηg_Λ)/‖∇L(θ−ηg_Λ)‖`. This is a SAM-style ascent costing one extra backward pass per step.
  - Matching the noise suggests `ρ_Λ ≈ E‖z_Λ‖ = σC√d_Λ / B` for the averaged update.
- **A structural insight for our study.** Every `roberta-base` block has exactly d_ℓ = 7,087,872 parameters, so **ρ_ℓ is identical for every candidate layer**.
  - DP-SelFT's dimension-aware term (Thm 1) therefore *does not discriminate among single layers*. The ranking is driven purely by learnability and robustness (curvature) under perturbation.
  - This cleanly separates "location" from "fraction": DP-SelFT is a **data-driven location oracle at fixed d_Λ**. That is exactly the comparison our research question needs.
- **Regime warning.** Their default is **k = 15 of 24 layers** (RoBERTa-Large). Fig. 2 reports that "smaller values lead to extremely low accuracy" for k < 9.
  - Our k = 3 of 12 (25%) is far below their validated regime. DP-SelFT may offer no advantage at k = 3.
  - This is a legitimate finding, but only if a k sweep is present (§2.4).
- **Hidden utility driver.** They "set the number of non-DP pre-fine-tuning steps to 300 by default" on synthetic data.
  - Their SST-2 accuracies (about 94% at ε = 5, RoBERTa-Large) are **not** comparable to ours. We should *not* pre-fine-tune: it adds a second training signal and confounds leakage attribution.

### 2.2 An API-free, faithful implementation (`src/select_layers.py`, design only)

DP-SelFT's Stage 1 (API generation) has **zero privacy cost** by construction: it never touches D_pri. **Replacing the API with a public candidate pool is therefore privacy-equivalent and methodologically faithful.**

| Stage | Paper | Ours |
|---|---|---|
| 1. Candidate pool C | GPT-4o / Claude generations from a task prompt | Public sentiment corpus sentences (e.g. IMDB or Yelp-polarity, split into sentences, 5–25 words). **Not** `rotten_tomatoes`: SST was built from Rotten Tomatoes reviews, which risks overlap with D_pri. |
| 2. DP select and label | Public encoder (sentence-t5-xl), nearest-neighbour votes, Gaussian histogram (sensitivity 1), top-k, noisy class-vote labels, ε_syn = 0.3 | Identical. Use sentence-t5-base or all-MiniLM if T4 memory or time is tight; document the swap. |
| 3. Perturbation-aware ranking | 7:3 synthetic train/val split; for each layer ℓ, a temporary train with `g + ξ*`; accuracy on synthetic val; top-k | Identical. Same optimizer and steps as a short final run (e.g. 100–200 steps). |
| 4. DP fine-tuning | DP-AdamW on Λ* with ε_ft = ε − ε_syn | Identical |

**Cost estimate:** encoding about 20–50k candidates plus 12 × (100–200) step trainings on about 700 samples. Tens of minutes on a T4. It runs once per (seed, ε).

### 2.3 Privacy accounting for selection

- **Running the ranking on a local split of private data without DP: invalid.** Λ* becomes a non-private function of D_pri, so the (ε, δ) claim for the released model is false.
  - The *empirical* leakage through the choice of Λ (one of 12 rankings) is far below what our audit can detect. "Undetectable" is still not "private", and a reviewer will flag it.
- **Selecting on a disjoint split of SST-2 treated as public, with ε_sel = 0:** valid only under an *explicit public in-distribution data assumption*. That assumption is stronger than DP-SelFT's and it changes the threat model. Acceptable as an ablation, not as "DP-SelFT".
- **Recommended, faithful option:** the DP histogram stage at ε_sel = 0.3 and δ_sel = δ_ft, with basic composition `ε_total = ε_sel + ε_ft` (their Eq. 30).
  - **Comparison policy:** match **total** ε = 5. DP-SelFT trains at ε_ft = 4.7 and heuristic configs at 5.0, which is the paper's own protocol.
  - Also log ε_ft separately so readers can see the cost of selection.

### 2.4 How the narrative changes

The paper moves from "depth sweep with a citation" to a **testable hypothesis about the coupling between utility-optimal and leakage-optimal selection**.
- **H4.** At fixed d_Λ and fixed ε, does a selector that is optimal for noise-robust utility choose layers with *higher* or *lower* empirical leakage than the heuristic late layers?
- **Mechanistic prediction (stated before the runs, to avoid HARKing):**
  - Loss-based and RMIA leakage is driven by the generalization gap.
  - Layers selected for *robustness to worst-case perturbation* sit in flatter regions, which generalize better, so the prediction is **equal or lower leakage at equal or better utility**: a Pareto improvement.
  - The competing prediction: layers with high signal retention s(Λ) also fit individual samples faster, which means **more leakage**.
  - Either outcome is publishable.
- **A likely null result to plan for:** on SST-2, DP-SelFT may simply choose the late layers. Then "the algorithm agrees with the heuristic" is the finding. The k sweep and an ε sweep are what make the study informative regardless.

---

## Critique 3: Training schedule and DP dynamics

### 3.1 The mathematics (a correction to the critique's SNR argument)

For the subsampled Gaussian mechanism with small sampling rate q = B/N, the RDP/moments accountant gives approximately

```
ε ≈ c · q · √(T · log(1/δ)) / σ     ⇒     σ ≈ c · (B/N) · √(T log(1/δ)) / ε
```

Noise std per coordinate of the *averaged* update is

```
σC/B ≈ c · C · √(T log(1/δ)) / (N ε)        (independent of B)
```

Over T steps, the summed signal is about `T·ηs‖g‖` and the summed noise is about `η·√T·(σC/B)·√d_Λ`, which is about `η·T·cC√(log(1/δ)·d_Λ)/(Nε)`. So:

```
SNR_total ≈ ‖g‖ · N · ε / ( c · C · √(d_Λ · log(1/δ)) )      (to first order, independent of T and B)
```

**Implications.**
1. **More steps do not buy SNR** at fixed ε, because σ grows as √T. The critique's mechanism ("160 steps can't overcome the noise") is not the right explanation.
2. The reasons 160 steps *is* too few are about **optimization**: a nonconvex fine-tune needs enough updates to move far at a stable learning rate. DP-SelFT's T = 1000 and B = 64 (62.5 epochs over 1024 samples) is the validated recipe, and we should match it to inherit comparability.
3. **Batch size is "free" in noise terms** (the per-step noise is independent of B). A larger B lowers sampling variance at fixed T. B = 64 is a reasonable match to Paper 2.
4. **N·ε dominates.** With N = 1024, the noise is about 65× worse than full SST-2 (67k). Gate thresholds must reflect this (§3.3).
5. The `√d_Λ` term is exactly Theorem 1's d_Λ σ²C² dependence, which keeps the theory section consistent.

### 3.2 Overlooked risks with real consequences

- **(R1) Opacus memory makes full DP fine-tuning with "all" infeasible.**
  - Per-sample gradients cost about `B × |θ_trainable| × 4` bytes. For "all" with B = 64, that is 64 × 124.6M × 4 ≈ **32 GB**. A T4 has 16 GB.
  - The word embedding (38.6M parameters) is the worst offender.
  - **Fix:**
    - (a) Freeze embeddings in "all". This is standard in DP NLP and is redefined as "all blocks + head", giving d = 85,646,594.
    - (b) Logical B = 64 via Opacus `BatchMemoryManager`, with physical batch 8–16.
    - (c) Optionally use Opacus ghost clipping (`grad_sample_mode="ghost"`) if the installed version supports it.
- **(R2) A single learning rate for DP and non-DP is wrong.**
  - DP-Adam fine-tuning typically needs about 10× the non-private learning rate. Non-DP full fine-tuning works around 2e-5 to 5e-5. DP with few layers or the head typically needs about 5e-4 to 1e-3. (These ranges are from the literature, not verified here.)
  - Marek tuned **each method and budget separately** to "comparable validation loss" (App. B.3).
  - **Revise I-R6:** parity means an *identical tuning protocol and budget*, not identical values.
- **(R3) Tuning privacy.** Hyperparameter search on D_pri leaks (Papernot & Steinke). Tune on a **disjoint `D_tune` partition of `train`, declared public**. Never tune on `validation` or audit splits.
- **(R4) Overfitting the clean baseline.** 62.5 epochs on 1024 samples without DP will push loss and RMIA AUC toward 1.0, consistent with Marek's ε = ∞ ≈ 1.00.
  - That is real and expected leakage, not a bug.
  - Report clean results at **both** the matched schedule and an early-stopped schedule (chosen on `D_tune`). This shows how much of the clean leakage is caused by the schedule.
- **(R5) Opacus API.** `make_private_with_epsilon(epochs=…)` takes an integer epoch count.
  - For step-based training, compute σ with `opacus.accountants.utils.get_noise_multiplier(target_epsilon, target_delta, sample_rate=B/N, steps=T)` and use `make_private(noise_multiplier=σ)`.
  - Verify after training that `epsilon_spent ≤ target`.
- **(R6) Clipping heterogeneity by location.** Per-sample gradient norms differ by depth, so at C = 1 the clip fraction differs between early, mid and late layers. That is a confound for location.
  - Log the median per-sample gradient norm and the clip fraction per run.
  - Note that Marek used C = 0.1 and DP-SelFT used C = 1. Under Adam the absolute value of C matters little once clipping is saturated. Keep C = 1 for Paper 2 comparability.

### 3.3 Recommended schedule and gates

- `--max_steps 1000` (replaces epochs), logical B = 64, physical batch 16 (8 for "all"), AdamW, constant learning rate with 5% warmup, C = 1, δ = 1e-5 (< 1/N).
- **Learning-rate pilot on D_tune** (seed 0 only):
  - DP: {1e-4, 3e-4, 1e-3} × {head, k3-late, all}.
  - Clean: {1e-5, 3e-5, 1e-4}.
  - Freeze the chosen value per *family* (head / k-layers / all) × {DP, clean}. Pre-register it.
- **Gates.** I cannot honestly guarantee "> 75% DP" in advance, given N·ε = 5120.
  - Keep A1 (clean full > 85%; likely around 88–91% for roberta-base at N = 1024, an estimate).
  - Replace A2 with **pilot-calibrated gates**: DP gate = (pilot accuracy for that family) − 3 points, with an absolute floor of 70% for k ≥ 3 layers and 60% for head-only.
  - Rationale: the gate exists to catch *broken* runs, not to force the result.
- **Kaggle compute (estimate):**
  - k3 DP: about 3–6 min per 1000 steps. "all" DP with virtual batches: about 15–25 min.
  - Main grid (§5): about 7 configs × 3 ε × 5 seeds × 2 (swap pair) ≈ 210 runs, mostly short. Roughly 15–25 GPU-hours total, which fits about one week of Kaggle quota (30 h).
  - The fraction sweep adds about 10 h.

---

## Critique 4: Task and architecture scope

### 4.1 Does sequence-classification MIA answer our question? **Yes, for the mechanism. No, for absolute risk in generative LLMs.**

What it answers well:
- Our question is *mechanistic*: how d_Λ and location shape leakage under DP.
- Theorem 1 is task-agnostic.
- Paper 2's entire evaluation is RoBERTa or OPT on SST-2, MNLI and similar classification tasks, so our setting **is** Paper 2's setting.
- RMIA, LiRA and Shokri-style attacks were defined for classification first. Marek's loss-based variant is the *adaptation* to LMs. Our primary attack (§1.3, the `p(y|x)` form) is the attack in its original habitat.

What it cannot answer:
- **Extraction and memorization risk.** A classifier emits one label, so the leakage channel per sample is about 1 bit. A generative model can regurgitate whole sequences.
- **Absolute AUC values.** These are not comparable to Marek's tables.
- **Scale.** At 125M parameters, `roberta-base` is a "pretrained transformer LM", not an "LLM" in the 2026 sense. Use precise wording in the title and abstract.

### 4.2 How to formalize it in ARCHITECTURE (proposed new §0.1 "Scope and Validity")

1. **Controlled-variable rationale.** Classification gives a fixed, low-dimensional output channel and a single scalar likelihood per sample. Any change in leakage across configurations is therefore attributable to *which parameters were trained*, not to differences in output length or decoding.
2. **Alignment statement.** "Our experimental unit (RoBERTa, SST-2, N = 1024, B = 64, T = 1000, C = 1, δ = 1e-5) replicates Sha et al. §5.1. Our audit follows Marek et al.'s strongest setting (offline RMIA with an in-distribution shadow, k = 1)."
3. **Construct validity.** Report a table of claims that *transfer*: relative ordering by d_Λ and by location, and the calibration gap. List the claims that *do not*: extraction, absolute AUC, behaviour at billion-parameter scale.
4. **Partial bridge (optional v2): label-flip canaries.** Insert m = 2 × 64 canaries with randomized labels (half IN, half OUT). Canary MIA AUC is a classifier analogue of Marek's canary exposure (Jagielski et al. 2020; Nasr et al. 2021) and approaches a worst-case audit.
5. **Explicit future work:** repeat the selection-location grid on a causal LM (Pythia-160M on Enron or SAMSum) with Marek's exposure metric.

---

## 5. Other risks the critiques did not raise

| ID | Risk | Mitigation |
|---|---|---|
| N1 | Member and non-member sets from different distributions (§1.6) | Partition `train` into seeded, disjoint splits; use `validation` for utility only |
| N2 | Duplicate phrases in SST-2 `train` crossing IN and OUT | Exact-deduplicate the text before partitioning |
| N3 | Seeds vary only initialization; `D_train` is always `train[:1024]` | The seed controls the **data partition** as well as initialization and sampling |
| N4 | Statistical power. Under DP, location effects in AUC may be ≤ 0.02. SE per run ≈ 0.013; seed-to-seed SD unknown | **5 seeds** (×2 from swap pairs = 10 targets per cell). Paired comparisons across configs within a seed. Report 95% CIs |
| N5 | All DP configs at AUC ≈ 0.5, giving a null result on location | **ε sweep {1, 5, ∞}** (+8 to match Marek). The location question is then answered where leakage is measurable, and the Pareto frontier spans ε |
| N6 | Oracle (best-of-grid) attack hyperparameters | Pre-register γ and α; report the oracle value only as an upper bound |
| N7 | Freezing embeddings changes the "all" baseline definition | Redefine and document it. The additivity invariant I-M4 still holds with a new total |

---

## 6. Revised blueprint (for approval before implementation)

### 6.1 Data partition (seeded, after deduplicating SST-2 `train`)

| Split | Size | Role |
|---|---|---|
| `D_A`, `D_B` | 1024 each | Swap-pair training sets (members of A or B respectively) |
| `D_out` | 1024 | Audit non-members |
| `D_pop` | 1024 | RMIA population Z |
| `D_tune` | 2048 | Learning-rate pilot and early stopping (declared public) |
| `validation` | 872 | Utility only |
| Public pool (IMDB or Yelp) | about 20–50k sentences | DP-SelFT candidate pool |

### 6.2 Module changes (to be specified in ARCHITECTURE v2)

- `src/data.py` (new): deduplication, seeded partition, split manifests (index lists saved with each run).
- `src/models.py`: "all" freezes embeddings; validation of index lists; per-layer parameter manifest.
- `src/select_layers.py` (new): DP-SelFT Stages 2–3 (DP histogram, worst-case perturbation ranking) → `selection.json` (ranking, scores, ε_sel).
- `src/audit.py`:
  - Caches per-sample `loss` and `p_true` for (model, split).
  - Implements loss, Carlini ratio, RMIA-prob and RMIA-Alg1-loss attacks.
  - Outputs AUC, TPR at 1% and 5% FPR, and the generalization gap.
- `run_experiment.py`:
  - `--max_steps`; σ from `get_noise_multiplier`; `BatchMemoryManager`.
  - Trains the swap pair. Logs σ, ε_spent, ε_sel, clip fraction and gradient-norm statistics.

### 6.3 Grid

- **Phase A, location (k = 3):**
  - Configs {head, early, mid, late, dpselft_k3, all} × ε ∈ {∞, 8, 5, 1} × 5 seeds × swap pair.
  - The clean "all" baseline is run at the matched schedule and early-stopped.
- **Phase B, fraction:** late-anchored k ∈ {1, 2, 3, 6, 9, 12} and DP-SelFT top-k for the same k, at ε ∈ {5, ∞}.
- **Primary outcomes:** `val_accuracy`, `rmia_auc`, `rmia_tpr@5%`.
- **Secondary outcomes:** `loss_auc`, `ref_auc`, calibration gap, generalization gap.

### 6.4 Decisions I need from you

1. Approve the **shadow-model swap pairing** (and drop the pretrained-reference idea).
2. Approve the **public-pool DP-SelFT** (ε_sel = 0.3, matched total ε). The cheaper alternative is "public SST-2 split, ε_sel = 0", labelled as an ablation.
3. Approve **freezing embeddings in "all"**, or accept virtual batching plus ghost clipping, which is slower.
4. Approve **per-family pilot-tuned learning rates** on `D_tune`, with calibrated gates replacing the hard 75%.
5. Scope: should the ε sweep {1, 5, 8, ∞} and the 5 seeds be in v1, or should v1 use ε ∈ {5, ∞} with 3 seeds?
