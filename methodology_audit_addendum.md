# Methodology Audit Addendum (Part 2)

**Status:** analysis only. No repo files were modified for this addendum. The only commands run were read-only: measuring SST-2 statistics, printing the Opacus signature, computing σ with Opacus's accountant, and extracting text from the PDFs.

**Measured facts below** were computed on the real GLUE SST-2 `train` split in this repo's environment (Opacus 1.6.0). **Estimates** (VRAM, runtime) are labelled and must be measured by the implementation before being relied on.

## 0. Summary and corrections

| Item | Verdict |
|---|---|
| 1. Dedup and stratification | **Accept, but your two examples are empirically null, and the real leakage is elsewhere.** Case and whitespace variants do not exist in GLUE SST-2. Punctuation normalization finds 8,734 extra duplicates and 152 label-conflict groups. The larger problem is *parent-phrase families*, which need group-aware splitting. |
| 2. Fix α = 0, γ = 1 | **Partially reject.** γ is provably irrelevant to AUC, so γ = 1 is harmless. α is not negligible for classification. With α = 0, Marek's loss-form Algorithm 1 degenerates toward the naive loss attack on well-fit classifiers. α needs a calibration step. |
| 3. Global clipping confound | **Inherent for "fraction", a mediator for "location".** Measure it with telemetry and a 2×2 factorial of diagnostic arms. |
| 4. Memory and disk | **Lifecycle accepted, with changes.** Run each model in its own subprocess, and cache logits (about 40 KB per model, not 8 KB). The 16 GB VRAM bound is plausible but must be measured, not asserted. |
| 5. DP-SelFT outcomes | **Accept, with a larger outcome space.** There are four outcomes, not two. Two controls are mandatory: the same layers at the *selection-matched* budget, and random triples. Equivalence margins must be pre-registered. |

### Corrections to my previous audit (`methodology_audit.md`)

- **C1. "Aggregate SNR is roughly independent of T" is only asymptotic.** I computed σ with Opacus RDP at ε = 5, δ = 1e-5, N = 1024:

  | Schedule | σ | σ/B (per-coordinate noise on the averaged update) |
  |---|---|---|
  | T = 160, B = 32 | 0.817 | 0.0255 |
  | T = 1000, B = 64 | 2.056 | 0.0321 |

  The √T-scaling argument assumed σ ≫ 1. Here, 6.25× more steps costs only 1.26× more per-coordinate noise. Under the same linear-signal approximation (total signal ∝ T, total noise ∝ √T·σ/B), aggregate SNR scales as √T·B/σ. That gives ≈ 495 vs ≈ 984, so **1000 steps roughly doubles aggregate SNR**. The recommendation (1000 steps, B = 64) stands, and now has a stronger justification. At ε = 1, σ = 8.1, so the asymptotic regime does apply there.
- **C2. "Report both the Carlini ratio and RMIA" is redundant with a shared reference and population.** §2.1 proves that their AUCs coincide (up to ties and α). I'll restructure the attack suite in §2.5.
- **C3. Per-run vector size.** `audit_preds.npy` at about 8 KB is too small. The right figure is about 40 KB per model (§4).

---

## 1. Item 1: Normalized deduplication and stratification

### 1.1 What the data actually look like (measured)

| Normalization | Unique texts | Duplicates removed | Label-conflict groups |
|---|---|---|---|
| none (exact) | 66,978 of 67,349 | 371 | 5 |
| `strip().lower()` | 66,978 | 371 (**same**) | 5 |
| NFKC + lower + strip punctuation + collapse spaces | 58,615 | 8,734 | **152** (309 rows) |

- **Case and whitespace variants do not exist in GLUE SST-2.** The text is already lowercased and tokenized, so `strip().lower()` finds nothing new. The scenario `"great movie."` vs `"Great movie "` is real in general but null for this dataset.
- **Punctuation stripping is where the action is.** It removes 13% more items. Typical near-duplicates are the same fragment with and without a trailing `,`, `.` or `--`.
- **Label-conflict groups are real.** Examples from the data:
  - `'...human nature is a goofball movie , in the way that malkovich was , but it tries too hard '` → 0, and the same text with a trailing `. ` → 1.
  - `'laughably '` → 0 and `'laughably '` → 1 (an exact-string conflict).

### 1.2 Does normalization distort semantics? Verdict: no, if conflicts are handled conservatively

- **Mechanism of the conflicts:** SST phrase labels come from thresholding a continuous sentiment score from tree annotation. Near-neutral fragments sit at the boundary, and trailing punctuation changes which node is labelled. This is **label noise intrinsic to SST-2**, not an artifact of our normalization.
- **Risks of aggressive normalization and mitigations:**
  - *Punctuation carrying sentiment* (`good !` vs `good ?`): any resulting label disagreement lands in a conflict group and is dropped.
  - *Apostrophes:* SST-2 is pre-tokenized (`does n't`, `'s`). Stripping `'` is applied identically to every string, so no false merges arise.
  - *Empty keys* (pure-punctuation items, e.g. emoticons): drop.
- **Policy: drop the entire conflict group, never keep-first.** Keeping one label makes the retained label wrong about half the time. As a non-member it would also be a near-copy of a labelled training text with the opposite label, which is a nonsensical audit item. Cost: 309 rows, 0.5%.
- After normalization, dedup and conflict-drop: **58,463 items**, positive fraction **0.558**.

### 1.3 The bigger leak is containment ("families"), not case or punctuation

SST-2 `train` includes the full sentences *and* their sub-phrases. Measured (word-boundary, ≥ 4-word phrases, after dedup):
- **83.4% (33,487 of 40,168)** of items with ≥ 4 words are contained in a longer train item.
- A first rough probe (character-level, on an order-biased sample) gave about 94%. Treat that as an **overestimate**; the word-boundary figure is the reliable one.
- A phrase in `D_out` whose parent sentence is in `D_A` gets a low loss on a "non-member". This is exactly the suppression you describe, and it is **larger than the case and whitespace effect**.

**Family structure.** Linking each item to its longest containing item (single-parent forest, min 4 words):
- 24,976 components, largest = 25, mean 2.34.
- Under the alternative of linking on any containment, the largest component is 122.
- **There is no giant component**, so group-aware splitting is feasible: about 25k families against about 6.1k items needed.
- 18,295 items (31%) have < 4 words and cannot be linked. They are generic fragments (`laughably`) that can't be attributed to a parent.

**DP-semantics point.** Our guarantee is record-level (ε, δ)-DP. If a parent sentence and its sub-phrases are all in `D_A`, then removing *one* record does not remove the sentence from the model's information, and the group-privacy factor applies. **Duplicates and family members inside a member set weaken the effective guarantee for the text** and violate the independence assumption behind the amplification-by-subsampling analysis in spirit.

### 1.4 Deterministic stratified, group-aware partition (specification)

Inputs: `seed`, targets `n_A = n_B = n_out = n_pop = 1024`, `n_tune = 2048`, and `MIN = 4`.

1. **Normalize** `key(x) = collapse_ws(strip_punct(lower(NFKC(x))))`. Drop empty keys.
2. **Conflict-drop:** group by `key`. If a group has more than one label, drop the whole group. Otherwise keep one representative, the lexicographically smallest raw string.
3. **Families:** for each item with ≥ `MIN` words, link it to its longest containing item (word-boundary match, ties broken by lexicographic order). The family ID is the tree root. Items with < `MIN` words are singleton families.
4. **Member-set rule, one record per family:** each of `D_A, D_B, D_out, D_pop, D_tune` takes at most one item per family, drawn uniformly (seeded) from the family. **This is the recommended policy (F1).** The alternative (F0) keeps whole families and leaves intra-split correlation.
5. **Bucket assignment by hash:** `bucket(family) = argmax over splits of SHA256(seed ‖ split ‖ family_id)`, weighted so bucket sizes are ≈ 1.3× the targets, plus an "unused" bucket. This is order-independent, stable under resume, and needs no RNG state.
6. **Exact class balance by trimming:** inside each bucket, order items by `SHA256(seed ‖ item_key)` and take the first `n_s/2` positives and `n_s/2` negatives. Dropping surplus items cannot leak.
7. **Assertions (run on every partition):**
   - Splits are pairwise disjoint on `key`.
   - Splits are pairwise disjoint on family ID.
   - Each split has exactly 50/50 labels and exactly the target size.
   - A standardized mean difference on word count between members (`D_A`) and non-members (`D_out`) is below 0.1. A KS test p-value > 0.01 is a second check.
   - Report the 5-gram overlap rate between splits. It will not be 0, because generic n-grams (`in the way that`) recur. Residual overlap is reported, not eliminated.
8. The split manifest (index lists + hashes) is saved with every run.

**Why exact 50/50 matters beyond the prior-fit concern.** The positive fraction is 0.558. If members and non-members have different class mixes, a loss attack gets AUC ≠ 0.5 from class-conditional difficulty alone. Identical class composition across `D_A`/`D_out` removes that confound. **Recommended extension:** stratify on class × word-count tercile (6 cells), so length is balanced too.

**Deviation from Paper 2.** DP-SelFT uses a random 1024-record SST-2 subsample (native 55.8% positive). Our balanced, one-per-family sample is a controlled variant. Record it as a documented deviation in ARCHITECTURE.

---

## 2. Item 2: RMIA hyperparameters (Marek App. H)

### 2.0 What App. H actually says

> "We focus on the importance of γ, as α has a much more limited effect, and we set it to 0. Figure 14 shows the importance [of] γ and suggests that γ = 1 is often the best choice."

- The evidence is **Figure 14**: Pythia 1B on **SAMSum only**, ε ∈ {8, ∞}, with a log-scale γ axis that appears to run from 10⁰ to 10¹ (from the extracted text; verify visually).
- "Proved" and "universally robust" overstate it. The paper says "often the best" and "a strong baseline", and gives no α ablation figure.
- If the γ axis really starts at 1, then γ = 1 sits at the **boundary** of the explored range, which is weaker evidence than an interior optimum.

### 2.1 Theorem: AUC does not depend on γ or on the population Z

For offline RMIA with a single reference and a shared population Z of size m (Alg. 1, ratio form):

```
Score_γ(x) = (1/m) · #{ z ∈ Z : Ratio_x < γ · Ratio_z } = 1 − F_Z( Ratio_x / γ )
```

where `F_Z(t) = #{z : Ratio_z ≤ t}/m` is the empirical CDF of the population ratios.

- `Ratio_x` depends only on `x` (target loss and reference loss on `x`). `Ratio_z` depends only on `z`.
- `F_Z` is non-decreasing, so `Score_γ(x)` is a **non-increasing step function of `Ratio_x` alone**.
- AUC is invariant to strictly monotone transformations of the score. A step function is monotone *except* that it creates **ties**.

**Consequences:**
1. The ROC curve of offline RMIA equals the ROC curve of the statistic `Ratio_x`, up to tie-coarsening. This holds for AUC **and** for TPR at any fixed FPR.
2. **Neither γ nor the choice of Z changes the ranking.** γ only shifts the point where `F_Z` saturates. With large γ, most `x` map to the same score and AUC falls toward 0.5.
3. The same argument holds for the difference form in Eq. 6, where the score is `F_Z(L(x) − γ)`, a monotone function of the **target** loss alone.
4. So the AUC-versus-γ curve in Fig. 14 reflects tie and saturation behaviour of a finite Z, not a property of the data distribution. This is my derivation, not a claim from the paper. It is checkable: a unit test should show AUC(RMIA, γ = 1, m = 1024) agrees with AUC(Ratio_x) to within tie resolution.
5. **Practical rule:** compute the continuous statistic `Ratio_x` (so there are no ties) as the primary score. Report the quantile form (γ = 1) for Marek-fidelity. The population Z is then only needed for TPR at a *fixed score threshold*, which we do not use.

Corollary on the earlier "RMIA (shadow) 0.73 vs Reference (pretrained) 0.51" comparison in Marek Table 2: the two rows differ in **which reference model is used**, not in the attack form. With the same reference, RMIA and the Carlini ratio produce the same ranking (up to α). The advantage in their tables comes from the shadow reference.

### 2.2 α is not negligible for classification

Alg. 1 line 4: `L̃(x) = ½[(1+α)·L_OUT(x) + (1−α)]`. At α = 0, `L̃ = ½(L_OUT + 1)`. The calibration strength is the elasticity of `L̃` in `L_OUT`: `d ln L̃ / d ln L_OUT = (1+α)L_OUT / ((1+α)L_OUT + (1−α))`.

| Form | α | Elasticity |
|---|---|---|
| Loss form, LM per-token loss (L ≈ 2–3 nats) | 0 | L/(L+1) ≈ 0.7–0.75 |
| Loss form, well-fit classifier (L ≈ 0.01–0.1) | 0 | **≈ 0.01–0.09**, so `L̃ ≈ ½` (nearly constant) |
| Probability form `p_OUT ∈ [0.5, 1]` | 0 | p/(p+1) ∈ [0.33, 0.5] |
| Probability form | 1 | 1 (full ratio) |

- For an LM, α = 0 keeps most of the calibration. For a classifier trained to low loss, the loss-form denominator is nearly a constant, so **the loss-form attack degenerates to a (reference-free) naive loss attack**, which is exactly the failure RMIA is meant to fix.
- So Marek's "α has a limited effect" is evidence at LM loss scales only. It **does not transfer** to our p(y|x) or loss values in [0, ~0.7].
- Numerical risk: at ε = ∞ with 62 epochs on 1024 samples, `p(y|x)` saturates near 1 in float32, producing ties. Use `log p` computed from logits in float64. LiRA's scaled logit `log(p/(1−p))` is the stable form.

### 2.3 Verdicts

1. **γ = 1:** hard-code it. It is harmless for AUC (§2.1) and matches Marek.
2. **α = 0 as a fixed constant: do not hard-code.** α remains a genuine hyperparameter in our setting.
3. **Calibration without tuning on evaluated targets:**
   - Designate **seed 0 as a calibration seed, excluded from all reported statistics.** The partition depends on the seed, so its data are disjoint from reported seeds.
   - Choose α per ε-level from {0, 0.25, 0.5, 0.75, 1}, maximizing the AUC *averaged across all configs* at seed 0 (pooled, so no per-config tuning).
   - Freeze α before running reported seeds. Report α = 0 (Marek-fidelity) alongside as a sensitivity result.
4. **Does this remove the need for a tuning stage?** For γ, yes. For α, no: it needs the seed-0 calibration pass. `D_tune` is still needed for learning-rate selection.
5. **Attack sanity gates (new):** on clean ε = ∞ models, calibrated-ratio AUC ≥ naive-loss AUC − 0.01, and > 0.6. A calibrated attack that is *worse* than naive loss signals a bug or a bad reference.

### 2.4 Resolution of the population

`|Z| = 1024` gives a quantile resolution of about 0.001. That is fine because the continuous score is primary. Z can be fixed across all targets (`D_pop`), which is what the Alg. 1 line 1 per-x resampling would approximate with extra noise.

### 2.5 Revised attack suite

| Attack | Role |
|---|---|
| Loss threshold | Reference-free baseline |
| Calibrated ratio with shadow reference (continuous `Ratio_x`, α from §2.3, probability or logit form) | **Primary.** Equals offline RMIA in ranking. |
| Offline RMIA quantile form (γ = 1, α as above and α = 0) | Fidelity to Marek. Unit-tested against the continuous score. |
| Alg. 1 loss form (α = 0, floor 1e-6) | Marek's exact variant, labelled "may degenerate on well-fit classifiers". |
| Calibration gap = AUC(primary) − AUC(loss) | Measures how much the naive attack under-reports leakage |

---

## 3. Item 3: Global clipping vs. the parameter-fraction confound

Opacus 1.6.0's default is `clipping="flat"` (global ℓ2 norm over all trainable parameters), confirmed from `make_private`'s signature. `max_grad_norm` can be a list only with per-layer clipping.

### 3.1 Is it a confound or inherent?

**It is inherent to flat DP-SGD, and its role depends on the estimand:**
- **Fraction effect (changing d_Λ).** Clipping dilution is part of the *mechanism* by which more trainable parameters hurt: a bigger Λ means a larger `‖g_i‖`, a smaller clip factor `s_i = min(1, C/‖g_i‖)`, and a smaller signal per coordinate. It is part of the treatment effect, not a nuisance. It *must be decomposed* when explaining results.
- **Location effect (equal d_Λ).** Per-sample gradient norms differ by depth, so `s_i` differs by location. Clipping is a **mediator** of the total location effect. We report the total effect (what a practitioner sees) and decompose it.

### 3.2 First-principles quantification

Let `‖g_i‖ ≫ C` for most samples (to be verified by telemetry; full-model per-sample gradient norms for transformers are typically ≫ 1, so a clip fraction near 100% is expected). Then the clipped per-sample gradient has norm exactly `C`.

- Signal norm per step ≈ `C`, **independent of d_Λ**.
- Noise norm per step ≈ `σC·√d_Λ / B`.
- Using σ = 2.056 and B = 64 (σ/B = 0.0321, C = 1):

  | Config | d_Λ | Noise norm | Noise/signal per step |
  |---|---|---|---|
  | head | 592,130 | 24.7 | ≈ 25 |
  | 3 blocks + head | 21,856,766 | 150 | ≈ 150 |
  | all blocks + head (embeddings frozen) | 85,646,594 | 297 | ≈ 297 |

  The ratio head : k3 : all ≈ 1 : 6 : 12, which is √d_Λ, exactly Theorem 1's scaling. So when every sample is clipped, **"gradient starvation" and "noise damage" are the same normalization seen from two sides**, and they only separate when clipping is *inactive*.
- **Under AdamW**, the update is `m̂/√v̂`. In the noise-dominated regime `√v̂ ≈ σC/B` per coordinate, which is unchanged by Λ. The signal per coordinate scales with the clip factor, which is smaller for larger Λ. Adam does not cancel the effect, because the noise term is not rescaled by `s_i`.
- **The absolute value of C is nearly irrelevant when clipping is saturated and the optimizer is Adam**, since signal and noise both scale with C and Adam normalizes the overall scale. This explains why Marek (C = 0.1) and DP-SelFT (C = 1) can both work. It breaks down when clipping is not saturated, so telemetry should confirm saturation.

### 3.3 Isolating starvation from noise damage: a 2×2 factorial of diagnostic arms

| | Noise off | Noise on (σC absolute) |
|---|---|---|
| **Clipping off** | clean (already in the grid) | **N-arm:** `max_grad_norm = C_big`, `noise_multiplier = σ·C/C_big`. This keeps the absolute noise std at σC but removes clipping. |
| **Clipping on** | **C-arm:** `noise_multiplier = 0` (clip only) | DP (already in the grid) |

- Effects: clipping/starvation effect ≈ (C-arm − clean); noise effect ≈ (N-arm − clean); **interaction** = DP − C − N + clean.
- Run at ε = 5 only, for Λ ∈ {head, late-k3, all}, 3 seeds, so 2 extra arms × 3 Λ × 3 seeds × 2 (swap pair) = **36 extra models**. The estimate is a few GPU-hours.
- The N-arm and C-arm have **no formal DP guarantee**. Label them "diagnostic" in every table.
- Also verify that Opacus accepts `noise_multiplier = 0` and very large `max_grad_norm`.
- **Optional treatment variant:** per-layer or group-wise clipping (`clipping="per_layer"` with a list) equalizes the budget across groups. It is a different mechanism, not a control, so label it as such and keep it out of the main claims.

### 3.4 Telemetry (aggregated over 50-step windows; a few KB per run)

Per-step quantities come from the Opacus hooks (`p.grad_sample` before clipping, `p.summed_grad` before noise):
1. Total per-sample gradient norm: p10, p50, p90.
2. **Clip fraction** and the mean clip factor.
3. **Group energy shares:** mean over samples of `‖g_i^{(group)}‖² / ‖g_i‖²` for the head and for each trainable block (the empirical signal-retention ratio s(Λ) inside the trainable set).
4. **Realized noise/signal:** `‖ξ‖ / ‖Σ clipped g_i‖` per step.
5. Training loss.

Plus **at initialization** (non-private diagnostic on `D_tune`): the clean per-layer gradient energy share for all 12 blocks and the head. This is the empirical test of H2's premise (`ρ_late ≥ ρ_mid ≥ ρ_early`) and the direct measurement of Sha et al.'s s(Λ) (Eq. 43).

**Analysis plan:** regress leakage and utility on d_Λ, clip fraction, realized noise/signal, and the generalization gap (`mean OUT loss − mean IN loss`), plus the factorial cells above. Report the interaction term.

**Caveat:** ghost clipping materializes only norms, not per-sample vectors, so the group-share telemetry would be unavailable under it. Use `grad_sample_mode="hooks"` with physical batch ≤ 16.

---

## 4. Item 4: Memory lifecycle and Kaggle disk discipline

### 4.1 Lifecycle for a swap pair

Yes, sequential training with caching is correct, and post-hoc bidirectional scoring is valid. Changes I recommend:

1. **Run each model in its own subprocess**, not `del model; empty_cache()` inside one process. With Opacus, `GradSampleModule` hooks, the `DPOptimizer` and the `PrivacyEngine` all hold references to parameters and per-sample buffers. `del model` alone does not free them, and `empty_cache()` does not release memory that is still referenced. Process exit is the only hard guarantee. It also gives crash isolation.
2. **Per model, cache one array per split, shape `(n, 2)`, logits as float32:** `D_A, D_B, D_out, D_pop, validation`. That is 4,968 samples × 8 B ≈ **40 KB per model**. Loss, probability, margin and every attack input derive from logits.
3. **Post-process on CPU:** RMIA needs target vectors from `M_A` and reference vectors from `M_B` on the same samples. With both caches on disk, the bidirectional scoring (`M_A` as target and `M_B` as reference, and vice versa) is a pure NumPy computation. The "no forward passes in post-processing" claim holds.
4. **Write atomically** (write to a temp file, then rename) and key every file by `(config, ε, seed, role, partition-hash)`. A **manifest** of completed files makes the grid resumable after a Kaggle session cutoff.
5. **No `.pt` checkpoints.**

### 4.2 Confirming the bounds

- **Disk:** 40 KB per model plus a few KB of telemetry. 72 runs (144 models, or 72 if a "run" is one model) is under 10 MB. Even a 210-pair grid is about 17 MB. **Under 50 MB: confirmed.**
- Other disk users: the Hugging Face cache (roberta-base about 500 MB, datasets) lives outside `/kaggle/working`. Check where it is written on Kaggle.
- **VRAM: plausible, not confirmed.** Estimates for fp32 per-sample gradients, embeddings frozen:

  | Config | Per-sample grad memory | Notes |
  |---|---|---|
  | all blocks (85.6M trainable) | 342 MB per sample, so about 5.5 GB at physical batch 16 | Plus parameters (0.5 GB), AdamW state (about 0.7 GB), activations (a few GB). **Estimated total ≈ 9–11 GB.** |
  | all blocks, physical batch 32 | about 11 GB for gradients alone | **Likely to exceed T4 capacity.** Do not use. |
  | k3 / head | small | Comfortable |

- The estimate is mine, not a measurement. Require `torch.cuda.max_memory_allocated()` logging per run and an automatic fallback that halves the physical batch size on OOM. Logical batch 64 via `BatchMemoryManager` is unaffected.
- Host RAM and the session wall-clock limit are further constraints. Check current Kaggle limits; I have not verified them.

---

## 5. Item 5: DP-SelFT selection and the pre-registered hypotheses

### 5.1 Mechanics of the ranking

For a singleton layer ℓ, the worst-case perturbation objective (Sha et al., Eq. 20) to second order gives a robustness penalty of roughly

```
ηρ · ‖∇_ℓ L(θ′)‖  +  (η²ρ²/2) · λ_max(H_ℓℓ)
```

with `θ′ = θ − ηg_ℓ`. With equal d_ℓ, ρ is the same for every layer, so the ranking depends on (a) the descent achievable, which scales with `‖g_ℓ‖²`, and (b) the gradient norm and top Hessian eigenvalue at the post-update point. **Prefer layers with large gradients and low curvature.** This part is my derivation, not from the paper.

Why the outcome is not obvious:
- **Singleton ranking ignores interactions.** The top-3 singletons may be redundant. In a residual network, adjacent layers can carry overlapping signal, and the best triple is not the three best singles.
- **Selection noise.** The synthetic validation set is about 30% of 1024, i.e. about 307 examples, so accuracy has SE ≈ 2.5 points. Layers within a few points of each other cannot be reliably ordered. A "divergent" selection (e.g. `2, 7, 11`) can simply be noise.
- **Regime.** DP-SelFT's validated setting is k = 15 of 24 layers (62.5%) for RoBERTa-Large, and they report extremely low accuracy for k < 9. The analogue for 12 layers is k ≈ 7–8. Our k = 3 is outside it.
- **The head is always trained in our setup.** Sha et al. say they unfreeze only the layer being evaluated. We assume the head stays trainable (a randomly initialized head cannot be left frozen). Document this as an assumption.
- **Budget accounting.** DP-SelFT trains at ε_ft = ε − ε_sel = 4.7. Even if it selects exactly `9, 10, 11`, its final model has higher σ than the heuristic arm at 5.0. So "Outcome A" does not make the two arms identical.

### 5.2 Four outcomes, not two

Define the selector output per seed as the top-3 set. Over 5 seeds:

| Outcome | Rule |
|---|---|
| **A: Convergence** | `{9,10,11}` is the modal set and appears in ≥ 3 of 5 seeds |
| **P: Partial** | modal set shares exactly 2 of 3 layers with `{9,10,11}` in ≥ 3 of 5 seeds |
| **B: Divergence** | ≥ 3 of 5 seeds select sets sharing ≤ 1 layer with `{9,10,11}` |
| **U: Unstable** | none of the above (modal frequency < 3/5), mean pairwise Jaccard < 0.5 |

Outcome U (selection noise) is the one the two-outcome framing omits, and it is quite plausible at this validation size. Report the full rank distributions and Kendall τ across seeds either way.

### 5.3 Arms and estimands

All at total ε = 5 (RMIA AUC is the primary leakage metric):

| Arm | Layers | ε_ft |
|---|---|---|
| **H** | heuristic late `{9,10,11}` | 5.0 |
| **H′** | `{9,10,11}` (selection-matched control) | 4.7 |
| **S** | DP-SelFT output for that seed | 4.7 |
| **R** | 5 random triples (control for "any 3 layers") | 4.7 |

- **S vs H′** isolates the effect of *which layers*, at identical σ.
- **S vs H** is the end-to-end procedure effect, including the cost of selection.
- **Under A, S vs H′ is a negative control.** The layer sets are identical, so the difference measures seed-level pairing noise. That gives a **noise floor** for all paired comparisons.
- Paired differences: `Δu = Acc(S) − Acc(H′)` and `ΔL = RMIA-AUC(S) − RMIA-AUC(H′)`. Pair by seed and partition. Use a paired bootstrap, clustering over seeds.
- **Equivalence margins (pre-registered):** δ_u = 1.0 accuracy point and δ_L = 0.02 AUC (about 1.5× the single-run SE of ≈ 0.013 at n = 1024).
- **Power gate:** estimate the SD of paired differences from repeated H′ runs on calibration seeds. If the minimum detectable effect exceeds δ_L, label the result **"inconclusive"**, not "equivalent".

### 5.4 Pre-registered hypotheses (text to add to ARCHITECTURE.md)

> **H4a (flatness).** Layers selected for robustness to worst-case perturbation lie in flatter regions and generalize better. Prediction: `ΔL < −δ_L` with `Δu ≥ −δ_u`.
>
> **H4b (signal retention).** Layers with high signal-retention ratio s(Λ) fit individual samples faster. Prediction: `ΔL > +δ_L`.
>
> **H4null.** Selection neither mitigates nor amplifies: `|ΔL| < δ_L` and `|Δu| < δ_u` (TOST), given adequate power.
>
> These predictions are stated before any selected-layer run. The prior (not a finding): late, contiguous layers are mildly favored, with low confidence.

**Decision table (every cell has a pre-stated reading):**

| | `Δu ≥ −δ_u` (utility kept) | `Δu < −δ_u` (utility lost) |
|---|---|---|
| `ΔL < −δ_L` (leaks less) | **Mitigates**, Pareto-improving | Mitigates by trading utility |
| `|ΔL| ≤ δ_L` | **No detectable effect** (if powered) | Dominated by H′ |
| `ΔL > +δ_L` (leaks more) | **Amplifies** | Dominated and leakier |

**Conditional follow-ups:**
- Under **B**, add a control with a contiguous block of 3 layers centered on the median selected layer, to separate "depth" from "non-contiguity".
- Under **U**, report the selector as noise-dominated and use R (random triples) as the comparison.

**Mediators:** test whether `ΔL` tracks the generalization gap difference (`mean OUT loss − mean IN loss`) and the telemetry quantities of §3.4.

**Limit on what the audit can say.** Selection consumes privacy budget ε_sel = 0.3 through the DP histogram, but our attack audits only the final model. The influence of selection on leakage is bounded by ε_sel and cannot be separately measured empirically.

---

## 6. Decisions needed

The five decisions from the earlier audit are still open. Beyond those, this addendum needs:

1. **Member policy:** one record per family (F1, recommended) or whole families (F0)?
2. **Length handling:** stratify on class × length tercile, or class only?
3. **α calibration:** approve seed 0 as an excluded calibration seed with α ∈ {0, 0.25, 0.5, 0.75, 1}?
4. **Factorial diagnostic arms:** approve the 36 extra diagnostic models (non-DP, labelled as such)?
5. **Selection-matched control H′ and random triples R:** approve both?
6. **Equivalence margins:** accept δ_u = 1.0 point and δ_L = 0.02 AUC?
7. **Per-model subprocess execution:** approve, with the auto-halving physical-batch fallback?
