# signatures.md
# The lens: each signature, what it measures, and what the literature says

This file is the project's master reference for every signature in the panel. For each measurement:
- Definition (formula)
- What it's sensitive to
- Data access tier (A: weights only / B: + small probe / C: + two distributions / D: + member/non-member / E: + second model)
- Literature it engages
- What we find vs what the literature predicts
- Where in the code

Cross-regime patterns of which signatures fire are the actual finding of the paper. No individual row here is novel; the combination and the decoupling pattern are.

---

## 1. Effective rank

**Definition.** For a weight matrix W, compute singular values σ_i, normalize p_i = σ_i² / Σ σ_j², then EffRank(W) = exp(−Σ p_i log p_i). Shannon entropy of squared singular value distribution.

**Sensitive to.** How concentrated the matrix's action is in a low-dimensional subspace. High effective rank means W uses many independent directions; low means W's action is dominated by a few.

**Tier.** A (weights only).

**Literature.**
- Yunis et al. 2024 (arXiv 2408.11804) — established the rank gap between memorizing and generalizing networks at the qualitative level across architectures.
- Galanti et al. 2022 — theoretical: SGD + WD induces an upper bound on rank that scales like 1/(λ × LR).
- Arora et al. 2018 — compression bounds via rank.

**Our finding.** Rank gap (M > G) holds in algorithmic (toy Transformer, 10-12× ratio) and CNN (25× in deep ResNet layers). **Fails in ViT head** — output projection compresses to ~10 rank regardless of training regime (architectural artifact: 10-class output bottleneck). Modest gap in ViT MLP layers.

**Code.** `effective_rank(W)` in `bulletproof3/_signatures.py`; per-layer via `all_ranks(model)`.

---

## 2. Top Hessian eigenvalue ("sharpness")

**Definition.** Largest eigenvalue λ_max(H) of the loss Hessian, computed via Lanczos iteration on Hessian-vector products.

**Sensitive to.** Local curvature of the loss landscape in the direction of steepest ascent.

**Tier.** B (weights + small probe set; needs to define a loss).

**Literature.**
- Keskar et al. 2017 (ICLR) — *sharp minima generalize worse than flat ones.* The dominant heuristic.
- **Dinh et al. 2017 (ICML) — "Sharp Minima Can Generalize For Deep Nets."** Theoretical critique: sharpness is not reparameterization-invariant. Any minimum can be made sharper or flatter by reparameterization without changing the function.
- Foret et al. 2021 (SAM) — sharpness-aware minimization; uses worst-case ε-ball loss.
- Andriushchenko & Flammarion 2022 — empirically, sharpness-generalization relationship is more nuanced.
- Petzka et al. 2021 — proposes relative flatness (next entry) as a reparameterization-invariant fix.
- Cohen et al. 2021 — Edge of Stability; sharpness self-adjusts to 2/LR.

**Our finding.** **Direction reverses across architectures, providing empirical confirmation of Dinh's theoretical critique in standard SGD training.**
- Algorithmic 4L Transformer: M top eig ~10⁴, G top eig ~10²-10³. M sharper. *Consistent with Keskar.*
- 1L Transformer (bp8): M top eig 200-300, G top eig 10⁻⁴-0.3. M sharper. *Consistent with Keskar.*
- ResNet-18 CIFAR-10 (tier2): M top eig 27-33, G top eig 91-115. **G sharper.** *Reverses Keskar; consistent with Dinh's critique.*
- ViT-Tiny CIFAR-10 (tier3b): M top eig 999-1905, G top eig 162-190. M sharper. *Consistent with Keskar.*
- ViT-Small CIFAR-100 (tier4): M top eig 120-189, G top eig 68-100. M sharper but only 2×. *Weak Keskar direction.*

Mechanism for the CNN reversal (mech4 ablation): WD constrains the CNN's spatial weight subspace, producing tighter local geometry even as it improves generalization. The reversal tracks WD alone, not augmentation.

**Position for paper.** This is one of the strongest empirical wedges we have. Frame as: "we provide empirical evidence in standard SGD training for Dinh's theoretical argument, which the prevailing Keskar/SAM heuristic has largely ignored."

**Code.** `lanczos_hessian(model, loss_fn)` and `hessian_top_bot(model, loss_fn)` in `_signatures.py`.

---

## 3. Bottom Hessian eigenvalue (saddle topology)

**Definition.** Most negative eigenvalue λ_min(H) of the loss Hessian. Negative ⇒ there is a direction of negative curvature ⇒ this is a saddle, not a minimum, on the full-data loss.

**Sensitive to.** Saddle structure. The geometric statement of "M is not a true minimum."

**Tier.** B.

**Literature.**
- Sagun et al. 2017 — empirical Hessian spectra; outlier eigenvalues + bulk near zero.
- Ghorbani et al. 2019 — Hessian eigenvalue density via stochastic Lanczos quadrature.
- arXiv 2602.18523 (geometry of multi-task grokking) — reports negative Hessian eigenvalues at BOTH M and G in their setup, with similar magnitudes.

**Our finding.** M has strictly negative bottom eigenvalues:
- 4L Transformer M (tier0): −1024 to −4542 across 5 seeds. **Strict saddle, large magnitude.**
- 1L Transformer M (bp8): −7.2 to −7.8 across 3 seeds (Lanczos with reorthogonalization caught what power iteration missed in bp1).
- ResNet-18 M (tier2): −0.15 to −0.43.
- ViT-Tiny M (tier3b): −343 to −491.
- G models: bottom eigenvalue near zero in algorithmic; in ResNet G is more negative than M (likely different geometry).

**Note on 2602.18523 disagreement.** That paper reported both M and G with similar negative eigenvalues in multi-task grokking. Our single-task results show much larger M vs G separation. Possibly because multi-task setups have more saddle structure inherent to the optimization. This needs honest discussion in the paper.

**Code.** Returned by `hessian_top_bot` as the second element of the tuple.

---

## 4. Relative flatness (Petzka)

**Definition.** Top Hessian eigenvalue × ||θ||². Scaling sharpness by the squared weight magnitude makes the measure reparameterization-invariant: rescaling weights by α scales the Hessian eigenvalue by 1/α² (locally), but multiplying by ||θ||² (which scales by α²) cancels the dependence.

**Sensitive to.** "Real" sharpness in a parameterization-independent sense.

**Tier.** B (needs Hessian eig + weight norm).

**Literature.**
- Petzka, Kamp, Adilova, Sminchisescu, Boley 2021 (NeurIPS), "Relative Flatness and Generalization."
- Directly responds to Dinh 2017's critique of raw sharpness.

**Our prediction.** If our tier2 sharpness REVERSAL (G sharper than M) is purely an artifact of Dinh's reparameterization issue, then Petzka's relative flatness should NOT reverse — both M and G should have similar relative flatness, OR M relative flatness > G as the "true" sharpness story predicts.

If Petzka also reverses, then even relative flatness isn't universal — supporting a broader claim that sharpness in any form is regime-conditional.

**Status.** Implemented as derived quantity in `compute_full_battery`. Will be in JSON for all future tier reruns once tier scripts are updated to capture init state.

**Code.** `relative_flatness(top_eig, weight_l2)` in `_signatures.py`; auto-included in battery output as `relative_flatness_full` and `relative_flatness_train`.

---

## 5. Gradient angle: cos(∇L_train(θ), ∇L_test(θ))

**Definition.** Cosine similarity between training-loss gradient and test-loss gradient at converged parameters.

**Sensitive to.** Whether the training signal and test signal "agree" geometrically at convergence. If they're anti-aligned, the model is at a point where decreasing train loss INCREASES test loss — the textbook signature of memorization.

**Tier.** C (weights + two distinguishable data subsets).

**Literature.** No direct prior work on this specific metric in the M-vs-G context.
- Sankararaman et al. 2020 (ICML) — gradient confusion: cosine between PER-EXAMPLE training gradients (different quantity).
- Chatterjee 2020 (ICLR) — coherent gradients within training set (different quantity).
- GrokAlign (arXiv 2506.12284) — Jacobian alignment in input-output space (different quantity).
- "Grokked Models are Better Unlearners" (arXiv 2512.03437) — gradient cosine between retain and forget sets (closest related, but different framing).

**Our finding.**
- 1L Transformer (bp9, 10 seeds each): M mean cos = −0.236 (9/10 negative). G mean cos = +0.105 (8/10 positive). Cohen's d > 1.9.
- 4L Transformer (tier0, 5 seeds): M mean cos = −0.207. G mean cos = +0.267 (one seed +0.876, others 0-0.26).
- ResNet-18 (tier2, 5 seeds): M cos = −0.07 to −0.20 (all negative). G cos = −0.04 to −0.22 (also negative but smaller magnitudes).
- ViT-Tiny (tier3b, 3 seeds): M cos = −0.06 to +0.05. G cos = −0.06 to +0.01. **Both near zero — signature washes out.**
- ViT-Small (tier4, 3 seeds): same washout pattern.

**Position for paper.** This is the closest thing we have to a genuinely novel measurement. Works clean in algorithmic and CNN, washes out in ViT. mech6 (forced grokking on tiny CIFAR) should test whether ViT washout is due to insufficient memorization pressure.

**Code.** `gradient_angle(model, train_loss_fn, test_loss_fn)` in `_signatures.py`.

---

## 6. Gradient norm ratio: ||∇L_test|| / ||∇L_train||

**Definition.** Ratio of the L2 norm of the test-loss gradient to the L2 norm of the train-loss gradient at converged parameters.

**Sensitive to.** Asymmetry in how strongly the train data and test data "pull" the parameters. At pure memorization, ||∇L_train|| ≈ 0 (perfect train fit) while ||∇L_test|| is large; ratio is huge.

**Tier.** C.

**Literature.** Connects to Entry 15 in our own log; not a standard measurement in the literature.

**Our finding.** Ratio is ~10¹⁰ at M (toy Transformer), ~3-5 at G. Ratio of ratios = 10⁹. Largest single-number gap in any signature.

**Position.** Supports the saddle topology claim — M is at a point where the train gradient is essentially zero but the test gradient is not. A geometric statement of "the model fits train perfectly but the test loss surface is not at a minimum here."

**Code.** Computed inside `gradient_angle` and returned as `grad_ratio_test_over_train`.

---

## 7. Weight L2 norm

**Definition.** ||θ||₂ = sqrt(Σ ||θ_i||²) over all parameters.

**Sensitive to.** Total magnitude of the network's weights. The simplest possible scalar summary.

**Tier.** A.

**Literature.**
- Liu et al. 2023 (Omnigrok, ICLR) — proposes the L_U mechanism: at grokking, the weight norm crosses a threshold where train loss stays flat but test loss drops. Weight norm is the central variable.
- Bartlett et al. 2017, Neyshabur et al. 2018 — norm-based generalization bounds.
- Jiang et al. 2020 — included as a baseline complexity measure.

**Our finding.** Universally, ||θ||_G < ||θ||_M for fair training regimes — WD shrinks weights, naturally. Magnitude varies hugely across architectures so direct comparison is meaningless without normalization.

**Code.** `weight_l2_norm(model)`.

---

## 8. Distance from initialization

**Definition.** ||θ_final − θ_init||₂ and relative variants (divided by ||θ_init||, ratio of norms).

**Sensitive to.** How far the optimizer traveled. Distinguishes lazy regime (small distance, NTK-like) from feature learning (large distance).

**Tier.** A but requires saving init weights at training start.

**Literature.**
- Jacot et al. 2018 (NTK) — distance from init characterizes lazy vs feature learning regime.
- Chizat & Bach 2019 — lazy vs feature learning distinction.

**Status.** Implemented; tier scripts need 2-line update to call `capture_init_state` before training and pass it to `compute_full_battery`.

**Code.** `distance_from_init(model, init_state_dict)` in `_signatures.py`. Returns dist, relative dist, init norm, final norm, ratio.

---

## 9. Path-norm proxy (sum of log spectral norms)

**Definition.** Σ_layer log(||W_layer||_op), where ||W||_op is the operator (spectral) norm of layer weight W.

This is the **log of the product of spectral norms** — the standard Lipschitz upper bound on the network's input-to-output Lipschitz constant. The "strict" path-norm sums over all input-to-output paths and is intractable for transformers, so this product-of-spectrals is the standard proxy in the norm-bound literature.

**Sensitive to.** Input-output sensitivity / Lipschitz character of the network.

**Tier.** A.

**Literature.**
- Bartlett et al. 2017 — spectrally normalized margin bounds.
- Neyshabur et al. 2018 — PAC-Bayes spectral.
- Jiang et al. 2020 — path-norm one of the TOP predictors of generalization gap in their 40+ measure benchmark.

**Our prediction.** Path-norm should be higher at M than G if it tracks generalization. If it doesn't reverse in our tiers, that's a confirmation of Jiang's finding. If it reverses, that's an extension.

**Code.** `path_norm_proxy(model)`. Returns `log_path_norm_proxy` (sum of logs to avoid overflow) and per-layer spectral norms.

---

## 10. MIA AUC (loss-based)

**Definition.** Per-example training-loss distribution vs per-example test-loss distribution. AUC of the binary classifier "score = −loss, member = 1 if in train."

**Sensitive to.** The statistical fact of memorization: whether the model's per-example loss is detectably lower on training examples than on held-out.

**Tier.** D (needs known member/non-member split).

**Literature.**
- Yeom et al. 2018 — theoretical link MIA advantage ≤ generalization gap.
- Shokri et al. 2017 — original MIA paper.
- Carlini et al. 2022 ("MIA from First Principles", LiRA) — gold-standard attack.

**Our finding.** M > G in 4/4 tested tiers (algorithmic, ResNet, ViT-T, ViT-S). Universal in direction.

**Important caveat.** MIA AUC is a **1-dimensional projection** — it can only distinguish "memorize-heavy" from "memorize-light." It cannot resolve:
- Pure overfit vs random-label memorization (both MIA ≈ 1.0)
- Grokked vs clean generalization (both MIA ≈ 0.55)

The four-regime classification requires the panel, not just MIA. mech5 (random-label CIFAR control) is the load-bearing test of this claim.

**Position for paper.** MIA is the universal AXIS (memorize ↔ generalize). The panel resolves WHICH KIND of memorization or generalization the model exhibits.

**Code.** `mia_loss_auc(train_losses_array, test_losses_array)` in `_signatures.py`.

---

## 11. Inter-layer singular vector alignment (Yunis)

**Definition.** For consecutive layers W_i and W_{i+1} with top singular vectors u_j (left) and v'_k (right), compute |⟨u_j, v'_k⟩|. Yunis uses the average of the top-10 diagonal entries of this overlap matrix as a scalar.

**Sensitive to.** Whether adjacent layers' learned subspaces align.

**Tier.** A.

**Literature.**
- Yunis et al. 2024 (arXiv 2408.11804, Eq. 3-4). Shows G has higher inter-layer alignment than M.

**Status.** Not yet implemented in `_signatures.py`. To engage Yunis directly we should add this as `inter_layer_alignment(model)`.

**Code.** TODO.

---

## 12. Mode-connectivity barrier

**Definition.** Linear interpolate θ_M and θ_G at α ∈ {0, 0.1, ..., 1}, evaluate loss at each. Barrier height = max loss along path − max loss at endpoints.

**Sensitive to.** Whether M and G are in the same loss basin (small barrier) or different basins (large barrier).

**Tier.** E (two models + eval data).

**Literature.**
- Frankle et al. 2020 (LMC and lottery ticket).
- Garipov et al. 2018 — curved low-loss paths.
- Draxler et al. 2018 — no barriers between SGD modes.
- Ainsworth et al. 2023 (git re-basin) — alignment via permutation.

**Our finding (Track A done; vision pending mech3).** Algorithmic Transformer M and G in clearly different basins (barrier ~10⁷). Vision/ViT pending.

**Code.** `mech3_mode_connectivity.py` in `bulletproof4/`.

---

## 13. Permutation-aligned LMC

**Definition.** Hungarian-match neurons of M to G first, then interpolate. Tests whether M and G are the "same solution up to permutation" (no barrier after alignment) or genuinely different (barrier persists).

**Tier.** E.

**Literature.**
- Ainsworth et al. 2023 (git re-basin).
- Entezari et al. 2022 — permutation invariance theorem.

**Our finding.** Track A: permutation alignment did NOT reduce the barrier (Entry 10 in so_far_results.md). M and G are genuinely different solutions, not permutations of each other.

**Code.** `mech7_permutation_lmc.py` (in flight).

---

## 14. Logit margin distribution

**Definition.** Per-example logit margin = logit_correct − max(logit_incorrect). Distribution shape over train and test sets.

**Sensitive to.** How confidently the model predicts on train vs test.

**Tier.** C (needs labeled probe data).

**Literature.**
- Bartlett et al. 2017 — margin bounds for generalization.

**Our finding (Track A).** M training margins are uniformly large (mean 24, tight std). M test margins are extremely negative (median −55, min −206). G margins are uniform on train and test.

**Code.** Computed in `bp7_structural_battery.py`; not yet in `_signatures.py`.

---

## 15. Fourier circuit presence (task-specific)

**Definition.** FFT of W_E (input embedding) along the input dimension. Concentration of energy in a small number of frequencies.

**Sensitive to.** Task-specific algorithmic structure.

**Tier.** A but task-specific (only applies to algorithmic tasks).

**Literature.**
- Nanda et al. 2023 (ICLR) — reverse-engineered the Fourier circuit for modular addition.

**Our finding.** G concentrates on 5 key frequencies (matches Nanda's reported {14, 35, 41, 42, 52}). M has flat spectrum.

**Code.** `taska/analysis/fourier.py`.

---

## What's still missing from the panel

To engage every relevant line of literature:

| Missing | Source | Priority |
|---|---|---|
| Inter-layer SV alignment (Yunis-style) | Yunis 2024 | High — direct competitor |
| Martin-Mahoney α exponent / weightwatcher | Martin-Mahoney 2021 | Medium — open-source library, runs on weights only |
| SAM-style sharpness (worst-case ε-ball loss) | Foret 2021 | Medium |
| LiRA MIA | Carlini 2022 | Medium — needs shadow models |
| PAC-Bayes flatness | Dziugaite & Roy 2017 | Low — bound is tight only sometimes |

---

## How to use this document

When writing a paper section about any signature, look it up here for:
1. The literature to cite
2. Our concrete numerical finding
3. What's surprising vs what's expected
4. Where the code lives

When implementing a new signature, add it here AT IMPLEMENTATION TIME with its literature anchor and our prediction. This prevents the "we measured X but forgot to position it" failure mode.
