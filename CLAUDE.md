# What Does an Overfit Network Know?
## A Mechanistic Characterization of Overfitting

**Target venue:** TMLR (Transactions on Machine Learning Research)
**Timeline:** 30 days
**Status:** Research plan / pre-registration draft v1

---

## 0. TL;DR

We claim that an overfit fine-tuned model is not merely a "worse" version of a generalizing model — it is a **qualitatively different object** with identifiable structural signatures. Specifically, we hypothesize that overfitting produces:

1. **A spectral signature in weight space:** intruder singular directions, increased effective rank, and departures from heavy-tailed power-law behavior that are absent in generalizing models.
2. **A causally-active "memorization subspace":** a low-dimensional set of singular directions / neurons that, when ablated, *selectively* destroys training-set fit while preserving (or improving) test behavior — restoring the generalizing model without retraining.
3. **Per-example structure in activations:** the overfit model's residual stream encodes training-example identity to a degree that linear probes can extract, while the generalizing model's representations have averaged this out.
4. **A sharper loss landscape with anisotropic curvature** that correlates causally (not just correlationally) with the spectral signature above.

The four claims are linked: we conjecture that the spectral signature (1) *is* the mechanism by which the memorization subspace (2) is implemented, which produces the activation structure (3) and the curvature anisotropy (4). The work's TMLR-level contribution is to test this linkage with controlled experiments, causal interventions, and falsification tests rather than merely showing correlations.

This is the kind of claim TMLR was built for: technically rigorous, not chasing a SOTA number, and a contribution that the community can build on regardless of whether the dominant interpretation in the field shifts.

---

## 1. The thesis, sharpened

The original idea was: *what does an overfit model know that a generalizing model doesn't?* The literature gives us a sharper formulation.

The conventional view is bimodal:

- **Classical statistical learning** says overfitting = high variance = bad estimator, throw it away.
- **Modern deep learning** says overfitting on training data is fine — benign overfitting, double descent, interpolation regimes generalize.

Both views treat the overfit model as a *quantitative* point on a generalization-error axis. We claim they are wrong about the relationship between overfit and generalizing models being purely quantitative. We claim there is a **qualitative**, mechanistically identifiable difference in the *structure* of what each model has computed, and that this structure can be read off the weights and activations using existing interpretability tools, then causally manipulated.

The closest existing framings:

- **Grokking** (Power et al., 2022; Nanda et al., 2023): training has three phases — memorization, circuit formation, cleanup. An overfit network is one where cleanup hasn't happened (or has been actively prevented).
- **Counterfactual memorization** (Feldman, 2019; Feldman & Zhang, 2020; Zhang et al., 2023): some training examples are memorized in the sense that omitting them from training changes the model's behavior on them specifically. The overfit model amplifies this — every training example becomes counterfactually memorized.
- **Intruder dimensions** (Shuttleworth et al., 2024): fine-tuning methods produce structurally different weight updates visible in the SVD. We extend this lens to the overfit-vs-generalizing axis specifically.
- **Spectral dynamics** (Yunis et al., 2024): generalizing training produces low-rank weight matrices; random-label training produces high-rank. We test whether overfitting on real labels sits on the same axis or produces its own signature.

We are taking the cleanup hypothesis seriously and asking: what is the structure of an *uncleaned* model? The answer matters because if it is just "the same circuits but with extra noise", overfitting is degradation. If instead it is "additional circuits that store training-example-specific information in identifiable substrates", then overfitting is a different *kind* of representation — and that has consequences for unlearning, privacy, model auditing, and fine-tuning practice.

---

## 2. Literature synthesis: the four lenses

We organize the relevant prior work into four lenses on the overfit-vs-generalize axis. Each lens gives a different observable, and our novelty is testing whether **all four lenses agree** on the same underlying mechanism and whether causally intervening on one moves the others.

### 2.1 Weight-space lens (spectral)

This is where the most actionable evidence lives.

- **Martin & Mahoney's Heavy-Tailed Self-Regularization theory** (Martin & Mahoney, 2018, 2019, 2021) is the foundational work. They show that the Empirical Spectral Density (ESD) of weight matrices in well-trained networks evolves through a "5+1 phase" sequence — Random-like → Bleeding-out → Bulk+Spikes → Bulk-Decay → Heavy-Tailed → Rank-Collapse. The Heavy-Tailed (HT) phase, fit by a power law with exponent α ∈ [2, 4], correlates strongly with good generalization. Their "Weighted Alpha" metric predicts test accuracy of pre-trained models without ever touching test data.

  **Our prediction:** Overfit models will either (a) fail to reach the HT phase, sitting in Bulk+Spikes or Bulk-Decay; or (b) overshoot into Rank-Collapse / pathological heavy tails (α < 2). Either way, this is a *qualitative* signature distinguishable from generalizing models, not a sliding scale.

- **Approaching Deep Learning through the Spectral Dynamics of Weights** (Yunis et al., 2024) shows directly that random-label training produces high-rank weights, true-label training produces low-rank weights, *and* alignment between consecutive layers' singular vectors differs. Their "alignment" metric is novel and underexploited.

  **Our prediction:** Overfit-on-real-labels sits between random-label and generalizing-real-label on rank metrics, but qualitatively differs in *which* directions get added beyond the generalizing solution. Inter-layer alignment should drop in the overfit case.

- **LoRA vs Full Fine-tuning: An Illusion of Equivalence** (Shuttleworth et al., 2024) introduces *intruder dimensions*: high-singular-value vectors in the fine-tuned weights that are nearly orthogonal to *every* pretrained singular vector. They show these dimensions cause forgetting of the pretraining distribution. Critically, they show this is a LoRA artifact in their setting; full fine-tuning does not produce them.

  **Our prediction:** Overfit full fine-tuning *does* produce intruder dimensions — they were absent in the original paper because the comparison used early-stopped full fine-tuning. Push past convergence and the intruder dimensions emerge as the model encodes per-example information.

- **Hessian-based sharpness** (Keskar et al., 2017; Dinh et al., 2017): generalizing models sit in flatter regions of the loss landscape; sharp minima generalize worse on average. This has been challenged (Dinh et al. show sharp minima *can* generalize, depending on reparameterization), but in the controlled comparison of paired-architecture models, sharpness remains a strong correlate.

  **Our prediction:** Top-k Hessian eigenvalues of overfit models are anisotropically inflated, and the eigenvectors of the largest Hessian eigenvalues *align with the intruder singular directions in weight space*. This is a non-trivial structural prediction — Hessian curvature and weight-space SVD have not been jointly analyzed in this context.

### 2.2 Representation-space lens (activations)

- **Memorization is localized to ~5 neurons** (Maini et al., 2023): memorization of individual examples is confined to a small set of neurons (often around 5), distributed across multiple layers — *not* concentrated in late layers as previously believed. Their *example-tied dropout* deliberately routes memorization to known neurons, and zeroing them drops memorized-example accuracy from 100% → 3% while reducing the generalization gap.

  **Our prediction:** Overfit models contain *more* memorization neurons than generalizing ones, identifiable by the same gradient-accounting and critical-neuron-removal procedures. The set of memorization neurons grows in a structured way as overtraining proceeds.

- **Sparse autoencoder analysis of fine-tuning** (Nadipalli, 2025): SAEs reveal that fine-tuning produces layer-wise specialization — early layers retain general features, middle layers transition, late layers specialize. SAEs are now a standard tool for finding interpretable features inside trained networks.

  **Our prediction:** Training a small SAE on the residual stream of paired models will recover *additional* features in the overfit model that activate on a small number of training examples (effectively per-example features). The generalizing model's SAE features fire on broader semantic categories.

- **Counterfactual memorization** (Zhang et al., 2023; Feldman & Zhang, 2020): training data extraction works on overfit models (Carlini et al., 2021); the membership inference attack literature has shown for years that overfitting is sufficient (though not necessary) for privacy leakage.

  **Our prediction:** The membership inference advantage (MIA AUC − 0.5) of the overfit model scales with the spectral signatures from §2.1. This is a non-trivial empirical linkage: a weight-space metric should predict a representation-level privacy property.

### 2.3 Circuit-level lens (causal structure)

- **Progress measures for grokking** (Nanda et al., 2023): the canonical paper. Three phases — memorization, circuit formation, cleanup — separable by interpretability progress measures (restricted loss, excluded loss). Crucially: **the memorization mechanism and the generalizing mechanism coexist for a while**. Cleanup is when the generalizing mechanism is fully formed and the memorization mechanism is removed by weight decay.

  **Our prediction:** Overfit models without weight decay show the memorization circuit *persisting indefinitely* alongside whatever generalizing circuit forms. We should be able to (a) identify the memorization circuit by activation patching, (b) ablate it without harming test accuracy, and (c) show that the generalizing circuit, when isolated, has lower training accuracy than the full model but maintains test accuracy.

- **Differential learning kinetics in ICL** (Nguyen & Reddy, 2024): memorizing and generalizing sub-circuits are "largely independent" in transformers learning in-context tasks. Independent learning rates between sub-circuits explain the transition.

  **Our prediction:** This independence generalizes beyond ICL. The two circuits are causally separable in fine-tuned models too — patching one doesn't disrupt the other.

- **Activation patching best practices** (Zhang & Nanda, 2023; Heimersheim & Nanda, 2024; Geiger et al., 2023): the methodology for clean causal claims about model components. We will use *denoising* patching (replace corrupted activations with clean ones) and resample ablations to avoid the off-distribution problems of zero-ablation.

### 2.4 Loss landscape lens (geometry)

- **Sharpness and flat minima** (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017; Foret et al., 2021): the SAM line of work establishes that the loss-landscape geometry around the minimum correlates with generalization.
- **Linear mode connectivity** (Frankle et al., 2020; Entezari et al., 2021): paired models trained from the same init reach connected basins; cross-init pairs do not (without re-basin). We can use the *barrier height* between an overfit and a generalizing copy as a coarse measure of how different they are.
- **Permutation-invariant LMC for Transformers** (Sharma et al., 2025): recent work extends LMC to transformers via richer symmetry classes. Useful if our setup includes transformer fine-tuning.

  **Our prediction:** Two generalizing seeds will linearly mode-connect more cleanly (lower barrier) than an overfit/generalizing pair from the same init. Sharpness anisotropy (top-k Hessian eigenvalue ratio) will be substantially higher for the overfit model.

### 2.5 What's missing from the literature

Reading across all four lenses, we see a clear gap. Existing work either:

- analyzes *one* lens at a time without checking whether the others agree (most work), or
- shows correlations between regimes (rank ↔ generalization, sharpness ↔ generalization) without performing the targeted interventions that distinguish causation from correlation, or
- compares architectures, optimizers, or scales — not the same-architecture, same-data, only-training-regime-different comparison.

Our contribution is precisely that controlled comparison, run through all four lenses simultaneously, with causal interventions linking them. We are not introducing a new interpretability tool. We are using the existing toolkit on the right comparison — and that comparison has not been done with this level of rigor.

---

## 3. Hypotheses (testable, falsifiable)

We pre-register four primary hypotheses (H1–H4) and four secondary ones (H5–H8). Each is paired with a falsification test.

### Primary

**H1 (Spectral signature exists and is qualitative).** Holding architecture, data, and optimizer fixed, the empirical spectral density (ESD) of overfit-model weight matrices differs from generalizing-model ESDs by a measurable *categorical* property — specifically, the power-law fit α (Martin–Mahoney) of at least one layer differs by more than one HT-SR phase, and intruder dimensions appear with singular values exceeding the largest pretrained / early-stopped singular value.
- **Falsification:** if the SVD distributions overlap within seed-to-seed variance, this hypothesis is wrong and the overfit model is just a noisier point on the same axis.

**H2 (Memorization subspace is causally identifiable).** There exists a low-dimensional weight-space subspace V ⊂ R^{d×d} (we hypothesize: spanned by intruder singular directions) such that projecting the overfit weights onto the orthogonal complement V^⊥ restores test accuracy to within ε of the generalizing model's, *without retraining*, while specifically reducing training-set accuracy on memorized examples and leaving train accuracy on non-memorized examples intact.
- **Falsification:** if no such subspace exists at any rank cutoff that simultaneously improves test and degrades training-only-on-memorized, the memorization subspace hypothesis fails — overfitting is then distributed throughout the weights in a way that resists low-rank intervention.

**H3 (Per-example structure in activations).** A linear probe trained on the overfit model's residual stream at some layer ℓ can predict training-example identity (which one of N training examples produced this input perturbation) at accuracy substantially above what the same probe achieves on the generalizing model, *for the same architecture, same examples, same probe regime*. The advantage must exceed the probe's selectivity baseline (Hewitt & Liang, 2019).
- **Falsification:** if probing accuracy is comparable, the overfit model is not encoding more per-example information in its activations than the generalizing model — even though it gets lower train loss.

**H4 (Causal coupling across lenses).** The intervention of H2 (orthogonal projection in weight space) reduces:
- (a) the linear-probe per-example identifiability of H3, and
- (b) the top-k Hessian eigenvalue ratio (sharpness anisotropy), and
- (c) the membership-inference attack AUC,
in a coordinated way: specifically, the same projection rank cutoff that restores test accuracy should simultaneously normalize these three quantities to within the generalizing model's range.
- **Falsification:** if the four lenses decouple under intervention — e.g., orthogonal projection restores test acc but doesn't affect MIA AUC — the lenses are measuring different mechanisms, and our "single mechanism" claim fails. (This would still be a publishable result, but a weaker one.)

### Secondary

**H5 (Inter-layer singular vector alignment drops).** Following Yunis et al. (2024), the alignment between top singular vectors of W_{ℓ} and W_{ℓ+1} is lower in overfit models. Falsification: alignment is unchanged or increases.

**H6 (Memorization neurons are MORE numerous in overfit models).** Following Maini et al. (2023), the set of "critical neurons" whose ablation collapses training accuracy on a held-out memorized subset is larger and more distributed in overfit models. Falsification: the count is comparable.

**H7 (Sparse autoencoder reveals per-example features).** A SAE trained on the residual stream of the overfit model recovers features whose firing patterns are concentrated on ≤10 training examples each, more frequently than a SAE trained with identical hyperparameters on the generalizing model. Falsification: feature concentration distributions overlap.

**H8 (Spectral signature is partially shared across overfit seeds).** Two overfit models trained from different seeds on the same data exhibit a similar *qualitative* spectral phase (Martin–Mahoney) per layer, despite differing in which specific singular vectors are populated. Falsification: phases vary randomly across seeds; the signature is just noise.

---

## 4. Experimental design

The design follows three model "tracks" of increasing realism. Each track answers different questions, and we run all three for triangulation.

### Track A — Algorithmic ground truth (modular addition)

**Why:** Nanda et al.'s setup is fully reverse-engineered. We know what generalizing looks like (Fourier-multiplication circuit). We can compare to a memorization-only model where weight decay is removed, which never groks. This gives us a clean "ground truth" for what a memorizing-only solution looks like mechanistically.

**Setup:**
- 1-layer transformer, 128-dim residual stream, 4 heads, modular addition mod p=113 (Nanda's exact setup).
- Training: 30% of data, AdamW.
- Two model regimes per seed:
  - **Memorize-only (M):** weight decay = 0, train to 100% training accuracy, hold there for 50k steps past convergence.
  - **Generalize (G):** weight decay = 1.0, train past the grokking transition + 10k steps.
- 8 seeds each.

**Predictions specific to this track:**
- M models will exhibit *full-rank* embedding matrices; G models will have ~10 dominant singular directions corresponding to the key Fourier frequencies.
- The Fourier-basis "key frequencies" identifiable by Nanda's procedure should be *absent* from M models.
- Restricted loss (loss after ablating non-key frequencies) is high for M, low for G. Excluded loss (loss when ablating key frequencies) is the inverse.
- Per-example probe identifiability (H3) is high for M, low for G.

**Why this is the right starting point:** Sanity check. If our methods don't recover the known difference here, they won't recover anything subtler downstream.

### Track B — Vision (small CIFAR / Tiny-ImageNet)

**Why:** Standard, reproducible, computationally cheap. The natural extension of the original idea to a realistic supervised setting.

**Setup:**
- ResNet-18 trained from scratch on CIFAR-10. (Optionally Tiny-ImageNet for scale check.)
- Two regimes per seed, *identical except for*:
  - **Overfit (O):** no weight decay, no data augmentation, train for 800 epochs (well past convergence; train loss → ~0 by epoch 100, then continued training causes the test loss to begin to climb — that climb is the overfitting we want to characterize).
  - **Generalize (G):** weight decay = 5e-4, mixup + RandAugment, early-stopped at peak validation accuracy.
- 5 seeds.
- *Critical control:* We also train **O-noisy:** O regime with 20% label noise on a fixed subset. This gives us a known set of "definitely memorized" examples (the noisy-labeled ones) to use as ground truth for memorization-vs-generalization analyses.

**Per-example memorization scores:** We compute the Feldman–Zhang memorization score on a subsample using leave-one-out style: train a few "leave-this-example-out" models on the same data minus 50 examples each, and compute Δ accuracy on those examples. Computationally feasible for CIFAR-10 / ResNet-18.

### Track C — LLM fine-tuning (Pythia)

**Why:** Maps to the original framing about *fine-tuned* models, where overfitting is the practical problem. Lets us test whether the signatures generalize to causal LMs and to LoRA.

**Setup:**
- Base model: Pythia-160m and Pythia-410m (small enough for single-GPU full-fine-tuning).
- Task: a small classification or instruction-following dataset (e.g., 1k examples of a synthetic regex-based task that has clear in-distribution and OOD splits — we want clean memorization vs. generalization separation, not noisy real-world data).
- Three regimes:
  - **G (generalize):** full FT, early-stopped on held-out, weight decay = 0.1.
  - **O (overfit):** full FT, no weight decay, train 100 epochs past convergence.
  - **O-LoRA:** LoRA fine-tune (rank 16) with the same 100-epoch over-training. This replicates Shuttleworth et al.'s intruder dimensions phenomenon and lets us compare full-FT overfit vs. LoRA overfit.

**Memorization probe:** for an LM, "training example identity" is fuzzier. Use:
- Verbatim memorization (Carlini et al., 2022): can the model complete training-set prefixes verbatim?
- Counterfactual likelihood difference (Feldman & Zhang–style on a subset).

### Architecture-level controls

All three tracks share the following:
- **Same-architecture controls:** every comparison is within-architecture. We are *never* comparing across architectures.
- **Multi-seed:** all results reported as mean ± stddev across at least 5 seeds.
- **Compute budget per regime is matched** *within rounding* (i.e., we don't accidentally give the generalizing model more / fewer gradient updates).
- **Identical optimization hyperparameters** except the explicit knobs (weight decay, augmentation, early stop) that define the regime.
- **Test-set frozen at start.** No test-set look-ups during method development.

---

## 5. Mechanistic analysis pipeline

This section is the heart of the methodology. For each model pair (O, G) in each track, we run the following analyses.

### 5.1 Weight-space analysis (H1, H2, H5, H8)

**Per-layer ESD analysis (Martin & Mahoney).**
For each weight matrix W ∈ R^{m×n} (m ≥ n), compute the singular values σ_1 ≥ σ_2 ≥ ... ≥ σ_n. The ESD is the empirical distribution of {σ_i^2 / n}. We fit:
- **Marchenko–Pastur** at initialization (random-like phase) as control.
- **Power-law tail** with exponent α via the maximum-likelihood estimator on the upper 1/3 of the tail. Classification into Martin–Mahoney phases is determined by α and tail behavior:
  - α > 5: bulk + spikes
  - 2 < α < 4: Heavy-Tailed (HT) — the "good generalization" phase
  - α < 2: pathological / rank-collapse
- Layer-wise α distribution → *Weighted Alpha* = ∑ α_ℓ × log σ_max,ℓ.

**Intruder-dimension detection (Shuttleworth et al.).**
For paired pretrained W_pre and fine-tuned W_ft (we use a synthetic pretrained for Track B — random init treated as pretrained; for Track C the actual Pythia init):
1. SVD both: W_pre = U_pre Σ_pre V_pre^T, W_ft = U_ft Σ_ft V_ft^T.
2. For each top-k singular vector v_ft^i, compute max cosine similarity to {v_pre^j}_j.
3. If max similarity < ε (default 0.4), flag as intruder dimension.
4. Compare intruder counts and intruder singular-value magnitudes between O and G.

**Effective rank and stable rank.**
- Effective rank: exp(entropy of normalized singular value distribution) — captures how "spread out" the spectrum is.
- Stable rank: ||W||_F^2 / ||W||_op^2 — captures spectral concentration.
Compute layer-wise and aggregated.

**Inter-layer singular-vector alignment (Yunis et al.).**
For consecutive layers ℓ, ℓ+1, compute |⟨v_ℓ^i, u_{ℓ+1}^i⟩| for top-k singular vector pairs after appropriate matching. Alignment metric = top-k cosine similarity averaged.

### 5.2 Loss-landscape analysis (H4)

**Top-k Hessian eigenvalues.**
We use the Lanczos algorithm (PyHessian library) to estimate the top-50 eigenvalues of the loss Hessian ∇^2 L(θ) at convergence, *evaluated on the training set*. We report:
- λ_1 (sharpness).
- λ_1 / λ_50 (anisotropy ratio).
- Trace(H) / dim(θ) (mean curvature).

**Hessian-SVD alignment.**
For each layer, project the top-k Hessian eigenvectors v_H^i onto the weight matrix's left/right singular vector bases. Specifically: viewing v_H^i restricted to W_ℓ's slice as a matrix V_H^{i,ℓ}, compute its overlap with the column spans of intruder singular vectors. Non-trivial alignment is the predicted structural connection between curvature and weight-space.

**Loss landscape visualization.**
2-D loss landscape plot (Li et al., 2018) in the plane spanned by the top-2 Hessian eigenvectors, for both O and G models, on the same scale.

### 5.3 Representation analysis (H3, H6, H7)

**Per-example identity probe (H3).**
For each layer ℓ and each model M ∈ {O, G}:
1. Pass each training example x_i through M, extract activation h_ℓ(x_i).
2. Train a linear classifier on {(h_ℓ(x_i), i)}_i with N_train classes (one per training example).
3. Hold out 20% of the (x_i, i) pairs for the probe's own test set.
4. Report probe accuracy.

Critical control following Hewitt & Liang (2019): also train a **control probe** on randomized labels and report **selectivity = real probe acc − control probe acc**. We compare *selectivity* across O and G, not raw accuracy.

A subtler version: instead of arbitrary index, use the *Feldman–Zhang memorization score* of each example as the regression target. We expect overfit-model activations to have a much larger linear subspace that correlates with the score.

**Sparse autoencoder analysis (H7).**
Train a sparse autoencoder (Cunningham et al., 2024; Templeton et al., 2024) on the residual stream of each model at layer L/2 and 3L/4.
- SAE width: 8× the residual stream dim.
- L1 sparsity penalty tuned so that ~20–50 features fire on average per token.
- Identical hyperparameters across O and G.

Then, for each feature f:
- Compute the firing frequency across the training set: how often does f activate above threshold?
- A "per-example feature" is one whose top-k activating examples all come from a tight cluster (small set of training examples), with low activations elsewhere.

Compare the distribution of feature concentration (Gini coefficient over training-example firing rates, say) between O and G.

**Memorization neuron identification (H6).**
Following Maini et al. (2023), for each model:
1. Compute per-example gradient norms ||∇_θ L(x_i)||_{by neuron} during the final epoch.
2. For high-memorization examples (Feldman–Zhang top-decile, or the noisy-labeled set in Track B's control), identify the neurons with largest gradient contributions.
3. Verify by **example-tied dropout-style ablation:** zero those neurons and check that training accuracy on those specific examples drops while test accuracy is preserved.

Report neuron count, distribution across layers, and overlap with the intruder singular directions when these neurons are viewed as one-hot vectors in weight space.

### 5.4 Causal interventions (H2, H4)

This is the most novel part of the pipeline — and the source of the strongest claims.

**Intervention 1: Orthogonal projection in weight space.**
For each weight matrix in the overfit model:
1. Compute SVD: W_O = U Σ V^T.
2. Identify intruder singular directions {v_O^i : i ∈ I_intruder}.
3. Project: W_O' = W_O − ∑_{i ∈ I_intruder} σ_i u_i v_i^T.
4. Evaluate W_O' on:
   - Train accuracy (overall)
   - Train accuracy (memorized subset, e.g., Feldman–Zhang top-decile)
   - Train accuracy (non-memorized subset)
   - Test accuracy
   - Top-k Hessian eigenvalues
   - Per-example probe accuracy
   - MIA AUC

We sweep the threshold ε for intruder classification and the rank cutoff to characterize the trade-off curve. Success looks like: a setting where test accuracy ↑, memorized-train accuracy ↓, non-memorized-train accuracy ≈ unchanged.

**Intervention 2: Activation patching memorization → generalization.**
Take a memorized training example x_mem. Forward-pass x_mem through O and G separately, saving residual stream activations at each layer. Then patch G's activations into O's forward pass at progressively more layers and locations. Identify the *minimal* patching set that flips O's prediction on x_mem to match G's. This tells us *where* O's memorization is implemented at the activation level.

Reverse direction: patch O's activations into G's forward pass to see whether memorization is transferable — i.e., does the generalizing model become "memorizing" when given the overfit model's late-layer activations on a memorized example?

**Intervention 3: Zero-shot un-memorization via subspace surgery.**
For each identified intruder direction with singular value σ:
1. Scale down: W' = W − γ σ u v^T, for γ ∈ {0, 0.25, 0.5, 0.75, 1.0}.
2. Measure how memorization-relevant behavior decays as γ → 1.
3. Compare to scaling down a *non-intruder* direction of similar singular value (control).

If intruder directions specifically carry memorization, scaling them down should be qualitatively more effective than scaling random directions of similar magnitude.

### 5.5 Mode-connectivity checks (H4)

For the same-init paired (O, G) models:
- Compute the loss along the linear interpolation θ_t = (1−t)θ_O + tθ_G for t ∈ [0, 1].
- Plot train and test loss along this path.
- The "barrier height" max_t L(θ_t) − max(L(θ_O), L(θ_G)) tells us whether O and G are in the same basin.

Predicted shape: train loss has a U-shape on this path (both endpoints are low-loss for train); test loss is monotone or unimodal toward G. If they're in the same basin, the path is smooth; if not, there's a hump.

For cross-init seeds, apply the permutation re-basin procedure (Ainsworth et al., 2023; for transformers: Sharma et al., 2025) before interpolating.

---

## 6. Falsification protocols and controls

TMLR explicitly evaluates whether claims are matched by evidence. The single biggest threat to this kind of work is "I found a difference, therefore my theory is right" — when in fact most reasonable perturbations would yield *some* difference. Pre-registering falsification is non-negotiable.

### 6.1 Pre-registered "negative" comparisons

We compare not just (O, G) but a battery of pairs:

- **(O, G)** — main comparison.
- **(G_seed1, G_seed2)** — two generalizing models with different seeds. *Should look similar* under all four lenses. If we see a "qualitative" difference here too, our O-vs-G claim is just noise sensitivity.
- **(O_seed1, O_seed2)** — two overfit models. *Should share the qualitative signature* (H8) but differ in details. If they share details, the signature isn't seed-robust; if they don't share the signature, H1/H8 are wrong.
- **(G, G+noise)** — generalizing model with small random Gaussian noise added to weights, matched to ||θ_O − θ_G||. *Should NOT look like overfit model.* If random-direction perturbation reproduces the overfit signature, then "overfitting is just adding noise" wins and our claim collapses.
- **(G, G_compressed)** — generalizing model after low-rank truncation to match O's stable rank. *Should NOT look like overfit model.* This is a particularly important control: if the spectral signature can be reproduced by simply matching rank statistics, then rank doesn't capture the qualitative thing we care about.

### 6.2 Pre-registered "sanity" checks

- The known-grokking case (Track A) must recover the documented Fourier-multiplication circuit in G models and not in M models. If this fails, the whole pipeline is broken.
- The Feldman–Zhang memorization scores must correlate with our probe's per-example identifiability scores in O models. If our probes don't correlate with the standard memorization metric, they're measuring something else.
- For Track B, the noisy-labeled examples must score in the top decile of our memorization-relevant signatures. If they don't, our metrics aren't tracking memorization.

### 6.3 Compute-matched ablations

A common reviewer objection: "Maybe your overfit model just has more capacity used because it's trained longer." Counter this by running:
- **G_longtrain:** generalizing regime (weight decay, augmentation) but trained for the same number of epochs as O. Should reach a similar test acc as G; should NOT show the overfit signature.
- **O_shorttrain:** overfit regime hyperparameters but stopped at epoch 100 (right at convergence). Should be intermediate.

This separates "training-long" from "training-without-regularization."

---

## 7. 30-day execution plan

The plan is calibrated for a single researcher with one GPU (or modest cluster access). Compute estimates are conservative.

### Week 1 (Days 1–7): Setup, pipeline, Track A

| Day | Task |
|---|---|
| 1 | Repo setup, dependencies (PyTorch, TransformerLens, PyHessian, fast.ai for vision, transformers/peft for LLMs). Reproduce Nanda's modular addition setup from his public colab. |
| 2 | Track A: train all 16 models (8 seeds × {M, G}). Each takes ~2 hours on a single GPU. |
| 3 | Implement weight-space analysis: ESD, Martin–Mahoney α fit, intruder dimensions, effective/stable rank, inter-layer alignment. Validate on Track A models. |
| 4 | Implement loss-landscape analysis: Lanczos for Hessian eigenvalues. Validate on Track A. |
| 5 | Implement representation analysis: per-example probes, basic SAE. Train and analyze on Track A. |
| 6 | Implement causal interventions: orthogonal projection, activation patching (use TransformerLens hooks). Validate on Track A. Critical: confirm M→G subspace surgery works on the well-understood case. |
| 7 | Track A writeup draft. Verify all hypotheses predicted-vs-observed on Track A. Decide whether to refine hypotheses (allowed before locking in Track B). |

**Gate:** at end of Week 1, hypotheses H1–H4 must be substantially confirmed on Track A or refined before proceeding. If they all fail on Track A, the project pivots — likely toward "characterizing the regime where the hypothesis *does* hold" rather than abandoning, since negative results on Track A would themselves be informative.

### Week 2 (Days 8–14): Track B (Vision)

| Day | Task |
|---|---|
| 8 | Set up CIFAR-10 + ResNet-18. Train 5 seeds × {O, G, O-noisy, G_longtrain, O_shorttrain}. ~5 hours each × 25 runs but parallelizable. |
| 9 | (Continue training; in parallel) Implement Feldman–Zhang memorization score computation on a subset of CIFAR-10. Train ~50 leave-out models on subsamples. |
| 10 | Run weight-space analysis on Track B models. Generate spectral signature plots. |
| 11 | Run loss landscape analysis on Track B. Hessian-SVD alignment plots. |
| 12 | Run representation analysis on Track B: probes, SAE, memorization neurons. |
| 13 | Run causal interventions on Track B. The big test: does orthogonal projection restore G behavior? |
| 14 | Track B writeup. Cross-check with Track A predictions. |

**Gate:** at end of Week 2, decide whether Track C (LLM) is necessary for the paper or whether tracks A + B suffice. TMLR doesn't require massive scale — a clean small story is fine. If we're tight on time, drop Track C and use the extra time for stronger controls and writing.

### Week 3 (Days 15–21): Track C (LLM) + controls

| Day | Task |
|---|---|
| 15–16 | Pythia-160m / 410m fine-tuning. 5 seeds × {G, O, O-LoRA}. The 410m models will take longer; budget accordingly. |
| 17 | Verbatim memorization extraction, MIA AUC computation. |
| 18 | Weight-space analysis on Pythia models. Compare to Shuttleworth et al.'s reported intruder dimensions. |
| 19 | Causal interventions on Pythia. Subspace surgery on attention weight matrices. |
| 20 | Pre-registered negative controls: (G_seed1, G_seed2) and (G, G+noise) and (G, G_compressed). |
| 21 | Cross-track integration. Build the master plot showing all four lenses agreeing on the O–G distinction across all three tracks. |

### Week 4 (Days 22–30): Writing, additional experiments, refinement

| Day | Task |
|---|---|
| 22 | Outline the paper. ~8 pages of TMLR (no hard limit, but stay focused). Sections: intro, related, hypotheses, methods, results, discussion, limits. |
| 23–24 | Write methods + experimental setup. |
| 25 | Write results. Lead with the cleanest, most surprising plot. (Likely candidate: orthogonal projection in weight space restoring generalization without retraining.) |
| 26 | Write related work, situating against grokking, double descent, Feldman, Martin–Mahoney, Shuttleworth. |
| 27 | Write intro, limitations, broader impact. |
| 28 | Run any follow-up experiments needed (reviewers will ask "what about X" — preempt). |
| 29 | Internal review. Tighten claims. Make sure no claim is stronger than the evidence. |
| 30 | Submit to TMLR via OpenReview. |

### Compute estimate

- Track A: 16 runs × 2 hours ≈ 32 GPU-hours
- Track B: 25 runs × 5 hours ≈ 125 GPU-hours
- Track B leave-out for FZ scores: 50 runs × 3 hours ≈ 150 GPU-hours (this is the big chunk; can be subsampled)
- Track C: ~20 runs at varying scale ≈ 100 GPU-hours
- Analysis runs (probes, SAEs, intervention sweeps): ≈ 100 GPU-hours
- **Total:** ~500 GPU-hours = ~21 days on a single A100, OR ~3 days on 8 A100s.

This is achievable on a single workstation (one A100 / 4090 / H100) running 24/7 if Tracks A and B only, ~2 weeks of compute. With cluster access, Track C is comfortable.

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| H1 partially holds but lenses don't agree (H4 fails) | Medium | Still publishable as "lenses are partially decoupled" — but the headline changes from "single mechanism" to "lens taxonomy". |
| Orthogonal projection doesn't restore generalization | Medium-low | Try other subspace identifications: weighted by Hessian-eigenvector overlap, weighted by per-example gradient magnitudes. Multiple shots at H2. |
| Feldman–Zhang scores are too compute-intensive | Medium | Approximate with single leave-out estimates or use the published ImageNet memorization scores from Feldman & Zhang directly on a Track B variant. |
| Pythia fine-tuning is too slow / unstable | Medium | Use Pythia-70m or DistilGPT-2 instead. The scale isn't the point. |
| Selectivity baseline (Hewitt & Liang) eats most of our probe advantage | Low-medium | Run randomized-label control probes; report selectivity, not raw accuracy. If selectivity is small, retract H3 claims to that effect. |
| Findings are obvious in hindsight ("of course an overfit model memorizes examples") | High | The novelty isn't *that* it memorizes — it's the *structural characterization* and *causal restoration without retraining*. Frame accordingly. |
| Reviewer says "you just rediscovered LoRA intruder dimensions" | Medium | (a) We extend to full fine-tuning and SGD-from-scratch. (b) We show the same signature in regimes where Shuttleworth et al. claimed intruder dimensions don't appear (full FT). (c) We link to three other lenses they didn't analyze. |
| Reviewer says "this is just grokking framed differently" | Medium-high | Show on standard CIFAR-10 + ResNet-18 with real labels where grokking does not occur. Our claim is broader than the grokking regime. |
| Out of compute for Track C | Medium | Track A + B is enough for TMLR. Track C becomes appendix. |

---

## 9. Why this clears the TMLR bar

TMLR's stated criterion is: are the claims supported by evidence, and is the work of interest to *some* part of the ML community? They explicitly de-emphasize "significance" and emphasize "correctness." Concretely:

- **Claims are matched to evidence.** Every primary hypothesis has a falsification test, multiple operationalizations, and a pre-registered control.
- **Methodology is standard, well-validated.** We use TransformerLens (Nanda et al.) for circuit analysis, PyHessian for spectral analysis, SAE methodology from Cunningham et al. / Templeton et al., and Maini et al.'s memorization-neuron procedure. We do not invent new tools.
- **The novel claim is the structural characterization of an under-studied object** — the overfit model as a distinct kind of representation, with three or four mutually-corroborating signatures. This is the type of "negative space" finding TMLR welcomes.
- **The work is reproducible.** Public datasets, small models, modest compute. Code release is built into the plan.

For a TMLR submission, the "Featured Certification" is the more aspirational target — it requires reviewer enthusiasm. The minimum acceptable claim is enough to clear TMLR's bar with normal acceptance: even if only H1 (spectral signature exists) and H3 (per-example identifiability) hold and H2 (causal restoration via subspace surgery) fails, the paper is still a useful contribution. If H2 holds, the paper becomes notable enough to be worth promoting.

---

## 10. Open questions to refine before locking the plan

These are decisions that should be made in the first 24 hours and locked:

1. **Do we want grokking as a "mode" or as a control?** Track A's M models never grok (no weight decay). We could also include a "grokked" condition. Argument for: makes the cleanup-incomplete framing very crisp. Argument against: complicates the design and adds another comparison axis.

2. **Pretrained vs. from-scratch for Track B.** From-scratch ResNet on CIFAR is the cleanest. Pretrained → fine-tuned matches the "overfit fine-tuned" framing more directly. We propose from-scratch for the main story and pretrained as a robustness experiment if time permits.

3. **Which singular vectors are "intruder"?** Threshold-sensitive. We will sweep ε ∈ {0.2, 0.3, 0.4, 0.5} and report results across the sweep. The most robust finding should not depend strongly on ε.

4. **How many training examples to track per-example?** Probing on all 50k CIFAR examples is heavy. Subsample to 500–2000 with stratified sampling by Feldman–Zhang score.

5. **Probe choice.** Linear probes only (per Belinkov, 2022 recommendations). No MLP probes — they memorize.

6. **Does "overfit" mean "trained until test loss climbs" or "trained way past test-loss peak"?** Pre-commit to: trained for K epochs past the epoch of peak validation accuracy, where K = 3× the time to peak. Lock the K choice on Day 1.

---

## 11. Annotated bibliography (most-cited 25)

Each entry is annotated with its role in our project.

### Foundational generalization-vs-memorization

- **Zhang, Bengio, Hardt, Recht, Vinyals (2017).** *Understanding deep learning requires rethinking generalization.* ICLR. — The starting point: deep nets fit random labels. Establishes that train loss → 0 is not informative about generalization. Provides our random-label baseline.
- **Feldman (2020).** *Does learning require memorization? A short tale about a long tail.* STOC. — Theory: memorization of long-tail examples is necessary for near-optimal generalization. Justifies treating memorization as informative, not bug.
- **Feldman & Zhang (2020).** *What neural networks memorize and why: Discovering the long tail via influence estimation.* NeurIPS. — Empirical version of Feldman 2020. Provides the per-example memorization-score methodology we use.
- **Zhang, Ippolito, Lee, Jagielski, Tramèr, Carlini (2023).** *Counterfactual memorization in neural language models.* NeurIPS. — Definitional refinement: counterfactual memorization separates "memorized" from "would have been predicted anyway." Used for Track C.

### Grokking and phase structure

- **Power, Burda, Edwards, Babuschkin, Misra (2022).** *Grokking: Generalization beyond overfitting on small algorithmic datasets.* arXiv. — Establishes the phenomenon: delayed generalization. The "overfit-then-grok" trajectory is exactly the regime where O and G differ structurally.
- **Nanda, Chan, Lieberum, Smith, Steinhardt (2023).** *Progress measures for grokking via mechanistic interpretability.* ICLR. — The Fourier-multiplication algorithm. Defines the three-phase model: memorization → circuit formation → cleanup. Our Track A's M condition is "stuck in phase 1."
- **Yunis, Patel, Wheeler, Savarese, Vardi, Livescu, Maire, Walter (2024).** *Approaching deep learning through the spectral dynamics of weights.* — Shows random-label vs. true-label produce qualitatively different spectra. The cleanest precedent for our weight-space lens.

### Spectral / Random Matrix Theory

- **Martin & Mahoney (2019, 2021).** *Implicit self-regularization in deep neural networks* (JMLR) and *Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data* (Nat. Commun.). — Heavy-Tailed Self-Regularization theory. The 5+1 phases. The α exponent. Our primary weight-space metric.
- **Shuttleworth, Andreas, Park, Kim (2024).** *LoRA vs full fine-tuning: An illusion of equivalence.* NeurIPS. — Defines intruder dimensions. We extend to overfit full fine-tuning.

### Localization of memorization

- **Maini, Mozer, Sedghi, Lipton, Kolter, Zhang (2023).** *Can neural network memorization be localized?* ICML. — Memorization is in ~5 neurons distributed across layers (not just the final layer). Provides our memorization-neuron identification methodology.
- **Hartke et al. (2024).** *Finding NeMo: Localizing neurons responsible for memorization in diffusion models.* — Generalizes to diffusion. Cross-modality robustness check.

### Loss-landscape geometry

- **Keskar, Mudigere, Nocedal, Smelyanskiy, Tang (2017).** *On large-batch training for deep learning: Generalization gap and sharp minima.* ICLR. — Sharp vs. flat minima. Establishes the Hessian-sharpness correlate.
- **Dinh, Pascanu, Bengio, Bengio (2017).** *Sharp minima can generalize for deep nets.* ICML. — Important caveat: sharpness is reparameterization-dependent. Used to refine our sharpness claims.
- **Frankle, Dziugaite, Roy, Carbin (2020).** *Linear mode connectivity and the lottery ticket hypothesis.* ICML. — Establishes that same-init pairs reach connected basins; cross-init pairs need re-basin.

### Mechanistic interpretability methodology

- **Olah, Cammarata, Schubert, Goh, Petrov, Carter (2020).** *Zoom in: An introduction to circuits.* Distill. — Foundational circuits paper.
- **Wang, Variengien, Conmy, Shlegeris, Steinhardt (2023).** *Interpretability in the wild: A circuit for indirect object identification in GPT-2 small.* ICLR. — The IOI circuit. Methodology template for our circuit-discovery work.
- **Zhang & Nanda (2024).** *Towards best practices of activation patching in language models.* ICLR. — Methodology paper for clean causal claims. Our patching follows their recommendations.
- **Heimersheim & Nanda (2024).** *How to use and interpret activation patching.* arXiv. — Practical guide.
- **Cunningham, Ewart, Riggs, Huben, Sharkey (2024).** *Sparse autoencoders find highly interpretable features in language models.* ICLR. — Our SAE methodology.
- **Templeton et al. (2024).** *Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet.* — Larger-scale SAE methodology.
- **Belinkov (2022).** *Probing classifiers: Promises, shortcomings, and advances.* Computational Linguistics. — Methodology and pitfalls of probing. Selectivity controls are from here.

### Memorization extraction / privacy

- **Carlini, Tramèr, Wallace, Jagielski, Herbert-Voss, Lee, Roberts, Brown, Song, Erlingsson, Oprea, Raffel (2021).** *Extracting training data from large language models.* USENIX Security. — The classic training-data extraction paper. We use the methodology for Track C MIA.
- **Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang (2022).** *Quantifying memorization across neural language models.* ICLR. — Discoverable memorization metric.

### Closely related "mechanistic memorization" works

- **Henighan, Carter, Hume, Elhage, Lasenby, Fort, Schiefer, Olah (2023).** *Superposition, memorization, and double descent.* Anthropic Transformer Circuits Thread. — Toy-model treatment of how memorization arises in superposition. Our most direct theoretical precedent.
- **Nguyen & Reddy (2024).** *Differential learning kinetics govern the transition from memorization to generalization during in-context learning.* — Memorizing and generalizing sub-circuits are largely independent. Justifies our "two-circuit" view.
- **Wang, Yue, Su, Sun (2024).** *Grokked transformers are implicit reasoners.* arXiv. — Generalizing-circuit formation in grokked transformers.

---

## 12. Decision points for the user

Before locking the plan, you should decide:

1. **Tracks.** Run all three (A + B + C) or only A + B? Recommendation: A + B as core, C as appendix.
2. **Fine-tuning vs. from-scratch.** Your original framing emphasized "overfit fine-tuned." Track C captures this directly. Tracks A and B use from-scratch training. If the FT framing is non-negotiable, Track C becomes Track 1 and we re-prioritize.
3. **TMLR's "Featured Certification" vs. acceptance.** Featured requires more experimental breadth; standard acceptance requires correctness. The plan above targets standard acceptance with Featured as upside.
4. **Whether to add a theoretical companion.** A simple linear-network analogue (where intruder dimensions can be computed in closed form on a toy long-tail distribution) would strengthen the paper substantially. Adds ~3 days. Recommend including.

---

*End of plan v1.*
*Lock decisions, then execute Day 1.*