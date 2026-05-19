# Project: Structural Signatures of Memorization and Generalization

## What this project is, in one paragraph

We assemble **a single lens — a unified suite of structural measurements — and apply it consistently across many (architecture × dataset × training-regime) configurations.** The point is not to "diagnose" a model in isolation; the point is that when you look at all the regimes through the same lens, **patterns emerge that no single-regime study can reach**: which signatures separate which pairs of regimes, where signatures *decouple* (different signatures fire in different combinations across architectures), and where they reverse direction. These cross-regime patterns are the actual finding, not any one measurement. The diagnostic interpretation ("this model is memorizing because X, Y, Z fire together") is one downstream use; the comparative one ("sharpness reverses between modular Transformers and CIFAR ResNets, which tells us something about what WD does in each") is the scientifically generative one. Target venue: TMLR. Title (working): *Structural Signatures that Distinguish Memorizing from Generalizing Neural Networks*.

## The lens — what we measure

A unified suite of structural measurements applied across every (architecture × dataset × training-regime) configuration. Not a list of separate diagnostics — the same lens turned on every regime, so cross-regime patterns become visible.

| Signature | What it measures | What we've found so far |
|---|---|---|
| **Effective rank** of weight matrices | Shannon entropy of normalized squared singular values | High in M, low in G; gap is large in algorithmic + ResNet, weak in ViT |
| **Hessian top eigenvalue** (full data) | Sharpness of the loss landscape via Lanczos | M sharp in toy/ViT; **reverses** in ResNet (G sharper). Architecture-dependent. |
| **Hessian bottom eigenvalue** | Strictly negative directions = saddle topology | Strictly negative at M in 3/4 tiers; ≈ 0 at G in toy |
| **Gradient angle** cos(∇L_train, ∇L_test) | Whether training and test loss "agree" at convergence | Anti-aligned at toy M (−0.24); decouples to ~0 in ViTs |
| **Gradient norm ratio** ‖∇L_test‖/‖∇L_train‖ | Asymmetry between train and test loss gradients | 10¹⁰ at toy M (train grad is zero, test is huge); much smaller at G |
| **Weight L2 norm** ‖θ‖ | Total parameter magnitude | Standard generalization predictor (Omnigrok); engages Bartlett/Neyshabur norm-bound lit |
| **Distance from init** ‖θ_final − θ_init‖ | How far the optimizer traveled | Distinguishes lazy vs feature-learning regimes |
| **Path-norm proxy** Π‖W_i‖_op | Lipschitz upper bound, product of spectral norms | Best generalization predictor in Jiang et al. 2020 benchmark |
| **Fourier circuit presence** (task-specific) | Concentration of Fourier energy in embedding | Sharp in algorithmic G models, absent in M, N/A elsewhere |
| **Inter-layer singular vector alignment** | Adjacent layers' top SV cosine | Higher in G (Yunis-style) |
| **Loss-based MIA AUC** | Per-example train/test loss separability | **The one universal signature: M > G in 4/4 tiers**. Privacy-relevant. |
| **Logit margin distribution** | Train and test logit margins | M has uniformly large train margins + extreme negative test margins; G uniform |
| **Mode-connectivity barrier** | Loss along linear path between M and G | Large barrier ⇒ different basins; small barrier ⇒ same basin different points |
| **Permutation-aligned LMC** | Barrier after Hungarian-matching neurons | Distinguishes "same solution up to symmetry" from "fundamentally different solutions" |

The cross-regime *pattern* of which signatures fire (and which decouple, and which reverse direction) is the actual finding, not any one row of this table.

## The regimes we characterize

- **Pure overfit**: train=100%, test ≈ chance. Classic memorization. Toy: 1L Transformer on (a+b) mod 113 with wd=0.
- **Grokked**: train=100%, test=100% via delayed generalization. Toy: same setup with wd=1.0.
- **Benign overfit**: train=100%, test still good (~80%). Standard: ResNet-18 on CIFAR-10 with no WD/no augmentation.
- **Clean generalization**: train≈100%, test high, no overfitting gap. Standard: same ResNet with full regularization.
- **Random-label memorization**: train=100% on corrupted labels, test = chance. CIFAR with label-noise control.

The panel should fingerprint each of these distinctly.

## Three "tracks" of evidence

- **Track A** — algorithmic grokking on modular arithmetic mod 113 (1L and 4L Transformer). Cleanest signal, multi-seed (10+10 in bp7/bp9), fully characterized.
- **Track B** — vision: MLP MNIST, MLP FashionMNIST, ResNet-18 CIFAR-10, ResNet-50 CIFAR-100, ViT-Tiny CIFAR-10, ViT-Small CIFAR-100. Multi-seed in flight via bulletproof3 tier1-4.
- **Track C** — language modeling: char-LM Shakespeare from scratch (tier5), Pythia-160m fine-tuning on a Pride and Prejudice subset (tier6).

## The mechanism claim

Generalization is caused by **smooth norm-based regularization** (weight decay, nuclear norm, Frobenius²) — not by arbitrary rank reduction. Aggressive rank-shaping penalties (Schatten-1/2, log-singular, tail-singular) reduce rank but fail to generalize (bp5). Weight decay is also **uniquely effective** among standard interventions: SAM, Gaussian noise, label smoothing, and additional training data all fail to escape M's saddle (Entry 30). The minimum-norm NTK interpolator fails completely on modular addition (bp17, 0.3% test acc), ruling out kernel-regression as the explanation.

The geometric story for *why* this works: at M, ∇L_train is ≈0 (perfect fit) while ∇L_test is huge — the model sits at a saddle on the full-data loss landscape whose unstable direction has positive component toward G (Entry 15, bp3). WD provides the force to descend that unstable direction. Memorization is also fully reversible: M_t for any t can be rescued by WD continuation (Entry 18).

## Why structural signatures and not just train/test loss curves

A reviewer will ask this and it's a sharp question. Loss curves already classify the obvious regimes (overfit, generalize, benign overfit, still-learning). If the paper's claim is "we can tell M from G," loss curves dominate and the structural battery looks redundant.

The contribution must be one of these seven, not the redundant one:

1. **You don't always have the original training data.** Released checkpoints (Pythia, LLaMA, public models) come with weights but not the data they were trained on. Loss-on-train-set is uncomputable; structural signatures are. Be precise about what each signature needs:

   - **Tier A — weights only** (zero data): effective rank, stable rank, nuclear norm, operator norm, inter-layer SV alignment (Yunis), Martin-Mahoney spectral phase, attention head asymmetry, Fourier energy in W_E. Runs in minutes on any open checkpoint.
   - **Tier B — weights + small probe set** (~500-2000 examples, not original train): Hessian top/bottom eigenvalues, gradient norm. The probe set can be any compatible distribution.
   - **Tier C — weights + two distinguishable data subsets**: gradient angle cos(∇L_A, ∇L_B), logit margin distribution. The two sets don't have to be train/test specifically; any two reference distributions work.
   - **Tier D — weights + known member/non-member split**: MIA AUC. Needs at least suspected members + clean non-members.
   - **Tier E — two models + eval data**: mode-connectivity barrier.

   None of these require the *original* training data. That's the real "from weights" advantage — auditing and privacy claims on published models.

2. **Loss says THAT, signatures say WHY and WHAT TO FIX.** "Overfit" is a diagnosis. "Rank inflated to 97 with anti-aligned train/test gradients and strict negative Hessian directions" is an actionable mechanistic explanation. Entry 18 (WD-rescue always works) is only actionable because we measured what WD changes geometrically.

3. **Loss can't distinguish privacy-leaking from privacy-safe models that look identical.** Two models with train=100%, test=85% — one trained with WD, one without. Loss curves identical. **MIA AUC differs by 10×** (Entry 28 vs G). Rank differs by 25×. Loss doesn't see this. Signatures do. This matters for auditing released models.

4. **Mid-training, loss misleads.** During grokking, train=100% for 10k epochs while test stays at chance. Loss says "stuck." Rank trajectory says "actively compressing, will grok soon" (Entry 13). Yunis's contribution is built on this — spectral dynamics carry information loss curves don't.

5. **Regimes that look identical in loss differ structurally.** Pure overfit and random-label memorization both show train=100%, test≈chance. Different neuron specialization, different probe behavior, different saddle topology. Loss conflates them; signatures separate them.

6. **Mechanism explanation.** The paper's contribution isn't "we can guess test acc from train acc." It's "we explain *why* WD works, *why* aggressive rank penalties fail despite reducing rank (bp5), and *why* M is reversible by gradient methods while spectral surgery isn't (Entry 18 vs Entries 6-8)." Loss curves don't explain. Signatures do.

7. **Privacy is the killer app.** MIA AUC tracks rank gap and gradient angle, not train/test accuracy gap directly. A model can have small acc gap but high MIA leakage. Structural signatures predict privacy vulnerability that loss curves miss.

**The selling line for the paper** (do NOT lead with "we predict regime from signatures" — that's redundant with loss curves and a reviewer will dunk):

> "Loss curves classify learning regimes. Structural signatures explain them, predict them from weights alone, distinguish privacy-leaking from privacy-safe models that look identical on loss curves, and tell you which intervention will fix a given failure mode."

That's a real contribution beyond what loss curves give you.

## Honest scope

- Strongest claims hold in the algorithmic grokking regime (Track A) with 10-seed precision.
- Benign overfitting (Entry 28) shows the signature gap is preserved even when test accuracy is high.
- ViT shows weaker categorical separation than ResNet/MLP — likely a "benign overfitting" continuum case rather than sharp memorize-vs-generalize.
- LM signatures are pending (tier5, tier6 in flight).

## Position relative to existing work

- **Yunis et al. ICML 2024** measures effective rank qualitatively across architectures. We replicate with multi-seed precision and add ~9 other signatures, MIA, causal subspace interventions, and the saddle/reversibility story Yunis explicitly avoids.
- **Notsawo et al. ICML 2025** shows nuclear norm and L1 enable grokking (positive result). We add the *negative* result that aggressive rank penalties fail despite reducing rank — pinning the mechanism to *smooth* norm regularization.
- **Nanda et al. 2023** reverse-engineered the Fourier circuit in grokking. We use their setup but ask the geometric/regime question, not the mechanistic-interpretability question.
- **2602.18523** (multi-task grokking geometry) reports negative Hessian eigenvalues at both M and G. Our single-task results show much larger gap (200-300 at M vs 10⁻⁴ at G) — needs honest discussion in related work.

## Repo layout

- `taska/` — Track A: 1L Transformer modular addition, training + analysis scripts
- `trackb/` — Track B: vision experiments
- `diverse/` — early diverse-tasks scope-checking (MNIST, Shakespeare LM, tabular)
- `overnight/`, `overnight2/` — early HPC batches (rank-WD law, optimizer sweep, nuclear norm escape, etc.)
- `bulletproof/` — Hessian comprehensive, gradient direction, alternative rank penalties
- `bulletproof2/` — 10-seed structural battery (bp7), Lanczos Hessian (bp8), gradient angle (bp9), WD threshold sigmoid (bp11), NTK interpolator (bp17), etc.
- `bulletproof3/` — scale ladder: tier0-tier6 from 4L Transformer modular up to Pythia-160m fine-tune; also bug-fix rewrites
- `literatures/` — captured prior-work notes
- `so_far.md` — chronological reasoning log
- `so_far_results.md` — append-only empirical results record
- `literature.md` — internal mapping of our claims vs prior art
- `findings.md` — honest inventory of what's new vs replicated

## Conventions

- Every claim in `so_far_results.md` is paired with a script path and seed count.
- Failed/buggy experiments stay in the log (annotated), never deleted.
- Multi-seed (n ≥ 3) before reporting any new claim as "confirmed."
- Negative results are first-class — bp17 (NTK fails), bp_fix_probe (identity not linearly probeable), and bp5's aggressive-penalty failures are paper material.
- All HPC scripts use `_signatures.py` as the shared signature-computation module so results across tiers are directly comparable.

## What's running right now

bulletproof3 tier1, tier1b, tier2, tier3, tier3b, tier4, tier5, tier6, and tier0 are queued/running. Each tier trains M (no WD) and G (with WD) for multiple seeds and computes the full signature battery. Results will land over the next 3-5 days. Once in, the paper has empirical material across the full scale ladder from algorithmic toy to Pythia-160m fine-tuning.
