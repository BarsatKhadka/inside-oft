# so_far_results.md

Empirical results as they come in. Append-only — never delete a result, even if it's wrong. Annotate instead.

Format per entry:
- **Date**
- **Setup:** what I ran, hyperparameters, seed
- **What I expected:** my prediction *before* looking at the result
- **What happened:** the actual result, ideally with a plot or number
- **What it means:** my honest interpretation, including "I'm not sure" if I'm not sure

---

## Entry 1 — Grokking curves (Phase 2.1)
**Date:** 2026-05-17
**Setup:** Plotted train/test loss + accuracy curves from `taska/checkpoints/{G,M}/history.json`. G = AdamW lr=1e-3 weight_decay=1.0, seed=0, 50k epochs. M = same except weight_decay=0.0. Output: `taska/results/fig_grokking.png`.
**What I expected:** G should grok somewhere around epoch 10k–20k (Nanda's reported window). M should never grok — train loss → 0, test loss stays high.
**What happened:**
- G grokked at epoch **10,800** (first epoch where test_acc ≥ 0.99).
- Final G: train_acc 1.0000, test_acc 1.0000, train_loss ~1e-7, test_loss ~4e-7.
- M never grokked. Final M: train_acc 1.0000, test_acc 0.0614, train_loss ~3e-11, test_loss ~54.
- M's test_acc 6.14% is ~7× the 1/113 chance baseline — model is still slightly above random on test, but nowhere near learning the rule.
**What it means:** Textbook outcome. Confirms our pipeline reproduces Nanda's setup. Both endpoints exist: G is a fully grokked + cleaned-up model, M is a fully memorized model that never generalized. Phase 1 exit criterion met.

---

## Entry 2 — Fourier analysis of W_E (Phase 2.2)
**Date:** 2026-05-17
**Setup:** Discrete Fourier transform of `W_E` along the input (token) dimension. For each frequency k ∈ {0..56}, sum squared Fourier coefficients across all 128 neurons. Plot per-frequency power. Script: `taska/analysis/fourier.py`. Output: `taska/results/fig_fourier_WE.png`.
**What I expected:** G should show 5-6 sharp spikes at specific frequencies matching Nanda's reported {14, 35, 41, 42, 52}. M should show no sharp spikes — flat across all frequencies — because M doesn't represent inputs as positions on a circle.
**What happened:**
- G: median non-const power = 0.021, key frequencies (>4× median) = **{14, 15, 29, 31, 35, 41, 42, 43, 52}**. All 5 of Nanda's main frequencies recovered exactly. The extras (15, 29, 31, 43) are sub-threshold in Nanda's paper but show up here, probably seed-dependent.
- M: median non-const power = 9.322, no key frequencies (all bars in a narrow band of 6-14).
- M's total Fourier energy is ~4× larger than G's (more bars, each not too small), but it's smeared uniformly across all 56 frequencies instead of concentrated in 5.
**What it means:** G discovered the right "language" for modular addition (rotations on circles) and concentrated its embedding into 5 specific frequencies. M didn't discover it; its embedding has no preferred frequency. This is the cleanest possible demonstration of "G and M are structurally different objects, not noisier-vs-cleaner versions of each other." Pipeline matches Nanda's published Figure 3 exactly for G.

---

## Entry 3 — SVD comparison of W_E (Phase 2.3)
**Date:** 2026-05-17
**Setup:** SVD of `W_E` for both G and M. Report singular value spectra, effective rank (= exp(entropy of σ²)), stable rank (= ||W||_F² / σ_max²), and cumulative energy curves. Script: `taska/analysis/svd_compare.py`. Output: `taska/results/fig_svd_WE.png`.
**What I expected:** G's spectrum should have a sharp cliff after ~10-20 singular values, matching the Fourier finding (5 frequencies × cos+sin ≈ 10 directions). M's spectrum should decay smoothly with no cliff. Effective rank should be much smaller for G.
**What happened:**
| Metric | G | M |
|---|---|---|
| Effective rank | 11.5 | 59.2 |
| Stable rank | 9.0 | 26.4 |
| Top-k directions for 90% energy | **10** | **50** |
| Top-k directions for 99% energy | **12** | **85** |

- G's spectrum: dramatic cliff between index 11 (σ≈2) and index 12 (σ≈0.14). Drops 20× in one step.
- M's spectrum: smooth gradual decay from σ_max ≈ 4.5 to σ_min ≈ 0.08. No structural break.
- σ_max is comparable (G: 4.16, M: 4.51) — M doesn't have BIGGER directions, just MORE of them carrying meaningful weight.
**What it means:** Basis-free confirmation of the Fourier finding. M uses ~5× more directions than G to encode the same 113 tokens. Both rank metrics (effective and stable) agree: G is genuinely low-rank, M is high-rank. This is the analysis we'll reuse on ResNet (Track B) and Pythia (Track C) — no task-specific assumptions baked in.

---

## Entry 4 — Subspace overlap via principal angles (Phase 3.2)
**Date:** 2026-05-17
**Setup:** Compared M's top-k singular subspace to G's top-k singular subspace using principal angles and energy capture. Tested k ∈ {5, 11, 20, 30, 50, 80, 113}. Did both U (residual-stream space) and V (token space). Script: `taska/analysis/subspace.py`. Output: `taska/results/fig_subspace_WE.png`.
**What I expected:** Two extreme possibilities:
  - **Dream:** M's top-11 ≈ G's top-11 (all cos ≈ 1). Means M = G + extras, surgery is "subtract the extras."
  - **Orthogonal:** M's top-11 ⊥ G's top-11 (all cos ≈ 0). Means M lives in a totally different region of weight space, surgery is harder.
**What happened (residual-stream space, k=11):**
- First 5 principal cosines: **[0.91, 0.89, 0.87, 0.80, 0.74]**
- Remaining 6 cosines: essentially 0
- Energy capture: 47% (vs random baseline of 11/128 = 9% → ~5× random)
- Token space (V) at k=11: first 5 cos = [0.73, 0.63, 0.54, 0.48, 0.38]. Capture 17% vs random 9% → ~2× random.

**What it means:** Neither extreme. M and G's top-11 subspaces **share approximately 5 dimensions in common, and are orthogonal in the other 6**. This is somewhere between the dream and the worst case.

This was the right tool — an earlier per-vector intruder analysis (`intruder.py`) made it look like M was nearly orthogonal to G (no individual M-vector had cos > 0.5 to any G-vector). That was misleading because SVD's basis within a near-degenerate subspace is arbitrary; per-vector cosines can be small even when subspaces overlap. Principal angles fix this and reveal the real ~5-dim shared core.

**Implication for H2:** the original "subtract intruder dimensions" framing doesn't apply cleanly. The reframed experiment is "project M's weights onto G's column span and see if the projection generalizes." Three variants planned in `surgery.py` (next).

---

## Entry 5 — Per-example probe on residual stream (Phase 3.4 / H3 test)
**Date:** 2026-05-17
**Setup:** Linear probe (sklearn `LogisticRegression`) trained to predict `a`, `b`, and `(a+b) mod p` from the residual stream at position 2 (above the "=" token). Probed at three points in the forward pass: `resid_pre` (after embed+pos_embed), `resid_mid` (after attention), `resid_post` (after MLP). Selectivity baseline = same probe on shuffled labels. 80/20 train/test split, 113 classes per target, 3830 training-set examples total. Script: `taska/analysis/probe.py`.
**What I expected:** Two scenarios were live coming in:
  - **Scenario A (interesting):** M preserves `a` and `b` substantially better than G at `resid_post`, while both predict the sum well. Selectivity for sel(a), sel(b) is much larger in M than G.
  - **Scenario B (boring):** Both models look similar. M's extra weight-space directions don't carry per-example info; they're just noise that happens to produce 100% training accuracy.
**What happened:** Scenario A — but with sharper localization than I predicted.

| Layer | Model | predict_a | predict_b | predict_sum | shuffled |
|---|---|---|---|---|---|
| resid_pre  | G | 1.2% | 1.2% | 1.2% | 1.2% |
| resid_pre  | M | 1.2% | 1.2% | 1.2% | 1.2% |
| resid_mid  | G | 42% | 40% | 46% | 0.4% |
| resid_mid  | M | **96%** | **96%** | **2%** | 1.0% |
| resid_post | G | 43% | 41% | **100%** | 0.6% |
| resid_post | M | **88%** | **86%** | 93% | 0.6% |

**What it means:**
- At `resid_pre` (just inputs + position embeddings): both models are at chance for everything. The "=" position hasn't been mixed with the input positions yet — this is the embedding of the "=" token alone, which doesn't depend on `a` or `b`.
- At `resid_mid` (after attention): M has near-perfect knowledge of (a, b) and almost zero knowledge of (a+b). G has partial knowledge of all three. So attention routes the input embeddings to position 2 in BOTH models, but M routes them more aggressively/cleanly.
- At `resid_post` (after MLP): M still has 87% knowledge of inputs and 93% of sum. G has dropped to 42% on inputs but jumps to 100% on sum. **The MLP is where the compression happens in G — and where it doesn't happen in M.**
- Selectivity (probe − shuffled): all the real numbers are >> 0, confirming this is genuine signal, not probe capacity.

The gap at `resid_post` is 45 percentage points for sel(a) and 44 for sel(b). M is doing roughly 2× more input preservation than G is. Both models compute the sum (their job), but they do it via structurally different computations: G compresses (a,b) → (a+b) and discards; M co-computes (a+b) and keeps (a,b) around for lookup.

**Caveat: this is correlational, not causal.** The probe shows information *is present*; it doesn't show M *uses* that information to produce its output. A reviewer can reasonably say "you measured that M's activations have a signature, not that the signature is what's responsible for memorization." That requires the surgery experiment (Phase 3.3) — kill the input-preservation circuit, see if training accuracy collapses.

**On the user's worry "isn't this just lookup tables being lookup tables?":** partly fair. The qualitative direction (M preserves inputs more) is what any researcher would predict. What this experiment adds is *the metric and the gap size*. We now have a number — sel(a) at resid_post — that:
  - Tracks the cleanup phase during G's training (can be measured at every checkpoint)
  - Will be the evaluation target for surgery
  - Generalizes across tracks (you can probe ResNet activations the same way)

But the user is right that this alone doesn't carry the paper. The probe is a *tool* and a *diagnostic* — the paper's punch line still depends on H2 (surgery).

---

## Entry 6 — Spectral surgery on W_E (H2 test, layer 1)
**Date:** 2026-05-17
**Setup:** Three variants on M's embedding matrix `W_E`: (a) truncate to top-k SVD components of M, (b) project M onto G's top-k column/row spans, (c) substitute G's top-k singular subspace into M's. Swept k ∈ {1, 2, 3, 5, 8, 11, 15, 20, 30, 50, 80, 113}. Restored W_E between each evaluation. Script: `taska/analysis/surgery.py`.
**What I expected:** If H2 holds and M's memorization lives in extra W_E directions: at some k (probably around 11, G's effective rank), test accuracy jumps from M's 6% toward G's 100% with a clean train/test trade-off curve.
**What happened:** **FAILED across all variants.** Best test accuracy: 6.2% (basically baseline, no improvement). Truncate degraded both train and test smoothly; project destroyed M without producing G; substitute at k=1 retained 74% train accuracy but zero test recovery.
**What it means:** M's memorization is NOT localized in W_E. Either it lives in downstream layers (MLP, attention) or it's distributed. Need to test MLP next.

---

## Entry 7 — Spectral surgery on MLP weights (H2 test, MLP layer)
**Date:** 2026-05-17
**Setup:** Same three variants as Entry 6, but applied to W_in (512×128) and W_out (128×512) simultaneously. Same k for both matrices. The probe (Entry 5) suggested the MLP is where compression happens in G, so the MLP is the likely location of memorization in M. Script: `taska/analysis/surgery_mlp.py`.
**What I expected:** If the MLP is where the memorization lives, we should see test recovery here even if W_E surgery failed. Mild prediction since priors were already lowered by Entry 6.
**What happened:** **FAILED again.** Best test accuracy: 6.1%. All three variants flat at M's baseline. Substitute at k=1 retained 47% train but no test improvement.
**What it means:** Memorization is not localized to the MLP either. The "memorization is in one layer" framing is increasingly untenable. Try modifying everything simultaneously.

---

## Entry 8 — Combined spectral surgery (H2 test, all weight matrices together)
**Date:** 2026-05-17
**Setup:** Apply the three variants to W_E, W_in, W_out simultaneously at the same k. Most aggressive spectral intervention possible without touching attention. Script: `taska/analysis/surgery_combined.py`.
**What I expected:** Coordinated intervention across all matrices SHOULD work if the issue with single-layer surgery was "matrices were inconsistent with each other after modification." If even this fails, the spectral approach is dead in our setting.
**What happened:** **FAILED for the third time.** Best test accuracy: 5.8%. All variants flat. The pattern from Entries 6 and 7 repeats exactly: spectral interventions can destroy the model but cannot rebuild it toward G.
**What it means:** Strong negative result. M's memorization is encoded as a coordinated, joint, *non-spectral* property across the entire network. SVD-based projection cannot extract it. This directly contradicts the "intruder dimensions are the memorization circuit" framing in the LoRA literature (Shuttleworth et al.) — at least in our from-scratch training of a 1-layer transformer.

This is itself a publishable negative finding. The naive transfer of LoRA intruder-dimension intuition to from-scratch overfit training does not hold.

---

## Entry 9 — Linear mode connectivity (basin geometry test)
**Date:** 2026-05-17
**Setup:** Linear interpolation in weight space between M's final checkpoint and G's final checkpoint. 21 alpha values from 0 (pure M) to 1 (pure G). At each alpha, compute training loss / accuracy and test loss / accuracy. Script: `taska/analysis/mode_connectivity.py`.
**What I expected:** Strong prior on barrier (different weight decay should produce different basins per Frankle et al. 2020). Mainly running this as a sanity check to confirm that the spectral surgery failures are explained by basin geometry rather than some quirk of our implementation.
**What happened:**
| alpha | train_loss | train_acc | test_acc |
|---|---|---|---|
| 0.00 (pure M) | 3e-11 | 100% | 6% |
| 0.50 (midpoint) | **4.25** | 20% | 8% |
| 1.00 (pure G) | 1e-7 | 100% | 100% |

Barrier ratio (midpoint train_loss / max endpoint train_loss) = **4.28 × 10⁷**.

**What it means:**
- M and G are in genuinely different loss basins, separated by a high ridge.
- This explains why spectral surgery failed: any small perturbation to M pushes it OFF its low-loss peak into the valley between basins. To reach G, you'd have to cross the ridge — a non-local move that gradient-style or projection-style interventions cannot do.
- Interesting sub-finding: along the path from M to G (alphas 0.5 → 0.85), test accuracy *rises faster than train accuracy*. At alpha=0.60, test=22% while train=33%. This is grokking visible in weight space — as the model moves into G's basin, generalization emerges *before* perfect memorization is re-acquired.

**Honest caveat (and the user's sharper point):** linear mode connectivity barrier between two trained models is a textbook phenomenon (Frankle 2020, Entezari 2021, Ainsworth 2023). The result confirms M and G are different attractors but doesn't *explain* what makes M's basin "memorizing" and G's basin "generalizing." The user pointed out: both basins have low *train* loss; the qualitative difference is in *test* loss, which the training procedure can't see. So the deeper question that mode connectivity raises but doesn't answer is: **what structural property distinguishes a memorizing-type basin from a generalizing-type basin, and why does weight decay reliably select the latter?**

The mode connectivity finding is real but not novel on its own. Useful as scaffolding for the paper's narrative; not a headline result by itself.

---

## Entry 10 — [pending]

Next candidate experiments:
- Trajectory mode connectivity: trace the barrier between M_t and G_t at intermediate training epochs. When does it first appear?
- Activation ablation: kill the probe's `a`/`b` directions in M's residual stream; does M's lookup collapse?
- Git Re-Basin permutation alignment: does permuting M's neurons reduce the barrier to G?
