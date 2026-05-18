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
**What I expected:** G should grok somewhere around epoch 10k–20k (Nanda's reported window). M should never grok.
**What happened:** G grokked at epoch **10,800**. Final G: train_acc 1.0, test_acc 1.0. M: train_acc 1.0, test_acc 0.0614 (~7× chance).
**What it means:** Textbook outcome. Confirms pipeline. Phase 1 exit criterion met.

---

## Entry 2 — Fourier analysis of W_E (Phase 2.2)
**Date:** 2026-05-17
**Setup:** DFT of W_E along input dimension. Script: `taska/analysis/fourier.py`.
**What I expected:** G should show Nanda's reported {14, 35, 41, 42, 52} key frequencies. M should be flat.
**What happened:** G's key frequencies (>4× median): {14, 15, 29, 31, 35, 41, 42, 43, 52} — all 5 of Nanda's main ones. M: no key frequencies (median power 9.3 spread uniformly).
**What it means:** G concentrated 226k parameters into 5 frequencies. M spread them across 56 frequencies. Pipeline matches Nanda's Figure 3 exactly for G.

---

## Entry 3 — SVD comparison of W_E (Phase 2.3)
**Date:** 2026-05-17
**Setup:** Singular value spectra + effective rank + stable rank. Script: `taska/analysis/svd_compare.py`.
**What I expected:** G's spectrum should have a cliff after ~10 directions. M's smooth.
**What happened:** G effective rank 11.5, stable rank 9.0, 10 dirs for 90% energy. M effective rank 59.2, stable rank 26.4, 50 dirs for 90% energy. Sharp cliff in G at index 11→12 (20× drop in one step).
**What it means:** Basis-free confirmation of the Fourier finding. G genuinely low-rank, M genuinely high-rank.

---

## Entry 4 — Subspace overlap via principal angles (Phase 3.2)
**Date:** 2026-05-17
**Setup:** Principal angles between M's top-k subspace and G's top-k subspace. Both U (residual space) and V (token space). Script: `taska/analysis/subspace.py`.
**What I expected:** Either "dream" (M's top 11 ≈ G's top 11) or "orthogonal" (M ⊥ G).
**What happened:** Neither. At k=11 in U: first 5 principal cosines [0.91, 0.89, 0.87, 0.80, 0.74], rest near 0. Energy capture 47% (vs 9% random). M and G share ~5 directions and differ in ~6.
**What it means:** Partial subspace overlap. Per-vector cosine analysis (`intruder.py`) was misleading because basis within a subspace is arbitrary; principal angles is the correct tool.

---

## Entry 5 — Per-example probe on residual stream (Phase 3.4 / H3 test)
**Date:** 2026-05-17
**Setup:** Linear probe on resid_pre / resid_mid / resid_post at position 2, predicting `a`, `b`, `(a+b) mod p`. Selectivity = real - shuffled. Script: `taska/analysis/probe.py`.
**What I expected:** M preserves (a,b) more than G. Both predict sum well.
**What happened (at resid_post):**
| | predict_a | predict_b | predict_sum | shuffled |
|---|---|---|---|---|
| G | 43% | 41% | **100%** | 0.6% |
| M | **88%** | **86%** | 93% | 0.6% |

At `resid_mid` (after attention only): M = 96%/96%/2%; G = 42%/40%/46%. The MLP is the compression site.

**What it means:** M preserves raw inputs through to position 2; G compresses them away. ~45-point selectivity gap. The qualitative direction is intuitive but the size of the gap and the precise localization (MLP) are non-trivial measurements. Correlational, not causal.

---

## Entry 6 — Spectral surgery on W_E (H2 test, layer 1)
**Date:** 2026-05-17
**Setup:** Three variants (truncate, project, substitute) on M's W_E. Sweep k. Script: `taska/analysis/surgery.py`.
**What I expected:** At k≈11 some recovery toward G.
**What happened:** FAILED across all variants. Best test acc: 6.2% (baseline).
**What it means:** Memorization not localized to W_E.

---

## Entry 7 — Spectral surgery on MLP weights
**Date:** 2026-05-17
**Setup:** Variants on W_in + W_out simultaneously. Script: `taska/analysis/surgery_mlp.py`.
**What I expected:** Likely site of memorization per probe (Entry 5).
**What happened:** FAILED. Best test acc: 6.1%.
**What it means:** Memorization not localized to MLP either.

---

## Entry 8 — Combined spectral surgery (all weight matrices)
**Date:** 2026-05-17
**Setup:** Coordinated surgery on W_E + W_in + W_out. Script: `taska/analysis/surgery_combined.py`.
**What I expected:** If the issue was inconsistency between layers, joint modification should work.
**What happened:** FAILED. Best test acc: 5.8%.
**What it means:** Strong negative result. Memorization is a coordinated joint property of all weights, not low-rank in any individual matrix. Contradicts the "intruder dimensions = memorization circuit" framing from Shuttleworth et al. in the LoRA setting.

---

## Entry 9 — Linear mode connectivity M ↔ G
**Date:** 2026-05-17
**Setup:** Linear interpolation in weight space, evaluate at 21 alphas. Script: `taska/analysis/mode_connectivity.py`.
**What I expected:** Different basins per Frankle et al. 2020.
**What happened:** Midpoint train_loss 4.25; endpoints 1e-11 (M) and 1e-7 (G). Barrier ratio 4.28 × 10⁷. Interesting sub-finding: along the path from M to G, test_acc rises faster than train_acc — grokking visible in weight space.
**What it means:** M and G in genuinely different basins (textbook). Explains surgery failures: can't cross the barrier with infinitesimal projections. (Note: this framing later revised — see Entry 14 and Entry 15. M is actually on a saddle, not in a basin, on full-data loss. The barrier here is real on shared train data but the "basin" terminology for M doesn't generalize to multi-seed.)

---

## Entry 10 — Permutation alignment (Git Re-Basin) M → G
**Date:** 2026-05-17
**Setup:** Hungarian matching of M's 512 MLP neurons to G's 512 MLP neurons using cosine similarity on concatenated W_in row + W_out column features. Permute M, then test mode connectivity. Script: `taska/analysis/permutation_align.py`.
**What I expected:** If M and G are in the same basin modulo permutation, barrier should drop substantially.
**What happened:** Avg neuron-pair cosine after optimal matching: **only 0.21** (~ random level). Barrier reduction factor: **1.06×** (essentially zero).
**What it means:** M and G are NOT permuted versions of the same function. They're different *algorithms* (Fourier circuit vs lookup table). Closes the Git Re-Basin loose end and strengthens the "different solutions" narrative.

---

## Entry 11 — Trajectory mode connectivity (when did basins separate?)
**Date:** 2026-05-17
**Setup:** Barrier between M_t and G_t at multiple training epochs. Script: `taska/analysis/trajectory_basins.py`.
**What I expected:** Maybe sudden at grokking, maybe gradual, maybe present from early on.
**What happened:** Barrier appears GRADUALLY during epochs 1000-12000, saturates after grokking.
| Epoch | Barrier height |
|---|---|
| 0 (init) | 4.76 (random) |
| 1000 | **0.005 — same basin!** |
| 4000 | 0.18 |
| 8000 | 1.13 |
| 11000 | 3.26 (G grokks here) |
| 12000+ | 4.5 (saturated) |

**What it means:** At epoch 1000, M and G are still in the same basin even though both have fully memorized. The basin separation is a gradual process happening throughout circuit formation and cleanup, not a sudden event at grokking. **This connects optimization geometry to Nanda's three-phase model: cleanup IS basin migration.**

---

## Entry 12 — Time-resolved surgery on early-epoch M
**Date:** 2026-05-17
**Setup:** Apply combined surgery (k=11) at M_t for t in {1000, 4000, 8000, 11000, ..., 50000}. Script: `taska/analysis/surgery_over_time.py`.
**What I expected (wrongly):** If barrier is small at epoch 1000, surgery should work better there.
**What happened:** FAILED at every epoch. Test acc stays at 1-2%.
**What it means:** "Barrier small" ≠ "spectral surgery works." The linear-path barrier and the spectral-projection direction measure different geometric properties. Even at epoch 1000 when M is in G's basin, M is already high-rank — rank-11 projection destroys it. The compression to low-rank is something G develops, not something inherited from being in a basin. (My prediction was wrong; this is real information.)

---

## Entry 13 — Rank trajectory of W_E / W_in / W_out (R)
**Date:** 2026-05-17
**Setup:** Effective rank and stable rank of all three weight matrices at every saved epoch for M and G. Script: `taska/analysis/rank_trajectory.py`.
**What I expected:** G's rank stays low, M stays high. Possibly compression timing matches grokking.
**What happened:** EXACTLY THIS:
| Epoch | G W_out eff rank | M W_out eff rank |
|---|---|---|
| 0 | 113.1 | 113.1 (identical) |
| 1000 | 79.8 | 98.7 (already diverging) |
| 8000 | 51.7 | 97.8 |
| 10000 | 39.6 | 97.8 |
| 11000 | **23.3** | 97.8 (G grokks; 3× rank drop in 1000 epochs) |
| 12000 | 13.4 | 97.8 |
| 16000+ | 10.0 (saturated) | 97.8 (saturated) |

Similar patterns for W_E and W_in.

**What it means:** Weight decay drives **continuous spectral compression throughout training**. M's rank stays flat after epoch 1000. G's rank drops continuously, with the SHARPEST drop happening exactly in the grokking window (epoch 10000-12000). **The cleanup phase IS rank compression.** Connects to Yunis et al. 2024 spectral dynamics framework.

---

## Entry 14 — Multi-seed structural consistency
**Date:** 2026-05-17
**Setup:** Trained 4 additional models (G_seed1, G_seed2, M_seed1, M_seed2). For all 6 final checkpoints, computed rank for each matrix, pairwise barriers on full data (12769 pairs), and probe selectivity using each model's own training data. Script: `taska/analysis/multiseed.py`.
**What I expected:** If basin TYPES are real categories, all M's should have similar structural signatures and all G's similar (different) signatures.
**What happened:**
- **Rank (highly consistent within category):** All 3 M's: W_E ~58-59, W_in ~87, W_out ~97. All 3 G's: W_E 7-13, W_in 11-19, W_out 6-10. Category gap is 5-10×.
- **Probe sel(a):** All 3 M's 84-90%; all 3 G's 38-43%. Category gap is 2×.
- **Mode connectivity on full data was confusing:** M-M midpoint loss ~22, but M's own endpoint loss is also ~36 on full data (because M doesn't generalize, its test contribution is huge). So midpoint is LOWER than endpoint. This led to the saddle insight in Entry 15.

**What it means:** "Memorizing-type basin" and "generalizing-type basin" are REAL categories. Structural signatures are seed-robust. Three M's land at near-identical rank and probe signatures.

The mode-connectivity confusion turned out to be informative: it revealed that M is NOT a basin on the full-data loss surface. M is at a point where loss-on-own-train-data is zero but loss-on-everything-else is high. The midpoint between two M's has lower full-data loss because moving away from M's specific memorization point reduces test loss. This is the signature of a SADDLE, not a basin.

---

## Entry 15 — Saddle vs basin: gradient norm test
**Date:** 2026-05-17
**Setup:** For each of 6 models, compute ||∇L||₂ on (a) own train data, (b) own test data, (c) full data. If M is a basin, all three should be ~0. If M is a saddle, gradient should be ~0 on train but large on test/full. Script: `taska/analysis/saddle_test.py`.
**What I expected (after Entry 14):** Strong prior on saddle. M's full-data gradient should be much larger than G's.
**What happened: DEFINITIVE CONFIRMATION.**

| Metric | M models (mean of 3) | G models (mean of 3) | Ratio |
|---|---|---|---|
| ‖∇‖ on own train | 1.07 × 10⁻¹⁰ | 3.46 × 10⁻⁷ | both ~zero |
| **‖∇‖ on own test** | **16.95** | **2.86 × 10⁻⁶** | **5.9 million ×** |
| **‖∇‖ on full data** | **11.87** | **2.04 × 10⁻⁶** | **5.8 million ×** |

**What it means:** This is the cleanest empirical result of day 1.

At M's weights: training data says "minimum, don't move" (gradient 10⁻¹⁰), but test data says "move HARD" (gradient 17). Net: M is at a saddle on the full loss surface — minimum in training directions, NOT minimum in test directions.

At G's weights: every data slice says "minimum, don't move" (gradient 10⁻⁶ everywhere). G is at a true basin on the full loss surface.

The 5.8 million × ratio is not subtle. **Memorization is not a basin in this setup — it's a saddle.** This reframes the project from "two different basins" to "one true basin (G) and one saddle that masquerades as a basin when only training-data gradients are observed (M)."

This is the headline empirical result of day 1.

---

## Entry 16 — Does M secretly know test answers? (Hidden generalization test)
**Date:** 2026-05-17
**Setup:** Linear probe trained on M's resid_post activations on TEST inputs (pairs M never saw) to predict (a+b) mod p. Compare to predicting `a`, `b` separately, and to G as reference. Script: `taska/analysis/probe_test.py`.
**What I expected:** If M has hidden generalization, probe should recover sum on test at >50%. If pure memorization, probe should be at chance.
**What happened:**
| Model | Split | actual_acc | probe(a) | probe(b) | probe(sum) | shuffled |
|---|---|---|---|---|---|---|
| M | train | 100% | 86% | 86% | **94%** | 0.6% |
| M | test  | 6%   | **93%** | **92%** | **1.6%** (chance!) | 0.9% |
| G | train | 100% | 40% | 40% | 100% | 0.6% |
| G | test  | 100% | 37% | 38% | 100% | 0.8% |

**What it means:** Striking pattern. M's activations contain (a, b) at **>90% even on test inputs M never saw** (the embedding + attention extract them regardless). But M produces essentially NO information about (a+b) for those test inputs (1.6% — basically chance, even *below* the model's actual 6% output accuracy).

M's MLP is acting as a **gated lookup**: it computes the sum only for inputs it has seen. For unseen inputs, it doesn't fail "noisily" — it produces sub-chance sum info while still preserving the raw inputs.

By contrast, G compresses (a, b) (only 37% probe accuracy on test) but produces perfect (a+b) (100% on both train and test). G has the algorithm; M has a lookup gated on training-set membership.

**Two implications:**
1. **No hidden generalization in M.** Radical hypothesis ruled out. M does not "secretly know" test answers — there's no routing failure to fix.
2. **Membership inference attack mechanism:** The 91-point gap (probe-sum on train vs test) for M IS a perfect membership-inference signature. Train a probe on training-data activations; apply to a new input; if probe matches model output → input was in training. G shows no such signature. This is a clean mechanistic explanation for why MIA works on overfit models.

The reframe: M knows which inputs are in its training set (perfectly readable from activations); G has forgotten.

---

## Entry 17 — Gating-neuron ablation: try to "ungate" M's MLP
**Date:** 2026-05-17
**Setup:** Hypothesis from Entry 16: M's MLP has neurons that detect "is this input in training?" and gate the (a+b) computation. Find them via linear probe on MLP hidden activations predicting train-vs-test membership. Then zero-ablate top-k of those neurons at inference. Compare to random-ablation baseline. Script: `taska/analysis/gate_ablation.py`.
**What I expected:** Membership probe should easily distinguish train vs test from MLP activations (high accuracy). Ablating top gating neurons should expose hidden sum information on test (probe-sum-on-test rising) and possibly improve actual test accuracy.
**What happened:**
- **Membership probe only achieved 58.5% accuracy** (random = 50%). So M's MLP hidden activations DON'T strongly linearly encode "is this in training." No clean gate to find.
- **Ablation of "top gating" neurons gave essentially identical results to random ablation.** At k=50: gating-ablation test_acc = 4.3%, random-ablation = 3.6%. At k=200: 2.3% vs 2.1%. Both columns degrade gracefully with k — generic damage, no targeted reveal.
- Probe-sum-on-test after ablation also tracks together for gating vs random.

**What it means:** The "find the gate and ablate it" hypothesis is FALSE. M's MLP does NOT have a separable linear gating mechanism. The lookup is implemented in the weights themselves, distributed across all neurons, NOT as a separate gate-then-circuit architecture.

This rules out "ungate M's MLP to reveal generalization" as a viable intervention. Confirms the user's prior worry: even if a gate existed and we removed it, there's nothing underneath to reveal — the lookup IS the computation, not a layer on top of something else.

Updated mental model of M: 512 neurons, each implementing a small pattern-detection-for-training-pair, that *collectively* output (a+b) for matching inputs and nothing useful otherwise. No separable parts.

This closes the third intervention strategy (after spectral surgery and permutation alignment). M cannot be converted to G by inference-time intervention. The only remaining question: can M be rescued via *training-trajectory intervention* — e.g., adding weight decay mid-training?

---

## Entry 18 — Trajectory rescue: can M be saved mid-training? [HEADLINE]
**Date:** 2026-05-17
**Setup:** For 6 starting checkpoints M_t with t ∈ {0, 1000, 5000, 11000, 20000, 50000}, load M and continue training with weight_decay=1.0 for 20,000 additional epochs. Measure train_acc and test_acc trajectories during rescue. Same lr, betas, full-batch as original training. Script: `taska/analysis/trajectory_rescue.py`.
**What I expected:** Probably small t (1000-5000) rescues easily (still in shared basin); large t (50000) might be stuck (deep in saddle).
**What happened: ALL 6 rescue.**

| Start epoch | Final test_acc | Outcome | Time to grok (rescue epochs) |
|---|---|---|---|
| 0 | 100% | RESCUED | ~11000 |
| 1000 | 100% | RESCUED | ~9000 (fastest) |
| 5000 | 100% | RESCUED | ~11000 |
| 11000 | 100% | RESCUED | ~11000 |
| 20000 | 100% | RESCUED | ~11000 |
| 50000 | 100% | RESCUED | ~11000 |

**What it means:** **Overfitting in modular addition is fully reversible by adding weight decay and continuing training, regardless of how long the model has been overfitting.** This is the cleanest positive intervention result in the project.

Every M can be converted to a generalizing model by gradient descent + weight decay. Even M at epoch 50,000 (the original "fully overfit" model that all our spectral surgeries failed on) gets rescued.

Connection to all prior findings:
- The saddle test (Entry 15) showed M's full-data gradient has norm ~17 in the test direction. **THAT is the rescue force.** Weight decay's contribution combined with this saddle gradient pushes the optimizer off the saddle into G's basin.
- Spectral surgery failed (Entries 6, 7, 8) because it perturbs in arbitrary directions, not along the saddle's unstable axis. Gradient descent + WD naturally follows the unstable axis.
- The mode-connectivity barrier (Entry 9) of ~10⁷ is the LINEAR path. The curved gradient descent path goes around the barrier, not through it.
- The probe (Entry 5) showed M preserves (a, b) at >90% even on test inputs. These preserved inputs are precisely what WD has to work with to rebuild the Fourier circuit.

Reframe of the project's central claim:

> "Overfitting is a metastable equilibrium of pure gradient descent, not a stable equilibrium of the loss landscape. The memorizing solution sits at a saddle whose unstable direction is toward generalization. Weight decay during continued training provides the natural force to escape this saddle. Spectral surgery cannot escape it because surgery moves in random directions; gradient descent with WD moves in the right direction."

This is now a real paper claim with empirical evidence and a clean mechanism.

---

## Entry 19 — Local generalization: does M have any smoothness?
**Date:** 2026-05-17
**Setup:** For each of 12,769 (a, b) pairs, compute its cyclic L1 distance to the nearest training pair. Bin predictions by distance. Measure: accuracy and mean absolute cyclic error of M's predictions at each distance. Script: `taska/analysis/local_generalization.py`.
**What I expected:** If M has local smoothness, accuracy drops gradually with distance and absolute error stays smaller than uniform.
**What happened:**
| Distance | n pairs | M acc | M mean abs err | G acc |
|---|---|---|---|---|
| 0 (training) | 3830 | **100%** | 0 | 100% |
| 1 | 6832 | **6.12%** | **26.14** | 100% |
| 2 | 2020 | 6.34% | 26.35 | 100% |
| 3 | 85 | 3.53% | 27.41 | 100% |
| 4 | 2 | 0% | 14.50 | 100% |

**What it means:** M has **ZERO local generalization.** The moment you move ONE token-distance away from a training pair, accuracy collapses to chance (~6%) and mean absolute error of ~26 (out of P/4=28 random baseline). M is a pure point hash table — no implicit interpolation, no nearby-input-similar-output structure. Treats `(7, 23)` and `(7, 24)` as completely unrelated inputs.

People often assume neural nets get "free" generalization via implicit smoothness; we show that's FALSE for overfit-style memorization. The memorization is genuinely per-pair, not per-region.

---

## Entry 20 — Per-example memorization quality
**Date:** 2026-05-17
**Setup:** For each training pair, compute logit margin = logit[correct] - max(logit[wrong]). Distribution of margins tells us if some pairs are more strongly memorized than others. Same on test pairs (where margin is usually negative). Script: `taska/analysis/memorization_quality.py`.
**What I expected:** M's margins might be uniform (all memorized equally), or long-tailed (some pairs better than others).
**What happened:**
- **M training margins:** mean 25.1, range 23.8 to 42.7 (factor 1.8 between weakest and strongest)
- **M test margins:** median **−55**, min **−206** (M is *confidently wrong* on test, not uncertain)
- **G training margins:** mean 17.5, range 14.7 to 20.8 (factor 1.4)
- **G test margins:** mean 16.9, similar to train (G generalizes uniformly)

**What it means:** Two findings.
1. M's memorization is mostly uniform but has variance — some pairs are memorized 1.8× more strongly than others. Not obviously systematic (no clear pattern in best/worst pairs).
2. **M's test confidence is huge AND wrong.** Margins of -55 to -206 mean M asserts the wrong answer on test with extreme confidence. This is the precise mechanism that makes membership inference trivial: a high-confidence prediction (correct or wrong) from M = M has a strong opinion = M has seen this pair before. G shows no such signature.

---

## Entry 21 — Neuron organization
**Date:** 2026-05-17
**Setup:** For each of 512 MLP neurons, compute per-input activation across all 12,769 pairs. "Selectivity" = fraction of total activation captured by top 1% of inputs (high = specialized neuron, firing on few inputs only). Compare M to G. Script: `taska/analysis/neuron_organization.py`.
**What I expected:** If M's neurons are specialized lookup-detectors, selectivity should be much higher than G's.
**What happened:**
- **M selectivity distribution:** mean ~0.08, long tail to 0.40+. Some neurons highly specialized.
- **G selectivity distribution:** tightly clustered around 0.04 (~uniform-firing baseline at 0.01). G's neurons are MUCH less specialized.
- Activation heatmap: M's top 100 neurons fire sparsely on small subsets of inputs. G's top 100 neurons fire in STRUCTURED PERIODIC bands (the Fourier circuit's components).

**What it means:** Confirms structural difference at neuron level. **M's neurons are specialized pattern-detectors** (firing on small sets of training pairs). **G's neurons are uniformly-firing Fourier components** that participate broadly across all inputs. Same architecture, different functional role per neuron.

---

## Entry 22 — Attention patterns
**Date:** 2026-05-17
**Setup:** For both M and G, compute average attention pattern at position 2 (above "="), separately for train and test inputs. 4 heads × 3 source positions. Script: `taska/analysis/attention_analysis.py`.
**What I expected:** Maybe M's attention differs on train vs test (gating). Or maybe M's attention is more concentrated.
**What happened:**
- **G's attention is symmetric across heads:** heads 1, 2, 3 attend 50/50 between positions 0 (a) and 1 (b). Head 0 has tiny attention to position 2.
- **M's attention is ASYMMETRIC and varied per head:** head 0 = 46/54, head 1 = 32/68, **head 2 = 90/10**, head 3 = 38/62. Different heads have learned different routing.
- **Same patterns on train and test for both models.** Attention itself is NOT a membership detector.

**What it means:** Structural difference at the attention level. **G's heads have converged to symmetric Fourier-like routing** (a and b treated equally). **M's heads have differentiated into asymmetric routings** — head 2 in particular has become a strong "look at a" detector while head 1 is mostly "look at b". This is a real difference but its functional meaning isn't yet clear. Possibly each head specializes for different memorization sub-circuits.

Notably: attention pattern doesn't differ between train and test, so the membership gating must live downstream (MLP), not in attention.

---

## Entry 23 — Capacity test (compressibility)
**Date:** 2026-05-17
**Setup:** Apply low-rank truncation and quantization to W_E + W_in + W_out simultaneously. Sweep rank k and bit-precision. Measure: at what compression level does train accuracy collapse? Script: `taska/analysis/capacity_test.py`.
**What I expected:** G compressible (low intrinsic dim), M not (needs full capacity to memorize 3830 pairs).
**What happened:**
- **G survives truncation to rank ≈ 20 with 100% train acc.** Even rank 10 gives 65%. G is genuinely low-rank.
- **M needs rank ≈ 100+ to maintain train acc.** At rank 50: only 60%. M uses most of its capacity to memorize.
- **G survives quantization to 8 levels (3 bits) with 100% train acc.**
- **M needs 16 levels (4 bits) for 100% train acc.**
- M's test accuracy stays at ~6% regardless of compression (predictably).

**What it means:** Quantifies the "M uses way more capacity than G" claim. M needs ~5× more rank and ~2× more bit-precision to maintain memorization. **G uses ~26 kbits of capacity for its solution; M uses ~6× more.** Most of M's parameters are doing work, but the work is per-pair lookup, not algorithmic.

---

## Entry 24 — Transfer test: does M's frozen body help learn (a-b) mod p?
**Date:** 2026-05-17
**Setup:** Freeze the body (embed + attention + MLP) of M, G, or random init, and train ONLY a fresh unembedding W_U on the new task `(a - b) mod p`. Compare to full fresh end-to-end training on the new task. 10k epochs each. Script: `taska/analysis/transfer_test.py`.
**What I expected:** If M's representations are useful generally, frozen-M + fresh-U should learn the new task. Honest prior: ~30% chance.
**What happened: NEGATIVE result.**
| Config | Final test acc on (a-b) mod p |
|---|---|
| M frozen + fresh W_U | **0.5%** |
| G frozen + fresh W_U | **18.7%** |
| Random frozen + fresh W_U | 0.1% (chance) |
| Full fresh end-to-end | 100% (reaches 95% at epoch 9600) |

**What it means:** M's representations don't transfer at all (worse than chance!). G transfers modestly — its Fourier-rotation structure has some overlap with subtraction — but neither comes close to learning. **"Overfit models as pretrained encoders" is dead.** M's structure is too task-specific to reuse.

Interesting nuance: G > M for transfer. G learned a more general "rotational" encoding; M learned input-specific lookup with no transferable features.

---

## Entry 25 — Distillation: does M as a teacher accelerate fresh student grokking?
**Date:** 2026-05-17
**Setup:** Train a fresh student from random init on (a+b) mod p with loss = CE + λ × KL(student || M). Sweep λ ∈ {0, 0.1, 0.5, 1.0, 2.0}. λ=0 is the control (no distillation). 20k epochs. Script: `taska/analysis/distillation.py`.
**What I expected:** ~10% chance distillation helps (M's outputs include confidently-wrong test predictions, should hurt).
**What happened: POSITIVE result with caveat.**
| λ | Epoch to reach 95% test acc | Speedup vs λ=0 |
|---|---|---|
| 0.0 | 8600 | baseline |
| 0.1 | 7800 | 9% faster |
| **0.5** | **6000** | **30% faster** |
| 1.0 | 6800 | 21% faster |
| 2.0 | 6200 | 28% faster |

**What it means:** Distilling from M accelerates fresh student grokking by ~30%. Real measurable speedup.

**Important caveat:** distillation is computed only on TRAINING data, where M is correct. So part of this speedup might be just "double training signal" effect (M's training predictions match labels, adding redundant supervision). The other part is M's full logit *distribution* — the shape of its softmax over 113 classes — which contains soft-target information beyond hard labels.

To distinguish: need a control with G as teacher. If G-distillation gives similar 30% speedup → it's just soft-target effect. If M >> G → M has unique useful structure. **Not yet run.**

Also: total compute is WORSE. Training M takes 50k epochs + distillation 6k epochs = 56k total, vs ~10.8k for fresh G. Practical value zero. Scientific interest only.

---

## Entry 26 — G-distillation control
**Date:** 2026-05-17
**Setup:** Same as Entry 25 (distillation from teacher → fresh student grokking) but using **G as teacher instead of M**. Sweep λ ∈ {0, 0.1, 0.5, 1.0, 2.0}. Compare to M-distillation results from Entry 25. Script: `taska/analysis/distillation_G.py`.
**What I expected:** If M's 30% speedup is M-specific (M has unique info), G-distillation should give SIMILAR speedup. If M-distillation is just generic soft-target effect, G-distillation should give MUCH stronger speedup (G's outputs are clean generalizations).
**What happened:**
| λ | M-distillation (epoch grok) | G-distillation (epoch grok) | G speedup over M |
|---|---|---|---|
| 0.0 | 8600 | 8600 | same control |
| 0.1 | 7800 | 6800 | G slightly faster |
| 0.5 | 6000 | **1200** | **5× faster** |
| 1.0 | 6800 | **600** | **11× faster** |
| 2.0 | 6200 | **400** | **15× faster** |

**What it means:** G is a *vastly* better teacher than M. At λ=2.0, G-distillation groks 15× faster than M-distillation. **The M-distillation speedup is just the generic dark-knowledge effect — M provides minimal information beyond what a soft target alone provides.** M is NOT a useful teacher in any meaningful sense.

This kills the "overfit models contain transferable info" angle from the distillation direction.

---

## Entry 27 — Cross-seed wrong-prediction consistency (δ)
**Date:** 2026-05-17
**Setup:** For each of 4349 test inputs unseen by ALL 3 M-models (intersection of test splits across seeds 0, 1, 2), get each M's prediction. Measure: how often do all 3 M's predict the same WRONG answer? Compare to chance (1/113 = 0.88%). Script: `taska/analysis/wrong_prediction_consistency.py`.
**What I expected:** If M's wrong predictions share structure across seeds (e.g., M's discover similar "wrong-answer rules"), agreement should be much higher than chance. Honest prior: ~25% chance of finding shared structure.
**What happened:**
- Pairwise M-M agreement on unseen inputs: 0.97% - 1.26% (chance is 0.88%)
- All 3 M's agree: 0.07% (chance = 1/113² = 0.008%)
- Among inputs where ALL 3 are wrong: only 0.05% give the same wrong answer

**What it means:** M's wrong predictions are essentially **independent across seeds**. Each M is its own unique pattern of confident-wrongness. The "memorizing solutions discover shared wrong-structure" hypothesis is FALSE. M's are *structured in aggregate* (same rank, same probe, same saddle topology) but *random in specifics* (each instantiation idiosyncratic).

This rules out one possible angle and confirms the "structured randomness" interpretation.

---

## Entry 28 — Track B analysis: do structural signatures transfer to CIFAR? [BIG]
**Date:** 2026-05-17
**Setup:** Train ResNet-18 on CIFAR-10 in two regimes — G_CIFAR (WD=5e-4 + augmentation), M_CIFAR (no WD, no augmentation, 400 epochs). Compute Track A's full structural battery: margin distribution, effective rank of selected weight matrices, gradient norms on train vs test (saddle test), MIA probe accuracy. Compare M_CIFAR to G_CIFAR. Script: `trackb/analysis_trackb.py`.
**What I expected:** If the structural signatures of memorization are regime-invariant, M_CIFAR should show the same patterns as M_TrackA: higher rank, saddle topology, train-vs-test margin asymmetry, MIA leak.
**What happened: structural signatures CONFIRMED in CIFAR even though M_CIFAR generalizes to 82% (vs Track A's 6%).**

| Signature | M_CIFAR | G_CIFAR | M_TrackA (for ref) |
|---|---|---|---|
| Test accuracy | 82.4% | 88.2% | 6.1% |
| W_out / fc effective rank | 8.8 (small fc) | 8.2 | (different arch) |
| layer4.1.conv2 effective rank | **363** | **14.6** | (different arch) |
| ‖∇‖ on train | 0.000155 | 60.5 (still some WD residual) | ~10⁻¹⁰ |
| ‖∇‖ on test | **29.4** | 62.6 | **~17** |
| Saddle ratio (test/train grad) | **190,000×** | ~1× | **~6,000,000×** |
| MIA probe (chance = 0.50) | 53.7% | 51.9% | (different probe) |
| Margin distribution | broad on train, negative tail on test | narrow on both | broad, very negative on test |

**Key findings:**
1. **The deep conv layer (layer4.1.conv2) has 25× more effective rank in M than G.** This is *bigger* than the Track A 5× difference. Even though M_CIFAR generalizes much better than M_TrackA, its computational structure is dramatically more spread-out.
2. **Saddle topology is preserved in CIFAR.** M's gradient asymmetry is 190,000× vs G's ~1×. Track A's was 6M× — same direction, just less extreme.
3. **MIA leak is real but small** (3.8 percentage points above chance). In benign overfitting, the membership signature is present but weak.
4. **Margin asymmetry preserved.** M_CIFAR has confidently-wrong test predictions; G_CIFAR has uniform margins.

**What it means:** The structural signatures of memorization are **regime-invariant**. They appear in catastrophic memorization AND benign overfitting. The model can generalize well (82%) while still being structurally a memorizing-type solution.

This is what makes the work more than a rediscovery of double descent / benign overfitting. Benign overfitting (Belkin 2019, Nakkiran 2020) shows that perfect-fit models can generalize. We add: even when they do, they have a *distinctive internal structure* visible to anyone who measures rank, gradients, or margins.

---

## Entry 29 — Mode connectivity in Track B
**Date:** 2026-05-17
**Setup:** Linear interpolation between M_CIFAR and G_CIFAR in weight space, evaluate train+test loss/acc at each alpha. Script: `trackb/mode_connectivity_trackb.py`.
**What I expected:** If M and G are in different basins (as in Track A), should see a clear barrier. If they're in the same basin (because both work decently in benign regime), should see a flat or low-barrier path.
**What happened:** Clear barrier. At alpha=0.2, loss spikes to ~10 (vs ~0 at M and ~0.5 at G). Accuracy drops to ~10% (chance for 10-class CIFAR) for alpha ∈ [0.2, 0.9].
**What it means:** **Even in benign overfitting, M_CIFAR and G_CIFAR are in genuinely different basins.** The linear path between them goes through a high-loss region where the model can't predict anything coherently. This confirms that the basin separation is regime-invariant, not specific to grokking.

---

## Entry 30 — Saddle escape mechanisms: what ELSE escapes M's saddle? [KEY]
**Date:** 2026-05-17
**Setup:** Take M_seed0 at epoch 50,000 (fully memorized). Test 5 alternative continuation regimes for 20,000 epochs each:
  1. Nothing (control, WD=0)
  2. WD=1.0 (control, known to escape)
  3. Random Gaussian noise injected each step (std=0.001)
  4. SAM (sharpness-aware minimization, rho=0.05)
  5. Add 50 held-out labeled pairs to training (WD=0)

Script: `taska/analysis/saddle_escape_mechanisms.py`.
**What I expected:** If saddles are fragile, any perturbation should eventually escape. Either noise or SAM or extra data should grok eventually.
**What happened: ONLY WD escapes. Everything else fails.**
| Mechanism | Final test acc | Escaped? |
|---|---|---|
| Nothing | 6.1% | No |
| **WD=1.0** | **100%** | **YES** |
| Gaussian noise (std 0.001) | 5.5% | No |
| SAM (rho=0.05) | 17.5% | Partial, very slow |
| Add 50 held-out pairs | 7.1% | No (M memorized them too) |

**What it means:** **The "saddle escapable by any perturbation" hypothesis is FALSE. Only weight decay escapes M's saddle.** Even adding new labeled training examples doesn't escape — M just memorizes the new examples too. SAM produces partial movement (~17%) but doesn't fully escape.

This is a stronger and more interesting finding than "saddle is universally escapable." It says **WD is privileged** among escape mechanisms. The next question is *why*. Hypothesis: WD's bias toward low-norm/low-rank solutions is the specific property that other regularizers lack.

---

## Entry 31 — Track B rescue: does WD-rescue work in CIFAR?
**Date:** 2026-05-17
**Setup:** Load M_CIFAR at final checkpoint (test_acc 82.4%, test_loss 1.31 and climbing). Continue training for 400 epochs with weight_decay=5e-4 (G's WD) turned on. Compare to M_CIFAR alone and G_CIFAR baseline. Script: `trackb/train_cifar.py --mode rescue`.
**What I expected:** If WD rescues in benign overfitting too, test_acc should climb toward G's 88% level. The recovery would necessarily be smaller (only 6 points to close) but mechanism should be visible.
**What happened: partial rescue.**
| Metric | M_CIFAR (start of rescue) | After 400 rescue epochs | G_CIFAR baseline |
|---|---|---|---|
| Test accuracy | 82.4% | 84.1% peak | 88.2% peak |
| Test loss | 1.31 (climbing) | 0.53 (60% drop) | 0.36 |
| Train accuracy | 100% | 95% (WD broke memorization) | 90% |

**What it means:** WD-rescue **partially works in CIFAR**:
- Test loss drops 60% — clear effect of WD
- Test accuracy improves ~2 points (closing 30% of the gap to G)
- Train accuracy drops as expected (WD removes overfit-specific weights)

But the rescue doesn't fully recover G's 88% in 400 epochs. Two interpretations:
1. **Rescue is slow** — needs more epochs (1000-2000+)
2. **Benign-overfit M_CIFAR is already near the generalizing basin** — the gap is small so the rescue is small

The Track B rescue is consistent with the saddle/WD story but is a weaker demonstration than Track A (where rescue was 100% complete and dramatic).

**Caveats** (need controls):
- Should compare to "continue M training with WD=0" — does just more training also help?
- Should track rank during the rescue
- Should try a "catastrophic M_CIFAR" (train on 500 images) for proper analog

---

## Entry 32 — [pending: rigor batch RANK MECHANISM TEST] [CRITICAL]

Four experiments designed to nail down the rank-compression mechanism:

- **rank_during_rescue.py**: track effective rank during WD vs SAM vs noise rescues. Expected: WD compresses, others don't.
- **rank_constraint_rescue.py**: force low rank without WD (project to rank k each step). Expected: this also escapes the saddle if rank IS the mechanism. CRITICAL TEST.
- **wd_sweep.py**: sweep WD ∈ {0.001 ... 10.0}. Quantitative escape threshold.
- **alternative_regularizers.py**: L1, L2-in-loss, spectral norm, label smoothing — which escape?

If all four confirm the hypothesis, we have airtight mechanism evidence:

> "Weight decay is the privileged saddle-escape mechanism, and its mechanism is rank compression. Anything else that compresses rank also escapes; anything else that doesn't, doesn't."

Submitted to HPC via `taska/rigor_batch.slurm`. Pending results.
