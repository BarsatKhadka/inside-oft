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

## Entry 18 — [pending: trajectory rescue]

If we take M at epoch t and continue training WITH weight decay from there, does it grok? At what t does this stop working — when is memorization "irreversible" in the optimization sense?
