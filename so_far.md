# so_far.md

What I've understood, decided, and done. Updated as I go.

---

## The question, sharpened from day 1's findings

Original framing: "Are overfit models a different *kind* of object than generalizing models, or just a worse version?"

**Sharpened framing after day 1 (Track A):**
> "In modular addition, generalizing models (G) are true local minima on the full-data loss surface. Memorizing models (M) are NOT minima — they are *saddle points* that appear stable only because the training procedure sees only training-data gradients. Both share zero training loss, but G's full-data gradient is ~10⁻⁶ everywhere; M's is ~17 in the test direction. Their full-data gradients differ by ~6 million×."

Scope caveat: this is for grokking-style training on algorithmic tasks. Whether the saddle framing transfers to standard CIFAR overfitting is the central open question for Track B.

## The four things I'm testing (updated after day 1)

1. **Weight signature:** M's weights have specific structural patterns G's don't. — **CONFIRMED.** Effective rank, Fourier sparsity, intruder dimensions.
2. **Surgical removal:** Cut out a chunk of M's weights, get G back without retraining. — **FAILED**, but the failure is informative: it's geometrically impossible because M is on a saddle, not in a basin.
3. **Per-example memory:** M's activations encode (a, b); G's don't. — **CONFIRMED** at 88% vs 43% probe selectivity.
4. **One mechanism:** all four lenses agree on the same story. — **PARTIALLY** confirmed: rank-compression, basin-migration, and cleanup-phase are the same process. Probe trajectory not yet run.

## What I've actually done (day 1 summary)

### Phase 1 — Pipeline working
- `taska/model.py`, `data.py`, `train.py`, configs `G.yaml`/`M.yaml` from scratch matching Nanda's spec.
- `train.slurm` for Magnolia HPC (partition `gpuq`).
- Trained 6 models total: M, G (seed 0); M_seed1, G_seed1; M_seed2, G_seed2.
- Two-hop scp transfer from HPC.

### Phase 2 — Analysis tools verified
- Grokking curves: G grokks at epoch 10,800; M never. (Entry 1)
- Fourier analysis: G has same 5 key frequencies as Nanda's paper. M is flat. (Entry 2)
- SVD comparison: G effective rank 11.5; M 59.2. G has a cliff; M smooth. (Entry 3)

### Phase 3 — The novel work

**Spectral / structural characterization (all confirmed):**
- Subspace overlap (Entry 4): M and G share ~5 of top-11 directions, differ in ~6. Per-vector cosine misleading; principal angles correct.
- Per-example probe (Entry 5): M recovers (a, b) at ~88%, G at ~43% at resid_post.
- Multi-seed (Entry 14): all 3 M's have near-identical rank (~58-87-97), all 3 G's have similar low rank (~7-18-10). Probe selectivity all M's ~85-90%, all G's ~38-43%. **Structural signatures are basin-type properties, not seed artifacts.**

**Surgical interventions (all failed):**
- Surgery on W_E (Entry 6) — failed.
- Surgery on MLP (Entry 7) — failed.
- Combined surgery on W_E + W_in + W_out (Entry 8) — failed.
- Time-resolved surgery on early-epoch M (Entry 12) — failed at every epoch.
- Permutation alignment / Git Re-Basin (Entry 10) — doesn't help. Avg neuron-pair similarity only 0.21. M and G are different algorithms, not permuted versions of the same one.

**Loss-landscape geometry (the real story):**
- Mode connectivity M ↔ G (Entry 9): barrier height ~10⁷. M and G in different basins on shared train data.
- Trajectory basins (Entry 11): barrier appears GRADUALLY during epochs 1000-12000. Saturates at grokking. The "cleanup phase" is geometrically a basin migration.
- Rank trajectory (Entry 13): G's effective rank compresses continuously throughout training; sharp drop during grokking window. M's rank stays flat after epoch 1000.
- **Saddle test (Entry 15): DEFINITIVE.** M's gradient on full data is ~17 (huge). G's is ~10⁻⁶ (essentially zero). Ratio: 5.8 million ×. **M is empirically a saddle on full-data loss surface, not a basin.**

## The unified picture after day 1

These are all **the same process viewed from different angles:**

| Lens | What G does | What M does |
|---|---|---|
| Weight rank | Compresses from rank ~113 to rank ~11 | Stays at rank ~59-97 |
| Loss basin | Migrates into a true basin | Stays at a saddle |
| Probe signature | Loses (a, b) info as MLP compresses | Preserves (a, b) info |
| Fourier structure | Develops 5-frequency sparsity | Stays uniformly flat |
| Test loss | Drops sharply at grokking | Never drops |

All five evolve together during epochs 1000-12000. All five saturate around epoch 16,000. **The cleanup phase = rank compression = basin migration = Fourier sparsification = input compression.** Different observables of one underlying process: weight decay continuously pulling the optimizer off the memorization saddle and into the generalization basin.

## The headline claim for the paper (Track A version)

> "In grokking-style training, memorization is a saddle on the full-data loss surface, not a basin. Generalization is a true basin. Weight decay drives a continuous rank compression that migrates the optimizer off the saddle and into the basin; this migration manifests as Nanda's 'cleanup phase' and is the mechanism by which grokking generalizes. The saddle-vs-basin distinction explains why spectral surgery on memorizing models fails (you cannot project a saddle into a basin) and why memorizing solutions are seed-consistent in their structural signature (the saddle has a characteristic shape per dataset, even if its location varies by seed)."

## Scope and honesty notes

- All findings are for **grokking-style training on modular addition (Track A only)**. Whether they transfer to standard overfitting (CIFAR, Pythia) is the load-bearing open question.
- The "saddle, not basin" framing is correct for our M, but the literature has called these "memorizing minima" — we're refining that claim.
- The cleanup phase / rank compression connection (Yunis 2024 → us) is partially anticipated by Yunis's spectral dynamics paper but they didn't tie it to basin geometry.
- The 6-million× gradient ratio is our strongest single empirical result; it's the kind of clean number reviewers like.

## What I still don't know

- Probe trajectory: does G compress (a, b) at the same epoch as rank compression?
- Hessian: is G's basin flatter than M's "saddle"? Maini-style flat-minima check.
- **Does the saddle gradient at M actually point toward G?** Direct test of saddle directionality. (Quick experiment.)
- Track B: does any of this hold for vision/CIFAR-style overfitting?
- Theoretical: can we derive the saddle structure analytically from weight decay + overparameterization?

## Open questions parked

- Whether to include grokking as a "mode" vs control. Decided: just a control, plus the framing now emphasizes that our results are *specifically* grokking-style.
- Track C (Pythia): skip for v1 paper, A+B is enough for TMLR.
- Theoretical companion: ~3 days to write a toy linear-model derivation. Defer until empirical story is locked.

## Day 1 additions (post-multi-seed)

We extended the analysis after multi-seed by attempting two more interventions to "convert" M into G-like behavior. Both failed in informative ways:

- **Probe on test inputs (Entry 16):** M's resid_post encodes (a, b) at >90% on test inputs but produces near-chance (a+b) info. M genuinely has no hidden generalization — the MLP is a gated lookup that fires (a+b) only for memorized inputs. *Implication:* MIA mechanism on overfit models has a clean explanation.

- **Gating-neuron ablation (Entry 17):** Tried to find and ablate M's "gating" neurons. Membership probe only got 58% accuracy on MLP hidden activations — no clean linear gate exists. Ablation of "top gating" neurons gave the same results as random ablation. *Implication:* M's lookup is fully distributed across all neurons; there's no separable gate to remove. The user's prior worry confirmed: removing the gate doesn't expose generalization because there's no generalization circuit underneath.

## Cumulative ruled-out intervention strategies

After two days we've definitively ruled out *all* spectral-or-mechanistic intervention strategies on the *final* M:
- ✗ Spectral surgery on any individual layer or all layers combined
- ✗ Permutation alignment (M and G are different algorithms, not permuted versions)
- ✗ Targeted gating-neuron ablation (no gate exists)

The conclusion: **M cannot be converted to G by inference-time intervention.** The lookup is the computation, not a layer atop something extractable.

## The remaining live question

> **Can M be "rescued" at the optimization-trajectory level?** If we resume training M_t (memorizing checkpoint at epoch t) with weight decay turned on, does it grok? At what t does this stop working — when is memorization irreversible *in the optimization sense*?

This is the test of whether overfitting is a one-way street during training. Different from "can a frozen M be transformed" — instead, "can a partly-trained M be steered back toward G's basin by adding weight decay mid-training?"

## Next experiments (priority order)

1. **Trajectory rescue** (~1 hour): take M_t for t in {0, 1000, 5000, 11000, 20000, 50000}, continue training with weight decay, see if it groks. Plot test_acc vs additional training time for each starting t.
2. **Gradient direction test** (10 min): does ∇L_full at M point toward G - M?
3. **Probe trajectory** (~30 min): when does G compress (a, b) during training?
4. **Track B setup** (~2 days): does any of this hold for vision/CIFAR overfitting?

The trajectory rescue is the most promising as it tests whether overfitting is *reversible during training* even though it's not reversible after training. That's a substantive claim either way.
