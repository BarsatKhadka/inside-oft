# so_far.md

What I've understood, decided, and done. Updated as I go.

---

## The question, in my own words

Take two copies of the same network. Train both on the same data. One I stop at the right time (it generalizes — call it **G**). The other I keep training past that point with no regularization (it overfits — call it **M**). Is M just G + noise, or is M a fundamentally different object?

If it's a different object, *what is the difference made of*, and can I find it, measure it, and surgically remove it?

## The four things I'm testing (plain English)

1. **Weight signature:** M's weights have specific structural patterns that G's don't.
2. **Surgical removal:** I can identify a small chunk of M's weights and cut it out, and that turns M back into G — without retraining.
3. **Per-example memory:** Inside M's activations, there's a readable signal telling you which training example it just saw. G doesn't have this signal.
4. **One mechanism:** 1, 2, 3 are the same underlying thing viewed from different angles. Intervening on one moves the others.

## Decisions locked in so far

- Target: TMLR submission.
- Approach: same-architecture, same-data, only-training-regime-different comparisons. Never compare across architectures.
- Tracks (in priority order):
  - **Track A:** modular addition (Nanda setup) — sanity check, known ground truth. *In progress.*
  - **Track B:** CIFAR-10 + ResNet-18 — the main experimental story.
  - **Track C:** Pythia fine-tuning — appendix / v2 if time permits.
- Learning policy: I write every experiment from scratch the first time I need it. No copy-paste from someone else's repo until I've already done the thing myself once at smaller scale.

## What I've actually done

### Phase 1 — Instrument working

- Wrote `taska/model.py`, `taska/data.py`, `taska/train.py` from scratch (matching Nanda's 1-layer transformer spec).
- Built YAML configs `G.yaml` and `M.yaml`. Only difference: weight_decay 1.0 vs 0.0. Same seed, same architecture, same num_epochs.
- Built SLURM submission (`taska/train.slurm`) for Magnolia HPC (Ole Miss MCSR, L40S GPUs, partition `gpuq`).
- Trained both G and M for 50,000 epochs on the cluster.
- Pulled checkpoints back via two-hop SCP (hpcwoods jump host blocks ProxyJump).

### Phase 2 — Analysis tools verified against Nanda's published results

- `taska/analysis/plot_history.py` — train/test loss + accuracy curves. **Result:** G grokked, M didn't. (See so_far_results.md entry 1.)
- `taska/analysis/fourier.py` — DFT of `W_E` per-frequency. **Result:** G has same 5 key frequencies as Nanda's paper. M has flat Fourier spectrum. (Entry 2.)
- `taska/analysis/svd_compare.py` — basis-free SVD analysis. **Result:** G's spectrum has a cliff at rank 11. M's smoothly spread. (Entry 3.)

Phase 2 = instrument verification, not novel science. Confirmed our pipeline can recover Nanda's results, so when it tells us something M-specific we can trust it.

### Phase 3 — The novel work (in progress, mostly negative)

- `taska/analysis/intruder.py` — per-vector cosine similarity. **Misleading** because basis within a subspace is arbitrary.
- `taska/analysis/subspace.py` — principal angles. **Real finding:** M and G's top-11 subspaces share ~5 directions and differ in ~6. (Entry 4.)
- `taska/analysis/probe.py` — linear probe on residual stream. **Finding:** M recovers `a`/`b` at ~88% probe accuracy vs G's ~43%. MLP is the compression site. (Entry 5.)
- `taska/analysis/surgery.py` — spectral surgery on `W_E`. **Failed**, no test recovery. (Entry 6.)
- `taska/analysis/surgery_mlp.py` — spectral surgery on `W_in + W_out`. **Failed**, no test recovery. (Entry 7.)
- `taska/analysis/surgery_combined.py` — spectral surgery on `W_E + W_in + W_out` jointly. **Failed**, no test recovery. (Entry 8.)
- `taska/analysis/mode_connectivity.py` — linear interpolation M -> G. **Barrier exists** (4.28 × 10⁷ loss ratio at midpoint), confirming different basins. (Entry 9.)

**The honest state of the project after one day of analysis:**

What we've established empirically:
1. M and G have different Fourier and SVD signatures (correlational, individually intuitive).
2. M's activations linearly encode the raw inputs `a, b` to a degree G's don't (correlational, partly intuitive).
3. M and G are in genuinely separate loss basins (textbook, but confirmed for our controlled M-vs-G setup).
4. Spectral surgery on any subset of weight matrices does NOT recover generalization from M (real negative result against the "intruder dimensions are the memorization circuit" framing from LoRA literature).

What we have NOT established:
- A causal intervention that converts M into G or vice versa.
- A mechanism distinguishing the *type* of basin (memorizing vs generalizing), not just the basin identity.
- Whether these findings generalize beyond Track A (modular addition).

**A reframing of the central question (motivated by user's observation):** the loss landscape contains at least TWO TYPES of low-train-loss basins — memorizing and generalizing — and weight decay is the lever that selects between them. The interesting question isn't "are M and G in different basins" (they are, trivially) but "what is the structural property of memorizing-type basins vs generalizing-type basins, and why does weight decay reliably select the latter?" This is a more productive framing for the paper.

**Pitch for the paper if we stop here:**
> "Overfit and generalizing networks trained from identical conditions (same init, data, architecture, optimizer) end up as structurally distinct attractors: different Fourier/SVD/probe signatures, different loss basins. The memorization-type attractor is not surgically connected to the generalizing-type attractor by any SVD-based weight intervention we tested. This contradicts the 'intruder dimensions are memorization' framing from prior LoRA work and suggests that, at least in from-scratch training of small transformers, memorization is encoded as a coordinated joint property of the full network rather than as a low-rank perturbation."

That's TMLR-acceptable but thin. To make it strong, we need ONE of:
- A working causal intervention (activation-level, or training-trajectory based)
- A timing claim (when during training do the basins separate?)
- Track B confirmation that the pattern persists at scale

Next experiments to consider:
- `mode_connectivity_trajectory.py` — trace barrier height through training epochs. When do M and G diverge into different basins?
- `activation_ablation.py` — use the probe's `a`/`b` directions; ablate them at inference; does M's train accuracy collapse?
- `permutation_alignment.py` — does Git Re-Basin permutation align M with G? If so, the "different basin" finding is a basis artifact.

## What I don't know yet but need to

- ~~How to compute an SVD of a weight matrix and read what it tells me.~~ Learned via svd_compare.py.
- ~~What "intruder dimensions" really look like when you plot them.~~ Learned via intruder.py and subspace.py; the per-vector framing was the wrong tool, principal angles is the right one.
- ~~What a linear probe is and why it's "linear" specifically.~~ Learned via probe.py — it's just logistic regression on activations; "linear" means the probe can't synthesize new structure, so what it recovers really is information the model put there.
- How a Hessian eigenvalue is actually computed in practice (Lanczos / power iteration / autograd-tricks). Needed for Phase 3.5.
- How to train a sparse autoencoder on activations.

I'll learn each of these by needing it for an experiment, not by reading about it in advance.

## Open questions I'm parking

- Whether to include grokking as a "mode" or just as a control. (Decided: just a control. Our M condition is "memorize without grokking ever" and we don't need a third "grokked-then-overtrained" condition.)
- Whether to start Track B from-scratch or from a pretrained checkpoint. Lock after Phase 3 of Track A is complete.
- Whether to include Track C. Honestly suggested I skip it for v1 — A+B with strong controls is enough for TMLR.
- The surgical H2 experiment as originally framed doesn't apply cleanly to our M because M is not a superset of G; reframed as projection experiment. Need to run that before committing to one specific variant.
