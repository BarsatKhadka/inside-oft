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

### Phase 3 — The novel work (started)

- `taska/analysis/intruder.py` — per-vector cosine similarity between M's top singular vectors and G's. **Result was misleading** because per-vector comparison is sensitive to basis rotation within a subspace.
- `taska/analysis/subspace.py` — principal angles between M's top-k subspace and G's top-k subspace. **The real finding:** they share ~5 directions in common and are orthogonal in the remaining ~6. Neither identical (dream) nor disjoint (worst-case). (Entry 4.)
- `taska/analysis/probe.py` — linear probe on residual stream at position 2 (above "="). Predicts `a`, `b`, and `(a+b) mod p`. **The finding:** M recovers `a` and `b` at ~88% probe accuracy vs G's ~43%. Both recover the sum well. The MLP is where compression happens — both models preserve inputs before the MLP, only G compresses afterward. (Entry 5.)

**Honest read on H1 + H3:** individually unsurprising. "Memorization has structure" and "generalization compresses inputs" are intuitive directions any researcher would predict qualitatively. What the experiments add is *quantitative size* (45-point gap in input recovery), *spatial localization* (MLP is where compression happens), and a *metric we can now use* to track training dynamics, evaluate surgery, and compare across tracks. But on their own, these are characterizations — not enough for a strong TMLR paper. The paper's novelty depends on **H2 (causal surgical removability)** linking these correlational findings together. H2 is the decider, not H1 or H3.

**Implication for H2 (surgical intervention):** the original framing "subtract M's intruder dimensions" doesn't work because every M direction *looks* like an intruder under per-vector cosine. But the principal-angles result says ~5 of G's directions ARE in M's subspace — just rotated. The probe further suggests M's extra directions are doing real work (preserving inputs). Reframed: "project M's weights onto G's column span. Does sel(a) drop to G-like level while sel(sum) stays intact? Does test accuracy move toward G's?"

Next: `taska/analysis/surgery.py` — three variants of the projection experiment.

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
