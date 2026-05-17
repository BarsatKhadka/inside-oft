# Track A Plan — Modular Addition

## The honest reframing

Nanda's repo solves "train a grokking model." That's not your contribution. Your paper is everything *after* the model exists. So:

- **Don't reinvent the training code.** Use his as the starting point.
- **Don't blindly copy it either.** Read it line-by-line until you understand every choice. The understanding is the learning; the typing is incidental.
- **Your novel code starts at "save the checkpoint" and goes forward.**

That last point is the most important one. Your paper isn't "we trained a transformer on modular addition." It's "we compared two transformers on modular addition through four lenses and intervened on weight space to convert one into the other." None of that exists in Nanda's repo.

## Track A plan, in three phases

### Phase 1 — Get the instrument working (target: 3-4 days)

**Goal:** two trained models on disk — one generalizing (G), one memorizing (M) — with checkpoints saved every N epochs so you can analyze training dynamics later. 

1. Clone the repo. Read `Grokking_Analysis.ipynb` top to bottom. Don't run anything yet. Take notes on:
   - How they construct the dataset
   - How the model class is defined (layer by layer)
   - How the training loop is structured
   - How they save activations and weights

2. **Rewrite the training script as a standalone `.py` file** (not a notebook). This is your one "rewrite from scratch" task — because you'll be running it twice with different configs and modifying it later for other tracks. Structure:
   ```
   train.py --config configs/G.yaml
   train.py --config configs/M.yaml
   ```
   Save checkpoints to `checkpoints/G/epoch_{N}.pt` every 1000 epochs.

3. Train G (weight decay = 1.0). 40k epochs, full batch. On a decent GPU this is ~30-60 minutes.

4. Train M (weight decay = 0.0). Same length. Same init seed if you want a clean paired comparison.

5. **Verify the instrument works** by reproducing the grokking curve for G — train loss drops fast, test loss stays high then suddenly drops. If you don't see grokking in G, something is broken; fix it before moving on. For M, you should see train loss drop and test loss *never* drop.

**Exit criterion for Phase 1:** the grokking plot from Nanda's Figure 2 reproduced on your machine, plus a "non-grokking" plot for M. Save both as PNGs. First entries in `so_far_results.md`.

### Phase 2 — Verify the analysis tools (target: 2-3 days)

**Goal:** confirm your analysis pipeline recovers known facts before you trust it on unknown questions.

1. **Fourier analysis of `W_E`** (Nanda's Figure 3). Take G's embedding matrix, do a Fourier transform along the input dimension, plot the norms. Expected: 5-6 spikes at the "key frequencies." This is the *known ground truth*. If your pipeline recovers it, the pipeline works.

2. **SVD of `W_E` for G and M.** Plot singular value spectra on log-y. Expected: G's spectrum drops sharply after ~10 directions (low effective rank, the Fourier structure). M's spectrum is more uniform (high effective rank, full memorization).

3. **Restricted/excluded loss** (Nanda's progress measures). Pick the 5 key frequencies for G. Ablate everything *except* them → restricted loss should be low. Ablate *only* them → excluded loss should be high. For M, both should be high (no key frequencies exist).

**Exit criterion for Phase 2:** Your tooling reproduces Nanda's qualitative findings on G. Now you know the instrument doesn't lie.

### Phase 3 — Your actual paper (target: 1-2 weeks)

**Now the novel work begins.** Everything before this was setup.

1. **Intruder dimensions in M.** Compare singular vectors of M's `W_E` to G's `W_E`. Which directions in M have no analog in G? Are they high-singular-value? Plot the singular values vs the max cosine-similarity to G's vectors. (This is H1 for the modular addition setting.)

2. **Per-example probe on residual stream.** For each training example, extract the residual stream after the MLP. Train a logistic regression to predict example index. Selectivity baseline: same probe on random labels. Compare M vs G. (This is H3.)

3. **Surgical intervention — the headline experiment.** Take M, identify its "intruder directions," subtract them from the weights, evaluate. Does test accuracy improve? Does train accuracy drop selectively on the most-memorized examples? Sweep how many directions you ablate. (This is H2.)

4. **Hessian top-k.** Compute top-50 Hessian eigenvalues at convergence for both M and G. Compare sharpness anisotropy `λ_1 / λ_50`. Then check: do the top Hessian eigenvectors align with the intruder singular directions from step 1? (This is H4 — the joint claim.)

**Exit criterion for Phase 3:** A figure for each of (1), (2), (3), (4) showing the M vs G difference, with the surgical intervention from (3) being the headline plot. These four figures *are* the Track A contribution to the paper.

## Repo structure to start

```
inside-oft/
  train.py                  # the one training script you'll rewrite
  configs/
    G.yaml
    M.yaml
  checkpoints/
    G/                      # G model checkpoints over training
    M/
  analysis/
    fourier.py              # Phase 2.1
    svd_compare.py          # Phase 2.2, Phase 3.1
    progress_measures.py    # Phase 2.3
    probe.py                # Phase 3.2
    surgery.py              # Phase 3.3 — THE BIG ONE
    hessian.py              # Phase 3.4
  results/
    fig_grokking_G.png
    fig_grokking_M.png
    ...
  so_far.md
  so_far_results.md
```

## Where to look in Nanda's notebook for what you need

- **Dataset construction:** the early cells that build `all_data`, `train_data`, `test_data`.
- **Model definition:** the `Transformer` class. Read it carefully — it's <100 lines.
- **Training loop:** straightforward, no surprises.
- **Fourier analysis:** look for `fourier_basis` definition and `make_fourier_basis` function.
- **Restricted/excluded loss:** look for cells that ablate frequencies.

**Skip for now:** the more elaborate circuit-discovery cells (head-by-head analysis, etc.). You don't need those for Phase 1-3. Come back to them when you write the paper.

## What to do first, concretely

Today's task: **clone the repo and read the notebook end-to-end with a notebook of your own next to you.** Write down every line you don't fully understand. Don't run anything. Tomorrow, we'll go through your questions and then start Phase 1.
