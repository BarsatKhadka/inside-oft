# so_far.md

What I've understood, decided, and done. Updated as I go.

---

## The question, in my own words

Take two copies of the same network. Train both on the same data. One I stop at the right time (it generalizes — call it **G**). The other I keep training past that point with no regularization (it overfits — call it **O**). Is O just G + noise, or is O a fundamentally different object?

If it's a different object, *what is the difference made of*, and can I find it, measure it, and surgically remove it?

## The four things I'm testing (plain English)

1. **Weight signature:** O's weights have specific structural patterns that G's don't.
2. **Surgical removal:** I can identify a small chunk of O's weights and cut it out, and that turns O back into G — without retraining.
3. **Per-example memory:** Inside O's activations, there's a readable signal telling you which training example it just saw. G doesn't have this signal.
4. **One mechanism:** 1, 2, 3 are the same underlying thing viewed from different angles. Intervening on one moves the others.

## Decisions locked in so far

- Target: TMLR submission.
- Approach: same-architecture, same-data, only-training-regime-different comparisons. Never compare across architectures.
- Tracks (in priority order):
  - **Track A:** modular addition (Nanda setup) — sanity check, known ground truth.
  - **Track B:** CIFAR-10 + ResNet-18 — the main experimental story.
  - **Track C:** Pythia fine-tuning — appendix / v2 if time permits.
- Learning policy: I write every experiment from scratch the first time I need it. No copy-paste from someone else's repo until I've already done the thing myself once at smaller scale.

## What I've actually done

(empty — will fill in as I go)

## What I don't know yet but need to

- How to compute an SVD of a weight matrix and read what it tells me.
- What "intruder dimensions" really look like when you plot them.
- How a Hessian eigenvalue is actually computed in practice (Lanczos / power iteration / autograd-tricks).
- What a linear probe is and why it's "linear" specifically.
- How to train a sparse autoencoder on activations.

I'll learn each of these by needing it for an experiment, not by reading about it in advance.

## Open questions I'm parking

- Whether to include grokking as a "mode" or just as a control. (Decide after Track A Day 1.)
- Whether to start Track B from-scratch or from a pretrained checkpoint. (Plan says from-scratch; revisit after Track A.)
- How to define "overfit" precisely — currently "K epochs past peak val accuracy, K = 3× time-to-peak." Lock or refine after first real run.
