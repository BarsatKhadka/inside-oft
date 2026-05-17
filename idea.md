TIME: 30 days 
TARGET : TMLR


Version A: "What does an overfit model know that a generalizing model doesn't?"
Setup: Train two versions of the same model on the same task. One generalizes (early-stopped, regularized). One overfits hard (trained way past convergence, no regularization). Compare them mechanistically: what circuits, features, or weight-space structure differ? The overfit model has memorized something the generalizing one hasn't. What is that something, structurally?
This is interesting because the conventional view is "overfitting = bad, throw it away." But an overfit model contains information about the training data that the generalizing model has averaged out. If you can extract that information mechanistically, you've shown overfitting isn't just degradation — it's a different kind of representation.
Connection to broader literature: this touches grokking (Power et al., Nanda et al.), double descent, memorization vs. generalization (Feldman, Zhang). The novel angle would be the mechanistic characterization — not just "they behave differently" but "this specific structure differs."


Title (working): "What Does an Overfit Network Know? A Mechanistic Characterization of Overfitting

Core hypothesis: Overfit fine-tuned models develop specific, identifiable structural properties — in their singular value spectra, in their weight-space geometry, in their internal representations — that distinguish them from generalizing models. These properties are not just "more extreme" versions of generalizing models' properties; they're qualitatively different.


---

## Ideas to revisit (parked after day 1 experiments)

### MIA via gated computation
Empirically (probe_test.py): for M, a linear probe on resid_post predicts (a+b) at 93% on training inputs but 2% on test inputs — a 91-point gap. For G the same probe gives 100% on both. This 91-point gap is essentially a perfect membership inference attack: train a probe to predict (a+b) on known-training inputs, apply to a novel input; if probe matches model output → input was in training. Mechanism: M's MLP only "fires the sum circuit" for inputs it has seen. Connect to Carlini et al. privacy literature. Could be a clean mechanistic explanation for why MIA works on overfit models in general.

### "Knows its training history" framing
M knows which inputs it was trained on (perfect record encoded in MLP activations). G has forgotten which inputs it was trained on (uniform computation regardless of input). Tweet-worthy: "An overfit model knows its training history. A generalizing model has forgotten it."

### Conditional/gated MLP → can we ungate it?
M's MLP appears to be a gated computation: fires (a+b) for seen inputs, fires nothing useful for unseen ones. Open question: if we find the gating neurons (the ones that detect "is this in training?") and ablate them, does M start computing (a+b) for everyone?

Worry: even without the gate, M may not have an underlying generalization circuit — just a lookup with nothing else. In that case ablation produces garbage on test inputs.
Hope: there might be a partial circuit underneath that the gate is suppressing.
Either result is informative — first real shot at "convert overfit into generalizing" via a mechanistic intervention, not spectral.

### What we ruled out
- Hidden generalization in M (probe predicts test sum at chance — M genuinely doesn't have a general circuit)
- Spectral surgery on any layer / combination (all three failed)
- Permutation alignment (M and G are different algorithms, not permuted versions)