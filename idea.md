TIME: 30 days 
TARGET : TMLR


Version A: "What does an overfit model know that a generalizing model doesn't?"
Setup: Train two versions of the same model on the same task. One generalizes (early-stopped, regularized). One overfits hard (trained way past convergence, no regularization). Compare them mechanistically: what circuits, features, or weight-space structure differ? The overfit model has memorized something the generalizing one hasn't. What is that something, structurally?
This is interesting because the conventional view is "overfitting = bad, throw it away." But an overfit model contains information about the training data that the generalizing model has averaged out. If you can extract that information mechanistically, you've shown overfitting isn't just degradation — it's a different kind of representation.
Connection to broader literature: this touches grokking (Power et al., Nanda et al.), double descent, memorization vs. generalization (Feldman, Zhang). The novel angle would be the mechanistic characterization — not just "they behave differently" but "this specific structure differs."


Title (working): "What Does an Overfit Network Know? A Mechanistic Characterization of Overfitting

Core hypothesis: Overfit fine-tuned models develop specific, identifiable structural properties — in their singular value spectra, in their weight-space geometry, in their internal representations — that distinguish them from generalizing models. These properties are not just "more extreme" versions of generalizing models' properties; they're qualitatively different.