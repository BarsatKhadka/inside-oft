 The plan says: take two copies of the same network trained on the same data. One you train normally and it learns to do the task on new examples (call this G, the generalizing one). The
  other you train too long with no regularization, so it gets perfect on training data but worse on new data (call this O, the overfit one).

  The big question: is O just a worse version of G, or is O a different kind of thing entirely?

  The four guesses (renamed in plain words):

  - Guess 1 — "the weights look different": If you crack open O's weights and look at them mathematically, you'll see specific patterns that aren't in G's weights. Not "more noise" — actual
   new structure.
  - Guess 2 — "you can surgically remove the overfitting": There's a small piece of O's weights that is the memorization. Cut it out and O becomes G, without retraining.
  - Guess 3 — "O remembers which training example it saw": If you feed O a training example, somewhere inside the network there's a signal that says "this is example #4,213." G doesn't have
   that signal.
  - Guess 4 — "all of the above are the same thing": Guesses 1, 2, 3 are different views of one underlying mechanism. The surgery from Guess 2 should also kill the signal from Guess 3.
