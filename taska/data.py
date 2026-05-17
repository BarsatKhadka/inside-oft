"""Modular addition dataset. (a, b, =) -> (a + b) mod p.

Matches Nanda et al.: third token is always `p` (the "=" token, vocab id p).
"""
import random
import torch as t


def gen_train_test(p=113, frac_train=0.3, seed=0):
    pairs = [(i, j, p) for i in range(p) for j in range(p)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train * len(pairs))
    return pairs[:div], pairs[div:]


def to_tensors(pairs, p, device):
    inputs = t.tensor(pairs, device=device)
    labels = t.tensor([(i + j) % p for i, j, _ in pairs], device=device)
    return inputs, labels
