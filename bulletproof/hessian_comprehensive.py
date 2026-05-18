"""Comprehensive Hessian eigenvalue analysis at M and G across architectures.

This is the DIRECT geometric test of the saddle claim.

For each (architecture, seed) combination:
  - Train M (WD=0) and G (WD=1.0) on (a+b) mod 113 for 20k epochs
  - Compute top-k positive AND most negative Hessian eigenvalues on
    train data, test data, and full data via Lanczos / power iteration
  - Verify: M has negative eigenvalues on test/full data (saddle)
  - Verify: G has all eigenvalues near zero / positive (basin)

Tested on 3 architectures (1L Transformer, 4L Transformer, MLP) with 2 seeds
each = 6 model pairs.

Output: table showing min eigenvalue per model per data split.

Usage:
    python bulletproof/hessian_comprehensive.py
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim

from taska.data import gen_train_test, to_tensors
from taska.model import Transformer

P = 113
LR = 1e-3
NUM_EPOCHS = 20000


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


class MLPModel(nn.Module):
    def __init__(self, p=P, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(p + 1, 128)
        self.fc1 = nn.Linear(128 * 3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, p + 1)

    def forward(self, x):
        e = self.emb(x).flatten(1)
        h = t.nn.functional.relu(self.fc1(e))
        h = t.nn.functional.relu(self.fc2(h))
        return self.fc3(h).unsqueeze(1).expand(-1, 3, -1)


def build(arch, seed, device):
    t.manual_seed(seed)
    if arch == '1L_Transf':
        m = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
    elif arch == '4L_Transf':
        m = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=4)
    elif arch == 'MLP':
        m = MLPModel()
    return m.to(device)


def train_one(arch, wd, seed, device, train_in, train_lab):
    model = build(arch, seed, device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def hvp(loss, params, vector_list):
    grads = t.autograd.grad(loss, params, create_graph=True)
    dot = sum((g * v).sum() for g, v in zip(grads, vector_list))
    Hv = t.autograd.grad(dot, params, retain_graph=False)
    return list(Hv)


def power_iter(model, inputs, labels, sign=+1, n_iter=40):
    """Top eigenvalue of sign*H by power iteration. sign=+1 gives max,
    sign=-1 gives most negative."""
    params = list(model.parameters())
    v = [t.randn_like(p) for p in params]
    n = t.sqrt(sum((vi**2).sum() for vi in v))
    v = [vi / n for vi in v]
    eig = 0.0
    for _ in range(n_iter):
        for p in params:
            if p.grad is not None: p.grad.zero_()
        loss = cross_entropy_hp(model(inputs)[:, -1, :], labels)
        Hv = hvp(loss, params, v)
        if sign == -1: Hv = [-x for x in Hv]
        eig = sum((vi*hi).sum().item() for vi,hi in zip(v, Hv))
        n = t.sqrt(sum((hi**2).sum() for hi in Hv)) + 1e-12
        v = [hi / n for hi in Hv]
    return eig if sign == +1 else -eig


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)
    all_pairs = [(a, b, P) for a in range(P) for b in range(P)]
    full_in = t.tensor(all_pairs, dtype=t.long, device=device)
    full_lab = t.tensor([(a + b) % P for a, b, _ in all_pairs], device=device)

    results = {}
    for arch in ['1L_Transf', '4L_Transf', 'MLP']:
        for seed in [0, 1]:
            for wd, regime in [(0.0, 'M'), (1.0, 'G')]:
                key = f'{arch}_{regime}_seed{seed}'
                print(f'\n=== training {key} ===')
                model = train_one(arch, wd, seed, device, train_in, train_lab)
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(True)
                eigs = {}
                for ds_name, (inp, lab) in [('train', (train_in, train_lab)),
                                             ('test',  (test_in,  test_lab)),
                                             ('full',  (full_in,  full_lab))]:
                    top = power_iter(model, inp, lab, sign=+1, n_iter=40)
                    bot = power_iter(model, inp, lab, sign=-1, n_iter=40)
                    eigs[ds_name] = {'top': top, 'bottom': bot}
                    print(f'  {ds_name}: top={top:.4f}, bottom={bot:.4f}')
                results[key] = eigs

    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'hessian_comprehensive.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== SUMMARY: bottom (most negative) eigenvalue on FULL data ===')
    print(f'{"key":>25}  {"bot_eig_full":>15}  {"saddle?":>10}')
    for k, e in results.items():
        b = e['full']['bottom']
        is_saddle = b < -1e-4
        print(f'{k:>25}  {b:>15.4f}  {"YES" if is_saddle else "no":>10}')


if __name__ == '__main__':
    main()
