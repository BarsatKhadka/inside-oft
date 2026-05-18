"""Comprehensive cross-product matrix: arch × task × WD strength.

Tests whether our claim holds across the full product:
  3 architectures × 4 tasks × 4 WD values = 48 trained models

For each cell record: final test acc, W_out effective rank, gradient ratio
(saddle test). Then build the master 3D table.

If signatures appear in >40 of 48 cells consistently, we have a robust
cross-architecture, cross-task, cross-WD claim. If only some cells confirm,
we have specific scope conditions to report.

Architectures:
  - 1L Transformer (d_model=128)
  - 4L Transformer (d_model=128)
  - MLP (hidden=512)

Tasks:
  - (a+b) mod 113
  - (a-b) mod 113
  - (a*b) mod 113
  - (a*b+1) mod 113

WD values:
  - 0.0, 0.01, 0.1, 1.0

5k epochs each for speed (long enough to see grokking).

Usage:
    python overnight/full_matrix.py
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # for taska imports

import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim

from taska.data import gen_train_test
from taska.model import Transformer

P = 113
LR = 1e-3
NUM_EPOCHS = 5000
LOG_EVERY = 500


def effective_rank(W):
    s = t.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


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
        out = self.fc3(h)
        return out.unsqueeze(1).expand(-1, 3, -1)

    def deep_weight(self):
        return self.fc2.weight


def build_model(arch, device):
    t.manual_seed(0)
    if arch == '1L_Transf':
        m = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
    elif arch == '4L_Transf':
        m = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=4)
    elif arch == 'MLP':
        m = MLPModel()
    return m.to(device)


def get_deep_W(model, arch):
    if arch.endswith('Transf'):
        return model.blocks[-1].mlp.W_out
    elif arch == 'MLP':
        return model.fc2.weight


OPS = {
    'add':       lambda a, b: (a + b) % P,
    'subtract':  lambda a, b: (a - b) % P,
    'mult':      lambda a, b: (a * b) % P,
    'mult_p1':   lambda a, b: (a * b + 1) % P,
}

ARCHS = ['1L_Transf', '4L_Transf', 'MLP']
WDS = [0.0, 0.01, 0.1, 1.0]


def to_tensors(pairs, fn, device):
    inp = t.tensor([(a, b, P) for a, b, _ in pairs], dtype=t.long, device=device)
    lab = t.tensor([fn(a, b) for a, b, _ in pairs], device=device)
    return inp, lab


def gradient_ratio(model, inputs, labels):
    """Quick saddle test."""
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    loss = cross_entropy_hp(model(inputs)[:, -1, :], labels)
    loss.backward()
    total = sum((p.grad ** 2).sum().item() for p in model.parameters() if p.grad is not None)
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    return float(np.sqrt(total))


def train_cell(arch, op_name, wd, device):
    print(f'\n--- arch={arch} op={op_name} wd={wd} ---')
    model = build_model(arch, device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    fn = OPS[op_name]
    train_in, train_lab = to_tensors(train_pairs, fn, device)
    test_in,  test_lab  = to_tensors(test_pairs,  fn, device)

    grok_epoch = None
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            if grok_epoch is None and te >= 0.95:
                grok_epoch = ep + 1

    tr = eval_acc(model, train_in, train_lab)
    te = eval_acc(model, test_in, test_lab)
    rank = effective_rank(get_deep_W(model, arch))
    g_tr = gradient_ratio(model, train_in, train_lab)
    g_te = gradient_ratio(model, test_in, test_lab)
    print(f'  result: train={tr:.4f}, test={te:.4f}, rank={rank:.2f}, grad_ratio={g_te/max(g_tr,1e-12):.2e}, grok@{grok_epoch}')
    return {'arch': arch, 'op': op_name, 'wd': wd,
            'train_acc': tr, 'test_acc': te, 'rank': rank,
            'grad_train': g_tr, 'grad_test': g_te, 'grok_epoch': grok_epoch}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    results = {}
    for arch in ARCHS:
        for op in OPS:
            for wd in WDS:
                key = f'{arch}_{op}_wd{wd}'
                results[key] = train_cell(arch, op, wd, device)

    out_json = HERE / 'results' / 'full_matrix.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Print master table
    print('\n=== Master cross-product table ===')
    print(f'{"arch":>10}  {"op":>10}  {"WD":>6}  {"test_acc":>9}  {"rank":>7}  {"grad_ratio":>11}  {"grok":>6}')
    for arch in ARCHS:
        for op in OPS:
            for wd in WDS:
                r = results[f'{arch}_{op}_wd{wd}']
                gr = r['grad_test'] / max(r['grad_train'], 1e-12)
                print(f'{arch:>10}  {op:>10}  {wd:>6}  {r["test_acc"]:>9.4f}  {r["rank"]:>7.2f}  {gr:>11.2e}  {str(r["grok_epoch"]):>6}')

    # Count how many cells confirm "M (wd=0) is high-rank, G (wd=1) is low-rank"
    n_confirm = 0
    n_total = 0
    for arch in ARCHS:
        for op in OPS:
            M_rank = results[f'{arch}_{op}_wd0.0']['rank']
            G_rank = results[f'{arch}_{op}_wd1.0']['rank']
            n_total += 1
            if M_rank > 2 * G_rank and results[f'{arch}_{op}_wd1.0']['test_acc'] > 0.95:
                n_confirm += 1
    print(f'\nCells where M_rank > 2x G_rank AND G groks: {n_confirm}/{n_total}')


if __name__ == '__main__':
    main()
