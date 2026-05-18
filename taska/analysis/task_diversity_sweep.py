"""Task diversity sweep: how does rank depend on task and modulus?

For 1-layer transformer (d_model=128), train G on many different modular
tasks and primes. Measure converged W_out rank as a function of:
  (a) operation: add, subtract, multiply, square
  (b) prime modulus: 23, 53, 113, 263

Tests:
  1. Does converged rank scale with prime (task complexity)?
  2. Does rank differ between operations (add vs mult)?
  3. Are some primes harder to grok than others?

If we get clean scaling rank = f(prime, op), that's a quantitative law.

Usage:
    python taska/analysis/task_diversity_sweep.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import json
import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.optim as optim

from model import Transformer

LR = 1e-3
WD = 1.0
NUM_EPOCHS = 30000
LOG_EVERY = 500
D_MODEL = 128

PRIMES = [23, 53, 113, 263]
OPERATIONS = {
    'add':       lambda a, b, p: (a + b) % p,
    'subtract':  lambda a, b, p: (a - b) % p,
    'mult':      lambda a, b, p: (a * b) % p,
}


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


def make_data(P, fn, device, frac_train=0.3):
    pairs = [(i, j) for i in range(P) for j in range(P)]
    rnd.seed(0)
    rnd.shuffle(pairs)
    div = int(frac_train * len(pairs))
    train, test = pairs[:div], pairs[div:]
    def to_t(ps):
        inp = t.tensor([(a, b, P) for a, b in ps], dtype=t.long, device=device)
        lab = t.tensor([fn(a, b, P) for a, b in ps], device=device)
        return inp, lab
    return to_t(train), to_t(test)


def train_one(op_name, fn, P, device):
    print(f'\n=== op={op_name}, P={P} ===')
    t.manual_seed(0)
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))

    (train_in, train_lab), (test_in, test_lab) = make_data(P, fn, device)

    grok_epoch = None
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            if grok_epoch is None and te >= 0.95:
                grok_epoch = ep + 1
        if (ep + 1) % 5000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}')

    final_te = eval_acc(model, test_in, test_lab)
    rank_W_out = effective_rank(model.blocks[0].mlp.W_out)
    rank_W_in  = effective_rank(model.blocks[0].mlp.W_in)
    rank_W_E   = effective_rank(model.embed.W_E[:, :P])
    print(f'  result: test_acc={final_te:.4f}, grok@{grok_epoch}, rank_W_out={rank_W_out:.2f}')
    return {'op': op_name, 'P': P, 'test_acc': final_te, 'grok_epoch': grok_epoch,
            'rank_W_out': rank_W_out, 'rank_W_in': rank_W_in, 'rank_W_E': rank_W_E}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    results = {}
    for op_name, fn in OPERATIONS.items():
        for P in PRIMES:
            key = f'{op_name}_p{P}'
            results[key] = train_one(op_name, fn, P, device)

    out_json = HERE / 'results' / 'task_diversity_sweep.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Plot: rank vs prime, per operation
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for op_name in OPERATIONS:
        xs = []
        ys = []
        for P in PRIMES:
            r = results[f'{op_name}_p{P}']
            if r['test_acc'] >= 0.95:
                xs.append(P)
                ys.append(r['rank_W_out'])
        ax.plot(xs, ys, marker='o', label=op_name)
    ax.set_xlabel('prime modulus p')
    ax.set_ylabel('W_out effective rank (G)')
    ax.set_title('Rank vs task size: does rank scale with p?')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Log scale fit
    ax.set_yscale('log')

    ax = axes[1]
    for op_name in OPERATIONS:
        xs = []
        ys = []
        for P in PRIMES:
            r = results[f'{op_name}_p{P}']
            xs.append(P)
            ys.append(r['grok_epoch'] if r['grok_epoch'] is not None else NUM_EPOCHS + 1)
        ax.plot(xs, ys, marker='s', label=op_name)
    ax.set_xlabel('prime modulus p')
    ax.set_ylabel('grok epoch')
    ax.set_title('Grok time vs task size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Task diversity: rank and grok time vs operation and prime')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_task_diversity_sweep.png'
    fig.savefig(out, dpi=130)
    print(f'\nplot -> {out}')

    # Print table
    print('\n=== Per-task table ===')
    for op_name in OPERATIONS:
        for P in PRIMES:
            r = results[f'{op_name}_p{P}']
            print(f'  {op_name:10s} p={P:>4}: test_acc={r["test_acc"]:.4f}, '
                  f'grok@{r["grok_epoch"]}, rank_W_out={r["rank_W_out"]:.2f}')


if __name__ == '__main__':
    main()
