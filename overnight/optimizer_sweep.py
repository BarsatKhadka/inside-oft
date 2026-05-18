"""Optimizer sweep: is the WD mechanism optimizer-independent?

Tests whether the "WD escapes saddle via rank compression" mechanism works
with SGD, AdamW, Adam, and decoupled vs coupled weight decay. If only AdamW
shows the effect, our story is AdamW-specific. If all optimizers show it,
the mechanism is fundamental to weight decay regardless of optimizer.

Optimizers tested (all with same lr=1e-3, on modular addition (a+b) mod 113):
  1. SGD with momentum, decoupled WD
  2. SGD with momentum, coupled WD (L2 in loss)
  3. AdamW (decoupled WD) — control, what we've been using
  4. Adam + L2 in loss (coupled WD)

For each: WD ∈ {0.0, 1.0}. 8 runs × 30k epochs each.

If grokking with WD=1.0 happens across optimizers, the WD mechanism is
optimizer-independent. If only some optimizers grok, we need to refine
the claim ("decoupled WD specifically" or "any low-norm bias").

Usage:
    python overnight/optimizer_sweep.py
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch as t
import torch.optim as optim

from taska.data import gen_train_test, to_tensors
from taska.model import Transformer

P = 113
LR = 1e-3
NUM_EPOCHS = 30000
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


def L2_pen(model):
    return sum((p ** 2).sum() for p in model.parameters())


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


OPTIMIZERS = ['SGD_dec', 'SGD_coupled', 'AdamW_dec', 'Adam_coupled']


def make_optimizer(opt_name, model, wd):
    if opt_name == 'SGD_dec':
        return optim.SGD(model.parameters(), lr=LR * 50, momentum=0.9, weight_decay=wd, nesterov=True)
    elif opt_name == 'SGD_coupled':
        return optim.SGD(model.parameters(), lr=LR * 50, momentum=0.9, weight_decay=0.0, nesterov=True)
    elif opt_name == 'AdamW_dec':
        return optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))
    elif opt_name == 'Adam_coupled':
        return optim.Adam(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))


def is_coupled(opt_name):
    return 'coupled' in opt_name


def train_one(opt_name, wd, device, train_in, train_lab, test_in, test_lab):
    print(f'\n--- opt={opt_name}, wd={wd} ---')
    t.manual_seed(0)
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
    optimizer = make_optimizer(opt_name, model, wd)
    coupled = is_coupled(opt_name)
    history = {'epoch': [], 'test_acc': [], 'rank_W_out': []}
    grok_epoch = None
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        if coupled and wd > 0:
            loss = loss + wd * L2_pen(model)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            rank = effective_rank(model.blocks[0].mlp.W_out)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
            history['rank_W_out'].append(rank)
            if grok_epoch is None and te >= 0.95:
                grok_epoch = ep + 1
        if (ep + 1) % 5000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}, rank={rank:.2f}')
    return {'history': history, 'grok_epoch': grok_epoch,
            'final_test_acc': history['test_acc'][-1],
            'final_rank': history['rank_W_out'][-1]}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    results = {}
    for opt_name in OPTIMIZERS:
        for wd in [0.0, 1.0]:
            key = f'{opt_name}_wd{wd}'
            results[key] = train_one(opt_name, wd, device, train_in, train_lab, test_in, test_lab)

    out_json = HERE / 'results' / 'optimizer_sweep.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Plot test acc over time per optimizer
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, target in zip(axes, ['test_acc', 'rank_W_out']):
        for opt_name in OPTIMIZERS:
            for wd in [0.0, 1.0]:
                h = results[f'{opt_name}_wd{wd}']['history']
                style = '-' if wd == 1.0 else '--'
                ax.plot(h['epoch'], h[target], style, label=f'{opt_name} wd={wd}')
        ax.set_xlabel('epoch')
        ax.set_ylabel(target)
        if target == 'rank_W_out':
            ax.set_yscale('log')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Optimizer sweep: is WD mechanism optimizer-independent?')
    fig.tight_layout()
    fig.savefig(HERE / 'results' / 'fig_optimizer_sweep.png', dpi=130)

    print('\n=== Summary ===')
    print(f'{"opt":>15}  {"wd":>4}  {"final_acc":>10}  {"final_rank":>10}  {"grok_epoch":>11}')
    for opt_name in OPTIMIZERS:
        for wd in [0.0, 1.0]:
            r = results[f'{opt_name}_wd{wd}']
            print(f'{opt_name:>15}  {wd:>4}  {r["final_test_acc"]:>10.4f}  '
                  f'{r["final_rank"]:>10.2f}  {str(r["grok_epoch"]):>11}')


if __name__ == '__main__':
    main()
