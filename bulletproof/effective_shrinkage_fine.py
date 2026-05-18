"""Effective shrinkage fine grid + multi-seed for quantitative law.

Refined version of effective_shrinkage with multi-seed for error bars.

7 LRs × 7 WDs × 2 seeds = 98 runs × 15k epochs each. Heavy but produces the
quantitative escape boundary with error bars.

Tests: is the escape boundary EXACTLY LR × WD = constant? Or is there an
exponent c such that LR^a × WD^b = constant?

Usage:
    python bulletproof/effective_shrinkage_fine.py
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
NUM_EPOCHS = 15000


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def effective_rank(W):
    s = t.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def cell(lr, wd, seed, device, train_in, train_lab, test_in, test_lab):
    print(f'\n--- lr={lr}, wd={wd}, seed={seed} ---')
    t.manual_seed(seed)
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    grok_ep = None
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 500 == 0:
            te = eval_acc(model, test_in, test_lab)
            if grok_ep is None and te >= 0.95:
                grok_ep = ep + 1
    return {'lr': lr, 'wd': wd, 'seed': seed,
            'final_test_acc': eval_acc(model, test_in, test_lab),
            'final_rank': effective_rank(model.blocks[0].mlp.W_out),
            'grok_epoch': grok_ep}


LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
WDS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    results = {}
    for lr in LRS:
        for wd in WDS:
            for seed in [0, 1]:
                key = f'lr{lr}_wd{wd}_seed{seed}'
                results[key] = cell(lr, wd, seed, device, train_in, train_lab, test_in, test_lab)

    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'effective_shrinkage_fine.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== Escape table (Y / N per seed) ===')
    print(f'{"lr/wd":>10}  ' + '  '.join(f'{w:>10}' for w in WDS))
    for lr in LRS:
        row = []
        for wd in WDS:
            ss = [results[f'lr{lr}_wd{wd}_seed{s}'] for s in [0, 1]]
            mark = ''.join('Y' if r['grok_epoch'] else 'N' for r in ss)
            row.append(f'{mark:>10}')
        print(f'{lr:>10.4g}  ' + '  '.join(row))


if __name__ == '__main__':
    main()
