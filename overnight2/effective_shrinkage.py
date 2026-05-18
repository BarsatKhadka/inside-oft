"""Effective shrinkage hypothesis: escape depends on LR × WD product, not WD alone.

If our rank-reduction story is right, the WD threshold for escape should scale
INVERSELY with learning rate. Specifically: the effective weight shrinkage per
step is approximately LR × WD. If escape requires effective_shrinkage ≥ S*,
then we should see escape iff LR × WD ≥ S*.

Sweep (LR, WD) on a 2D grid for 1L Transformer on (a+b) mod 113.
LR ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
WD ∈ {0.01, 0.1, 1.0, 10.0, 100.0}

25 cells × 20k epochs.

If escape boundary is a hyperbola LR × WD = constant, our story is confirmed
quantitatively.

Usage:
    python overnight2/effective_shrinkage.py
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
NUM_EPOCHS = 20000
LOG_EVERY = 500
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
WDS = [0.01, 0.1, 1.0, 10.0, 100.0]


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


def cell(lr, wd, device):
    print(f'\n--- lr={lr}, wd={wd}, lr*wd={lr*wd:.4f} ---')
    t.manual_seed(0)
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)
    grok_ep = None
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            if grok_ep is None and te >= 0.95:
                grok_ep = ep + 1
        if (ep + 1) % 5000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}')
    return {'lr': lr, 'wd': wd, 'lr_wd_product': lr * wd,
            'final_test_acc': eval_acc(model, test_in, test_lab),
            'final_rank': effective_rank(model.blocks[0].mlp.W_out),
            'grok_epoch': grok_ep}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {}
    for lr in LRS:
        for wd in WDS:
            results[f'lr{lr}_wd{wd}'] = cell(lr, wd, device)
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'effective_shrinkage.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== 2D table: did it grok? (cell value = lr*wd if grokked, else NO) ===')
    header = 'LR/WD'
    print(f'{header:>10s}  ' + '  '.join(f'{w:>9}' for w in WDS))
    for lr in LRS:
        row = []
        for wd in WDS:
            r = results[f'lr{lr}_wd{wd}']
            if r['grok_epoch']:
                row.append(f'{lr*wd:>9.4f}')
            else:
                row.append('NO       ')
        print(f'{lr:>10.4f}  ' + '  '.join(row))


if __name__ == '__main__':
    main()
