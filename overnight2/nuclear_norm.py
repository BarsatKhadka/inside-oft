"""Nuclear norm penalty (no WD): does explicit rank reduction escape?

Most direct test of "rank reduction IS the mechanism." Use nuclear norm
penalty (sum of singular values) instead of WD. This directly penalizes rank.

L_total = L_CE + λ * Σ ||W_i||_*

For each λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1.0}, train fresh model on modular
addition without WD. Test if it escapes the memorization saddle.

If nuclear norm escapes (similar to WD), rank IS the mechanism, not WD's
specific form. STRONG support for unifying claim.

If nuclear norm fails, WD has a special property beyond just reducing rank.
Refines the claim.

Usage:
    python overnight2/nuclear_norm.py
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
NUM_EPOCHS = 20000
LOG_EVERY = 500
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def effective_rank(W):
    s = t.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def nuclear_norm_penalty(model):
    """Sum of nuclear norms of matrix-shaped weights."""
    total = 0.0
    for name, p in model.named_parameters():
        if p.ndim >= 2 and 'W_' in name:
            W = p.reshape(p.shape[0], -1)
            try:
                s = t.linalg.svdvals(W)
                total = total + s.sum()
            except Exception:
                pass
    return total


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def cell(lam, device):
    print(f'\n--- lambda={lam} ---')
    t.manual_seed(0)
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)
    grok_ep = None
    history = {'epoch': [], 'test_acc': [], 'rank': []}
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        if lam > 0:
            loss = loss + lam * nuclear_norm_penalty(model)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            rank = effective_rank(model.blocks[0].mlp.W_out)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
            history['rank'].append(rank)
            if grok_ep is None and te >= 0.95:
                grok_ep = ep + 1
        if (ep + 1) % 5000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}, rank={rank:.2f}')
    return {'lam': lam, 'grok_epoch': grok_ep, 'history': history,
            'final_test_acc': history['test_acc'][-1],
            'final_rank': history['rank'][-1]}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {}
    for lam in LAMBDAS:
        results[f'lam{lam}'] = cell(lam, device)
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'nuclear_norm.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== Nuclear norm escape results ===')
    for lam in LAMBDAS:
        r = results[f'lam{lam}']
        print(f"  lambda={lam}: test_acc={r['final_test_acc']:.4f}, "
              f"rank={r['final_rank']:.2f}, grok@{r['grok_epoch']}")


if __name__ == '__main__':
    main()
