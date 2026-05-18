"""Alternative rank penalties: do MANY ways of reducing rank escape, or just nuclear norm?

Test:
  1. Nuclear norm (Schatten-1): sum of singular values (already tested)
  2. Schatten-0.5 norm: smooth lower-rank-inducing
  3. Schatten-2 (Frobenius squared): standard L2 — should also escape
  4. log-rank surrogate: sum(log(σ_i^2 + ε))
  5. Top-k singular value penalty: penalize σ_k for k > target

If multiple rank-reducing penalties escape, the mechanism IS rank reduction
(not nuclear norm specifically).

For each penalty, sweep its strength to find escape vs no-escape regime.

Usage:
    python bulletproof/alternative_rank_penalties.py
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
NUM_EPOCHS = 12000


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


def matrix_weights(model):
    return [(n, p) for n, p in model.named_parameters() if p.ndim >= 2 and 'W_' in n]


# Penalty functions

def penalty_nuclear(model):
    total = 0.0
    for n, p in matrix_weights(model):
        W = p.reshape(p.shape[0], -1)
        total = total + t.linalg.svdvals(W).sum()
    return total


def penalty_schatten_half(model):
    """Schatten-1/2 norm — more aggressive rank-reducing than nuclear."""
    total = 0.0
    for n, p in matrix_weights(model):
        W = p.reshape(p.shape[0], -1)
        s = t.linalg.svdvals(W) + 1e-8
        total = total + (s ** 0.5).sum()
    return total


def penalty_frobenius_squared(model):
    """Schatten-2 = Frobenius — equivalent to L2-in-loss."""
    total = 0.0
    for n, p in matrix_weights(model):
        total = total + (p ** 2).sum()
    return total


def penalty_log_singular(model):
    """log-determinant surrogate: penalize log(σ_i^2 + ε). Smooth low-rank prior."""
    total = 0.0
    for n, p in matrix_weights(model):
        W = p.reshape(p.shape[0], -1)
        s = t.linalg.svdvals(W) ** 2 + 1e-4
        total = total + t.log(s).sum()
    return total


def penalty_tail_singular(model, k=10):
    """Penalize only the tail (σ_k onward) — directly rank-reducing."""
    total = 0.0
    for n, p in matrix_weights(model):
        W = p.reshape(p.shape[0], -1)
        s = t.linalg.svdvals(W)
        if s.shape[0] > k:
            total = total + (s[k:] ** 2).sum()
    return total


PENALTIES = {
    'nuclear':       (penalty_nuclear,           [1e-4, 5e-4]),
    'schatten_half': (penalty_schatten_half,     [1e-3, 5e-3]),
    'frobenius_sq':  (penalty_frobenius_squared, [1e-4, 1e-3]),
    'log_singular':  (penalty_log_singular,      [1e-4, 1e-3]),
    'tail_singular': (lambda m: penalty_tail_singular(m, k=10), [1e-3, 1e-2]),
}


def cell(pen_name, lam, device, train_in, train_lab, test_in, test_lab):
    print(f'\n--- {pen_name} lam={lam} ---')
    t.manual_seed(0)
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))
    pen_fn = PENALTIES[pen_name][0]
    grok_ep = None
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        loss = loss + lam * pen_fn(model)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 500 == 0:
            te = eval_acc(model, test_in, test_lab)
            if grok_ep is None and te >= 0.95:
                grok_ep = ep + 1
        if (ep + 1) % 3000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}')
    rank = effective_rank(model.blocks[0].mlp.W_out)
    return {'pen': pen_name, 'lam': lam,
            'final_test_acc': eval_acc(model, test_in, test_lab),
            'final_rank': rank, 'grok_epoch': grok_ep}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    results = {}
    for pen_name, (fn, lams) in PENALTIES.items():
        for lam in lams:
            key = f'{pen_name}_lam{lam}'
            results[key] = cell(pen_name, lam, device, train_in, train_lab, test_in, test_lab)

    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'alternative_rank_penalties.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== Alternative rank penalty escape ===')
    for k, r in results.items():
        print(f'  {k}: test={r["final_test_acc"]:.4f}, rank={r["final_rank"]:.2f}, grok@{r["grok_epoch"]}')


if __name__ == '__main__':
    main()
