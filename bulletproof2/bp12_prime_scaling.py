"""bp12: Prime-size scaling.

G on (a+b) mod p for p in {53, 113, 257, 509, 1009}, 5 seeds each.
Plot converged G-rank vs p.

Prediction: rank scales as some clean function of p (Nanda found ~5-10 key
Fourier frequencies; rank should grow modestly with p).
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.optim as optim

from _common import (HERE, LR, cross_entropy_hp, effective_rank, eval_acc)
from taska.data import gen_train_test, to_tensors
from taska.model import Transformer

NUM_SEEDS = 5
PRIMES = [53, 113, 257, 509, 1009]


def epochs_for_p(p):
    # Scale training budget with p (larger vocab -> harder grok).
    return {53: 15000, 113: 20000, 257: 30000, 509: 45000, 1009: 60000}[p]


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = []
    for p in PRIMES:
        for seed in range(NUM_SEEDS):
            print(f'\n--- p={p} seed={seed} ---')
            t.manual_seed(seed); np.random.seed(seed)
            d_model = 128
            model = Transformer(p=p, d_model=d_model, num_heads=4, n_ctx=3, num_layers=1).to(device)
            opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0, betas=(0.9, 0.98))
            train_pairs, test_pairs = gen_train_test(p=p, frac_train=0.3, seed=seed)
            tr_in, tr_lab = to_tensors(train_pairs, p, device=device)
            te_in, te_lab = to_tensors(test_pairs, p, device=device)
            grok_ep = None
            n_epochs = epochs_for_p(p)
            for ep in range(n_epochs):
                loss = cross_entropy_hp(model(tr_in)[:, -1, :], tr_lab)
                opt.zero_grad(); loss.backward(); opt.step()
                if (ep + 1) % 1000 == 0:
                    te = eval_acc(model, te_in, te_lab)
                    if grok_ep is None and te >= 0.95:
                        grok_ep = ep + 1
                    if (ep + 1) % 5000 == 0:
                        print(f'  ep={ep+1}: test={te:.4f}')
            final_te = eval_acc(model, te_in, te_lab)
            rank_out = effective_rank(model.blocks[0].mlp.W_out)
            rank_in = effective_rank(model.blocks[0].mlp.W_in)
            rank_E = effective_rank(model.embed.W_E)
            entry = {
                'p': p, 'seed': seed, 'd_model': d_model,
                'final_test_acc': final_te, 'grok_epoch': grok_ep,
                'rank_W_out': rank_out, 'rank_W_in': rank_in, 'rank_W_E': rank_E,
                'n_epochs': n_epochs,
            }
            results.append(entry)
            print(f'  test={final_te:.4f}, rank_W_out={rank_out:.2f}, rank_W_E={rank_E:.2f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp12_prime_scaling.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Summarize
    print('\n=== rank vs p (G models that grokked) ===')
    for p in PRIMES:
        cell = [r for r in results if r['p'] == p and r['final_test_acc'] >= 0.9]
        if cell:
            rs = np.array([r['rank_W_out'] for r in cell])
            print(f'  p={p}: rank_W_out = {rs.mean():.2f} +- {rs.std():.2f}  (n={len(cell)})')


if __name__ == '__main__':
    main()
