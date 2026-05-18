"""bp11: Fine-grained WD threshold characterization.

WD in {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6} x 10 seeds.
Sigmoid fit to grok probability gives threshold location + transition width.
"""
import json
from pathlib import Path
import numpy as np
import torch as t

from _common import (HERE, effective_rank, train_one)

NUM_SEEDS = 10
NUM_EPOCHS = 20000
WD_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]


def sigmoid(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def fit_sigmoid(xs, ps):
    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(sigmoid, xs, ps, p0=[xs[len(xs)//2], 10.0], maxfev=10000)
        return popt
    except Exception:
        return None


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = []
    for wd in WD_VALUES:
        for seed in range(NUM_SEEDS):
            print(f'\n--- wd={wd} seed={seed} ---')
            model, meta, (tr_in, tr_lab, te_in, te_lab) = train_one(
                seed=seed, wd=wd, num_epochs=NUM_EPOCHS, device=device)
            entry = {
                'wd': wd, 'seed': seed,
                'final_test_acc': meta['final_test_acc'],
                'final_train_acc': meta['final_train_acc'],
                'grok_epoch': meta['grok_epoch'],
                'grokked': meta['final_test_acc'] >= 0.95,
                'rank_W_out': effective_rank(model.blocks[0].mlp.W_out),
            }
            results.append(entry)
            print(f'  test={meta["final_test_acc"]:.4f}, grokked={entry["grokked"]}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp11_threshold_fine.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Per-WD grok probability
    print('\n=== Grok probability by WD ===')
    xs, ps = [], []
    for wd in WD_VALUES:
        cell = [r for r in results if r['wd'] == wd]
        p_grok = np.mean([r['grokked'] for r in cell])
        xs.append(wd); ps.append(p_grok)
        print(f'  wd={wd}: p_grok = {p_grok:.2f} ({sum(r["grokked"] for r in cell)}/{len(cell)})')
    fit = fit_sigmoid(np.array(xs), np.array(ps))
    if fit is not None:
        x0, k = fit
        print(f'\nSigmoid fit: threshold = {x0:.4f}, sharpness k = {k:.2f}')
        with open(HERE / 'results' / 'bp11_threshold_fit.json', 'w') as f:
            json.dump({'threshold': float(x0), 'sharpness': float(k),
                       'xs': xs, 'p_grok': ps}, f, indent=2)


if __name__ == '__main__':
    main()
