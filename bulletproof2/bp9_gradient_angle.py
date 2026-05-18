"""bp9: Gradient angle - reframed saddle claim.

Compute cos(grad_L_train(M), grad_L_test(M)) and same at G, across 10 seeds.

Prediction:
  - M: near-orthogonal or negative cosine (train wants one thing, test wants another)
  - G: strongly positive cosine (train and test gradients aligned)

This replaces "M is a saddle" with the precise geometric statement:
"M's training-loss and test-loss gradients are conflicting."
"""
import json
from pathlib import Path
import numpy as np
import torch as t

from _common import (HERE, P, train_one, flat_grad)

NUM_SEEDS = 10
NUM_EPOCHS = 20000


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    for seed in range(NUM_SEEDS):
        for label, wd in [('M', 0.0), ('G', 1.0)]:
            print(f'\n=== {label} seed={seed} ===')
            model, meta, (tr_in, tr_lab, te_in, te_lab) = train_one(
                seed=seed, wd=wd, num_epochs=NUM_EPOCHS, device=device)
            for p in model.parameters():
                p.requires_grad_(True)
            g_tr = flat_grad(model, tr_in, tr_lab)
            g_te = flat_grad(model, te_in, te_lab)
            cos = float(t.dot(g_tr, g_te) / (g_tr.norm() * g_te.norm() + 1e-12))
            angle_deg = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            entry = {
                'seed': seed, 'wd': wd, 'final_test_acc': meta['final_test_acc'],
                'cos_grad_train_test': cos,
                'angle_deg': angle_deg,
                'grad_train_norm': float(g_tr.norm()),
                'grad_test_norm': float(g_te.norm()),
                'grad_ratio': float(g_te.norm() / (g_tr.norm() + 1e-12)),
            }
            results[label].append(entry)
            print(f'  cos(g_tr, g_te) = {cos:.4f}  (angle = {angle_deg:.1f} deg), '
                  f'ratio = {entry["grad_ratio"]:.2e}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp9_gradient_angle.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Summary
    print('\n=== Summary ===')
    for label in ['M', 'G']:
        cs = np.array([r['cos_grad_train_test'] for r in results[label]])
        print(f'  {label}: cos mean={cs.mean():.4f}, std={cs.std():.4f}, '
              f'95% CI=[{np.percentile(cs, 2.5):.4f}, {np.percentile(cs, 97.5):.4f}]')


if __name__ == '__main__':
    main()
