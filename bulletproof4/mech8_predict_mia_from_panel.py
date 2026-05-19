"""mech8: Does the structural panel ADD INFORMATION beyond MIA AUC?

PURE ANALYSIS — no training. For every (tier, regime) cell, fit a regression
that predicts MIA AUC from the structural panel features (top eig, bot eig,
cos grad, grad ratio, rank). Report R^2 per cell.

Three regimes of result:
  1. High R^2 (>0.7): the panel "explains" MIA. Panel is a structural
     instantiation of the same property MIA reads directly.
  2. Low R^2 (<0.3): the panel captures something MIA doesn't. Distinct signal.
  3. Intermediate: panel is partially redundant with MIA.

This directly answers the question: "is the panel just a noisy proxy for MIA, or
does it carry independent information?"
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_BP3 = HERE.parent / 'bulletproof3' / 'results'
OUT_DIR = HERE / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIERS = [
    'tier0_modular_4L',
    'tier2_resnet18_cifar10',
    'tier3_resnet50_cifar100',
    'tier3b_vit_tiny_cifar10',
    'tier4_vit_small_cifar100',
    'tier5_charlm_shakespeare',
]

FEATURES = ['hessian_top_full', 'hessian_bot_full',
             'cos_grad_train_test', 'grad_ratio_test_over_train']


def load_tier_records(name):
    p = RESULTS_BP3 / f'{name}.json'
    if not p.exists(): return []
    d = json.load(open(p))
    recs = []
    if isinstance(d, dict):
        for mode in ('M', 'G'):
            if mode in d:
                for r in d[mode]:
                    if 'error' in r: continue
                    if all(r.get(k) is not None for k in FEATURES + ['mia_loss_auc']):
                        recs.append({**r, '_mode': mode, '_tier': name})
    return recs


def rsq(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-12: return float('nan')
    return float(1 - ss_res / ss_tot)


def fit_linear_predict(X, y):
    """OLS via numpy. Adds intercept."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(X) < 2: return None, float('nan')
    X_aug = np.column_stack([X, np.ones(len(X))])
    try:
        beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    except Exception:
        return None, float('nan')
    y_pred = X_aug @ beta
    return beta, rsq(y, y_pred)


def main():
    summary = {}
    print('Per-tier per-regime: predict MIA AUC from structural panel features')
    print('('+', '.join(FEATURES)+')')
    print()
    for tier in TIERS:
        recs = load_tier_records(tier)
        if not recs: continue
        for mode in ('M', 'G', 'all'):
            if mode == 'all':
                sub = recs
            else:
                sub = [r for r in recs if r['_mode'] == mode]
            if len(sub) < 3: continue
            X = [[r[k] for k in FEATURES] for r in sub]
            y = [r['mia_loss_auc'] for r in sub]
            beta, r2 = fit_linear_predict(X, y)
            summary.setdefault(tier, {})[mode] = {
                'n': len(sub),
                'features': FEATURES,
                'r2': r2,
                'beta_per_feature': beta[:-1].tolist() if beta is not None else None,
                'intercept': float(beta[-1]) if beta is not None else None,
                'y_mean': float(np.mean(y)),
                'y_std': float(np.std(y)),
            }
            print(f'{tier:35s} {mode:4s}: n={len(sub):2d}  R^2={r2:+.3f}  '
                  f'MIA mean={np.mean(y):.3f} ± {np.std(y):.3f}')
        # Also pooled M+G
        print()
    # Also: pooled across all tiers (with tier identity as one-hot) to see if
    # ANY linear combination of panel features predicts MIA universally
    print('\nPooled across all tiers (n total):')
    all_recs = sum([load_tier_records(t) for t in TIERS], [])
    if all_recs:
        X = [[r[k] for k in FEATURES] for r in all_recs]
        y = [r['mia_loss_auc'] for r in all_recs]
        beta, r2 = fit_linear_predict(X, y)
        summary['pooled'] = {
            'n': len(all_recs), 'r2': r2,
            'features': FEATURES,
            'beta_per_feature': beta[:-1].tolist() if beta is not None else None,
        }
        print(f'  pooled all tiers: n={len(all_recs)} R^2={r2:+.3f}')
    with open(OUT_DIR / 'mech8_predict_mia.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to {OUT_DIR / "mech8_predict_mia.json"}')


if __name__ == '__main__':
    main()
