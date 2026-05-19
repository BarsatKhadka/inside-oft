"""mech1: MIA AUC vs structural signature correlation analysis.

PURE ANALYSIS - no training. Reads existing tier JSONs and computes:
  - Within each tier, the Pearson correlation between MIA AUC and each
    structural signature (top eig, bot eig, |bot eig|, cos grad, rank gap)
    across seeds (M and G separately)
  - Cross-tier comparison: does MIA correlate with the same signatures
    in the same way in different architectures?

Question this answers (Q4): is MIA AUC just measuring the same underlying
property that the structural signatures are noisy proxies for? If MIA
correlates strongly with rank/Hessian within a regime, they capture the
same memorization fact. If MIA is uncorrelated with the others, MIA captures
something the others miss.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_BP3 = HERE.parent / 'bulletproof3' / 'results'
OUT_DIR = HERE / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIERS = {
    'tier0_modular_4L': {'wd_key': 'wd', 'm_val': 0.0, 'g_val': 1.0},
    'tier1_mlp_mnist': {'wd_key': 'wd', 'm_val': 0.0, 'g_val': 1e-3},
    'tier1b_mlp_fashionmnist': {'wd_key': 'wd', 'm_val': 0.0, 'g_val': 1e-3},
    'tier2_resnet18_cifar10': {'wd_key': 'mode', 'm_val': 'M', 'g_val': 'G'},
    'tier3b_vit_tiny_cifar10': {'wd_key': 'mode', 'm_val': 'M', 'g_val': 'G'},
    'tier4_vit_small_cifar100': {'wd_key': 'mode', 'm_val': 'M', 'g_val': 'G'},
}

SIG_KEYS = [
    'hessian_top_full',
    'hessian_bot_full',
    'cos_grad_train_test',
    'grad_ratio_test_over_train',
    'mia_loss_auc',
]


def pearson(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan'), len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def load_tier(name, conf):
    p = RESULTS_BP3 / f'{name}.json'
    if not p.exists():
        return None, None
    data = json.load(open(p))
    # Some tiers store as {M: [...], G: [...]}; others store list. Normalize.
    if isinstance(data, dict) and 'M' in data:
        m_seeds = [r for r in data['M'] if 'error' not in r]
        g_seeds = [r for r in data['G'] if 'error' not in r]
    elif isinstance(data, list):
        m_val, g_val = conf['m_val'], conf['g_val']
        wd_key = conf['wd_key']
        m_seeds = [r for r in data if r.get(wd_key) == m_val and 'error' not in r]
        g_seeds = [r for r in data if r.get(wd_key) == g_val and 'error' not in r]
    else:
        return None, None
    return m_seeds, g_seeds


def main():
    summary = {}
    for tier, conf in TIERS.items():
        m, g = load_tier(tier, conf)
        if m is None:
            print(f'\n=== {tier}: not found ==='); continue
        if not m or not g:
            print(f'\n=== {tier}: empty M ({len(m or [])}) or G ({len(g or [])}) ==='); continue
        print(f'\n=== {tier}: |M|={len(m)} |G|={len(g)} ===')

        tier_out = {'M': {}, 'G': {}, 'M_vs_G': {}}

        for label, seeds in [('M', m), ('G', g)]:
            # Within-regime correlations: MIA vs each structural signature
            mia = [r.get('mia_loss_auc') for r in seeds if r.get('mia_loss_auc') is not None]
            if len(mia) < 3:
                continue
            print(f'  {label} (n={len(mia)} seeds):')
            for sig in ['hessian_top_full', 'hessian_bot_full',
                        'cos_grad_train_test', 'grad_ratio_test_over_train']:
                vals = [r.get(sig) for r in seeds if r.get(sig) is not None]
                if len(vals) != len(mia):
                    continue
                r, n = pearson(vals, mia)
                tier_out[label][f'mia_vs_{sig}'] = r
                tail = ''
                if not np.isnan(r):
                    if abs(r) > 0.8: tail = '   strong'
                    elif abs(r) > 0.5: tail = '   moderate'
                print(f'    corr(MIA, {sig}) = {r:+.3f} (n={n}){tail}')

        # M vs G group separation: t-test-like effect size (Cohen's d) per signature
        print(f'  M vs G separation (Cohen\'s d, |d|>0.8 = large):')
        for sig in SIG_KEYS:
            m_vals = [r.get(sig) for r in m if r.get(sig) is not None]
            g_vals = [r.get(sig) for r in g if r.get(sig) is not None]
            if len(m_vals) < 2 or len(g_vals) < 2:
                continue
            m_vals = np.array(m_vals, dtype=float); g_vals = np.array(g_vals, dtype=float)
            m_vals = m_vals[np.isfinite(m_vals)]
            g_vals = g_vals[np.isfinite(g_vals)]
            if len(m_vals) < 2 or len(g_vals) < 2: continue
            pooled = np.sqrt(0.5 * (m_vals.var(ddof=1) + g_vals.var(ddof=1)))
            if pooled < 1e-12: continue
            d = (m_vals.mean() - g_vals.mean()) / pooled
            tier_out['M_vs_G'][sig] = {
                'cohen_d': float(d),
                'm_mean': float(m_vals.mean()), 'g_mean': float(g_vals.mean()),
                'm_std': float(m_vals.std(ddof=1)), 'g_std': float(g_vals.std(ddof=1)),
            }
            print(f'    {sig}: M={m_vals.mean():.3g}+-{m_vals.std(ddof=1):.2g}, '
                  f'G={g_vals.mean():.3g}+-{g_vals.std(ddof=1):.2g}, d={d:+.2f}')

        summary[tier] = tier_out

    with open(OUT_DIR / 'mech1_mia_correlation.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to {OUT_DIR / "mech1_mia_correlation.json"}')


if __name__ == '__main__':
    main()
