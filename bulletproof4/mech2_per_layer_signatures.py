"""mech2: Per-layer signature decomposition.

PURE ANALYSIS - no training. Reads existing tier JSONs and pulls out the
per-layer effective ranks, then computes the M-vs-G gap per layer per tier.

Question this answers (Q3, Q7): does the M-vs-G rank signal live in
specific layers, or is it diffused? In ViTs, head.weight rank is identical
between M and G — but is the signal in earlier layers?
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_BP3 = HERE.parent / 'bulletproof3' / 'results'
OUT_DIR = HERE / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIERS_WITH_RANKS = [
    'tier0_modular_4L',
    'tier1_mlp_mnist',
    'tier1b_mlp_fashionmnist',
    'tier2_resnet18_cifar10',
    'tier3b_vit_tiny_cifar10',
    'tier4_vit_small_cifar100',
]


def main():
    for tier in TIERS_WITH_RANKS:
        p = RESULTS_BP3 / f'{tier}.json'
        if not p.exists():
            print(f'\n=== {tier}: not found ==='); continue
        data = json.load(open(p))
        if not isinstance(data, dict) or 'M' not in data:
            print(f'\n=== {tier}: not in dict form, skipping ==='); continue
        m_seeds = [r for r in data['M'] if 'error' not in r and 'ranks' in r]
        g_seeds = [r for r in data['G'] if 'error' not in r and 'ranks' in r]
        if not m_seeds or not g_seeds:
            print(f'\n=== {tier}: missing ranks ==='); continue
        print(f'\n=== {tier} | per-layer rank (M vs G, top 20 layers by gap) ===')

        # Collect all layer keys present in both
        common_layers = set(m_seeds[0]['ranks'].keys()) & set(g_seeds[0]['ranks'].keys())
        layer_stats = []
        for layer in common_layers:
            m_vals = [r['ranks'].get(layer) for r in m_seeds]
            g_vals = [r['ranks'].get(layer) for r in g_seeds]
            m_vals = [v for v in m_vals if v is not None]
            g_vals = [v for v in g_vals if v is not None]
            if not m_vals or not g_vals: continue
            m_mean = float(np.mean(m_vals)); g_mean = float(np.mean(g_vals))
            m_std = float(np.std(m_vals, ddof=1)) if len(m_vals) > 1 else 0.0
            g_std = float(np.std(g_vals, ddof=1)) if len(g_vals) > 1 else 0.0
            ratio = m_mean / max(g_mean, 1e-12)
            layer_stats.append({
                'layer': layer, 'm_mean': m_mean, 'g_mean': g_mean,
                'm_std': m_std, 'g_std': g_std,
                'ratio_m_over_g': ratio, 'abs_gap': abs(m_mean - g_mean),
            })

        # Sort by |gap| descending
        layer_stats.sort(key=lambda x: -x['abs_gap'])
        print(f'{"layer":50s} {"M":>10s} {"G":>10s} {"M/G":>8s}')
        for s in layer_stats[:20]:
            print(f'{s["layer"]:50s} {s["m_mean"]:10.2f} {s["g_mean"]:10.2f} {s["ratio_m_over_g"]:8.2f}')

        with open(OUT_DIR / f'mech2_perlayer_{tier}.json', 'w') as f:
            json.dump(layer_stats, f, indent=2)


if __name__ == '__main__':
    main()
