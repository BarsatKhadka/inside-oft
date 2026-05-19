"""mech9: Cross-tier PCA of panel features.

PURE ANALYSIS. Pulls all models across all tiers, standardizes features,
runs PCA. Plots in 2D show whether models cluster by (tier, regime) or whether
the panel produces overlapping fingerprints.

Strongest possible figure: each regime has a unique fingerprint visible at
2D-projection level.

Weakest possible figure: models cluster by tier (architecture) only, with M/G
within a tier overlapping — would weaken the "panel discriminates regimes"
claim.

Either outcome is informative.
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
    'tier1_mlp_mnist',
    'tier1b_mlp_fashionmnist',
    'tier2_resnet18_cifar10',
    'tier3_resnet50_cifar100',
    'tier3b_vit_tiny_cifar10',
    'tier4_vit_small_cifar100',
    'tier5_charlm_shakespeare',
]

# Feature set: log-transform anything with potentially huge dynamic range
def get_features(r):
    """Extract the standardizable feature vector."""
    feats = {}
    feats['log_top_eig'] = np.log10(max(r.get('hessian_top_full', 1e-10), 1e-10))
    feats['log_neg_bot'] = np.log10(max(-r.get('hessian_bot_full', -1e-10), 1e-10))
    feats['cos_grad'] = r.get('cos_grad_train_test', 0)
    feats['log_grad_ratio'] = np.log10(max(r.get('grad_ratio_test_over_train', 1), 1))
    feats['mia'] = r.get('mia_loss_auc', 0.5)
    return feats


def load_all_records():
    records = []
    for tier in TIERS:
        p = RESULTS_BP3 / f'{tier}.json'
        if not p.exists(): continue
        d = json.load(open(p))
        if isinstance(d, dict):
            for mode in ('M', 'G'):
                if mode in d:
                    for r in d[mode]:
                        if 'error' in r: continue
                        feats = get_features(r)
                        if all(np.isfinite(v) for v in feats.values()):
                            records.append({
                                'tier': tier, 'mode': mode, 'seed': r.get('seed', -1),
                                **feats,
                            })
    return records


def pca_2d(X):
    """Center, standardize, compute first 2 principal components."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(0)
    sigma = X.std(0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    Xs = (X - mu) / sigma
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    pc = U[:, :2] * S[:2]
    # Variance explained
    total_var = (S ** 2).sum()
    var_explained = ((S ** 2)[:2] / total_var).tolist() if total_var > 0 else [0, 0]
    return pc, Vt[:2], var_explained


def main():
    records = load_all_records()
    if not records:
        print('No records found')
        return
    feat_names = ['log_top_eig', 'log_neg_bot', 'cos_grad', 'log_grad_ratio', 'mia']
    X = [[r[k] for k in feat_names] for r in records]
    pc, components, var_explained = pca_2d(X)
    print(f'Loaded {len(records)} models across {len(TIERS)} tiers')
    print(f'PCA variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}')
    print('\nFeature loadings on PC1, PC2:')
    for j, name in enumerate(feat_names):
        print(f'  {name:18s}: PC1={components[0, j]:+.3f}  PC2={components[1, j]:+.3f}')

    # Group records by (tier, mode), report cluster center + spread
    print('\nCluster centers (tier × mode):')
    clusters = {}
    for i, r in enumerate(records):
        key = f"{r['tier']}_{r['mode']}"
        clusters.setdefault(key, []).append(pc[i])
    for key, pts in sorted(clusters.items()):
        pts_arr = np.array(pts)
        center = pts_arr.mean(0)
        spread = pts_arr.std(0)
        print(f'  {key:45s} center=({center[0]:+.2f}, {center[1]:+.2f}) '
              f'spread=({spread[0]:.2f}, {spread[1]:.2f}), n={len(pts)}')

    # Within-cluster vs between-cluster distance: silhouette-like measure
    # For each point, distance to its own cluster center vs nearest other cluster
    sep_score = []
    cluster_centers = {k: np.array(v).mean(0) for k, v in clusters.items()}
    for i, r in enumerate(records):
        own_key = f"{r['tier']}_{r['mode']}"
        own_center = cluster_centers[own_key]
        dist_own = np.linalg.norm(pc[i] - own_center)
        dist_others = [np.linalg.norm(pc[i] - c) for k, c in cluster_centers.items() if k != own_key]
        dist_min_other = min(dist_others) if dist_others else float('inf')
        sep_score.append((dist_min_other - dist_own) / max(dist_own, dist_min_other, 1e-12))
    print(f'\nMean silhouette-like score: {np.mean(sep_score):+.3f} '
          f'(>0 = clusters separated; <0 = overlapping)')

    # Save
    out = {
        'n_records': len(records),
        'feature_names': feat_names,
        'variance_explained': var_explained,
        'pc1_loadings': components[0].tolist(),
        'pc2_loadings': components[1].tolist(),
        'cluster_centers': {k: cluster_centers[k].tolist() for k in cluster_centers},
        'silhouette_mean': float(np.mean(sep_score)),
        'per_model': [{
            'tier': r['tier'], 'mode': r['mode'], 'seed': r['seed'],
            'pc1': float(pc[i, 0]), 'pc2': float(pc[i, 1]),
            **{k: r[k] for k in feat_names},
        } for i, r in enumerate(records)],
    }
    with open(OUT_DIR / 'mech9_tier_clustering.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {OUT_DIR / "mech9_tier_clustering.json"}')


if __name__ == '__main__':
    main()
