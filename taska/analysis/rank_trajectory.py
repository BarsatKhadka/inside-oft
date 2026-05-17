"""How does effective rank of W_E, W_in, W_out evolve over training, for M vs G?

If G's low-rank structure emerges DURING training (and M's stays high-rank),
that's a real, novel finding: weight decay causes spectral compression, not
just at convergence but as a continuous process throughout training.

For each checkpoint epoch in both M and G:
  - Compute effective rank of W_E (first 113 cols), W_in, W_out.
  - Plot rank vs epoch for each matrix, both models.

Predicted shapes:
  - G: rank decreases (or stays low) over training, especially through grokking.
  - M: rank stays high (or grows) throughout.

Bonus: also plot stable rank, which is more sensitive to the largest singular
value relative to the bulk.

Usage:
    python taska/analysis/rank_trajectory.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch as t

P = 113


def load_state(ckpt):
    return t.load(ckpt, map_location='cpu', weights_only=True)['model']


def effective_rank(sigma):
    p = sigma ** 2
    p = p / p.sum()
    p = p[p > 0]
    H = -(p * t.log(p)).sum()
    return float(t.exp(H))


def stable_rank(sigma):
    return float((sigma ** 2).sum() / sigma[0] ** 2)


def analyze_matrix(W):
    sigma = t.linalg.svdvals(W)
    return effective_rank(sigma), stable_rank(sigma)


def collect_ckpts(dir_):
    ckpts = sorted(int(p.stem.split('_')[1]) for p in dir_.glob('epoch_*.pt'))
    return [0] + ckpts + [50000]


def state_for(dir_, ep):
    if ep == 0:
        return load_state(dir_ / 'init.pt')
    if ep == 50000:
        return load_state(dir_ / 'final.pt')
    return load_state(dir_ / f'epoch_{ep}.pt')


def main():
    G_dir = HERE / 'checkpoints' / 'G'
    M_dir = HERE / 'checkpoints' / 'M'

    # Subsample for speed
    G_ckpts = collect_ckpts(G_dir)
    M_ckpts = collect_ckpts(M_dir)
    common = sorted(set(G_ckpts) & set(M_ckpts))
    epochs = [e for e in common if e == 0 or e == 50000 or e in {1000, 2000, 4000, 6000, 8000, 10000, 11000, 12000, 16000, 20000, 28000, 36000, 44000}]
    epochs = sorted(set(epochs))
    print(f'Evaluating at {len(epochs)} epochs: {epochs}')

    matrices = ['W_E', 'W_in', 'W_out']
    metrics = {'eff_rank': {m: {'G': [], 'M': []} for m in matrices},
               'stable_rank': {m: {'G': [], 'M': []} for m in matrices}}

    for ep in epochs:
        s_G = state_for(G_dir, ep)
        s_M = state_for(M_dir, ep)

        for name, key in [('W_E', 'embed.W_E'), ('W_in', 'blocks.0.mlp.W_in'),
                          ('W_out', 'blocks.0.mlp.W_out')]:
            W_G = s_G[key]
            W_M = s_M[key]
            if name == 'W_E':
                W_G = W_G[:, :P]
                W_M = W_M[:, :P]
            er_G, sr_G = analyze_matrix(W_G)
            er_M, sr_M = analyze_matrix(W_M)
            metrics['eff_rank'][name]['G'].append(er_G)
            metrics['eff_rank'][name]['M'].append(er_M)
            metrics['stable_rank'][name]['G'].append(sr_G)
            metrics['stable_rank'][name]['M'].append(sr_M)

    # Print table
    print(f'\n{"epoch":>6}  ', end='')
    for m in matrices:
        print(f'{m + "_G_er":>10}  {m + "_M_er":>10}  ', end='')
    print()
    for i, ep in enumerate(epochs):
        print(f'{ep:>6}  ', end='')
        for m in matrices:
            print(f'{metrics["eff_rank"][m]["G"][i]:>10.2f}  '
                  f'{metrics["eff_rank"][m]["M"][i]:>10.2f}  ', end='')
        print()

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, mat in enumerate(matrices):
        ax = axes[0, col]
        ax.plot(epochs, metrics['eff_rank'][mat]['G'], marker='o', label='G', color='tab:blue')
        ax.plot(epochs, metrics['eff_rank'][mat]['M'], marker='s', label='M', color='tab:red')
        ax.axvline(10800, color='gray', linestyle='--', alpha=0.4)
        ax.set_xlabel('epoch')
        ax.set_ylabel('effective rank')
        ax.set_title(f'{mat} effective rank')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(epochs, metrics['stable_rank'][mat]['G'], marker='o', label='G', color='tab:blue')
        ax.plot(epochs, metrics['stable_rank'][mat]['M'], marker='s', label='M', color='tab:red')
        ax.axvline(10800, color='gray', linestyle='--', alpha=0.4)
        ax.set_xlabel('epoch')
        ax.set_ylabel('stable rank')
        ax.set_title(f'{mat} stable rank')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Rank trajectories: when does G become low-rank and M stay high?')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_rank_trajectory.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
