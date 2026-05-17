"""SVD comparison of W_E for G and M.

The basis-free version of the Fourier sanity check.

Expected:
    G's singular values drop sharply after the first ~10 -- because G has
    compressed its embedding into ~5 frequencies x (cos+sin) ~ 10 underlying
    directions. SVD should "discover" this concentration without us telling
    it that sine/cosine is the right basis.

    M's singular values are roughly uniform across all 113 -- because M's
    embedding has no preferred basis. Every direction carries similar weight.

Metrics reported:
    - effective rank = exp(entropy(sigma_normalized^2))
        "how many directions are meaningfully used."
        Big = spread spectrum. Small = concentrated spectrum.
    - stable rank = ||W||_F^2 / sigma_max^2
        Different formula, same idea.

This is the analysis we'll reuse on ResNet conv weights (Track B) and
Pythia attention matrices (Track C) -- no architecture-specific assumptions.

Usage:
    python taska/analysis/svd_compare.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # taska/
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch as t

from model import Transformer

P = 113
D_MODEL = 128


def load_W_E(ckpt_path):
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    return model.embed.W_E.detach()[:, :P]   # (d_model, p)


def effective_rank(sigma):
    """exp(Shannon entropy of normalized squared singular values).

    Interpretation: "an N-dimensional matrix whose energy is uniformly spread
    over k directions has effective rank ~ k."
    """
    p = sigma ** 2
    p = p / p.sum()
    p = p[p > 0]                # avoid log(0)
    entropy = -(p * t.log(p)).sum()
    return float(t.exp(entropy))


def stable_rank(W):
    """Frobenius norm squared / operator norm squared.

    Another concentration measure. Always <= true matrix rank.
    """
    sigma = t.linalg.svdvals(W)
    return float((sigma ** 2).sum() / sigma[0] ** 2)


def main():
    ckpts = {
        'G': HERE / 'checkpoints' / 'G' / 'final.pt',
        'M': HERE / 'checkpoints' / 'M' / 'final.pt',
    }

    spectra = {}
    for name, ckpt in ckpts.items():
        W = load_W_E(ckpt)
        sigma = t.linalg.svdvals(W)        # sorted descending
        spectra[name] = sigma
        er = effective_rank(sigma)
        sr = stable_rank(W)
        print(f'{name}:  n_singular_values={len(sigma)}  '
              f'sigma_max={sigma[0].item():.3f}  '
              f'sigma_min={sigma[-1].item():.3e}  '
              f'effective_rank={er:.1f}  '
              f'stable_rank={sr:.1f}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel 1: log-y spectrum overlay
    ax = axes[0]
    for name, sigma in spectra.items():
        ax.plot(range(1, len(sigma) + 1), sigma.numpy(), marker='o',
                markersize=3, label=name)
    ax.set_yscale('log')
    ax.set_xlabel('singular value index (1 = largest)')
    ax.set_ylabel('singular value (log scale)')
    ax.set_title('Singular value spectra of W_E')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: cumulative energy
    ax = axes[1]
    for name, sigma in spectra.items():
        energy = (sigma ** 2).numpy()
        cum = np.cumsum(energy) / energy.sum()
        ax.plot(range(1, len(cum) + 1), cum, marker='o', markersize=3,
                label=f'{name}')
    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.7,
               label='90% energy')
    ax.set_xlabel('top-k directions')
    ax.set_ylabel('fraction of total spectral energy captured')
    ax.set_title('Cumulative spectral energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('SVD of W_E -- G should concentrate, M should spread')
    fig.tight_layout()

    out = HERE / 'results' / 'fig_svd_WE.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')

    # Print a tiny summary table
    print()
    print('How many directions to capture 90% / 99% of spectral energy?')
    for name, sigma in spectra.items():
        energy = (sigma ** 2).numpy()
        cum = np.cumsum(energy) / energy.sum()
        k90 = int(np.argmax(cum >= 0.90) + 1)
        k99 = int(np.argmax(cum >= 0.99) + 1)
        print(f'  {name}: top {k90:3d} dirs for 90%  |  top {k99:3d} dirs for 99%')


if __name__ == '__main__':
    main()
