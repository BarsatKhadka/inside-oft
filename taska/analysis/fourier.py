"""Fourier analysis of W_E for G and M. Sanity check the analysis pipeline.

Expected for G (final checkpoint):
    A handful of "key frequencies" with much larger Fourier power than the
    rest. Nanda et al. reported k in {14, 35, 41, 42, 52} for one run; the
    exact set varies by seed, but the *sparsity* should be very clear.

Expected for M:
    Roughly uniform Fourier power across all frequencies. No sharp spikes.
    Means M's W_E has no cyclic structure -- it just encodes 113 unrelated
    embeddings, not 113 points on a learned circle.

This sanity-checks the pipeline: if we recover Nanda's qualitative finding
on G, our SVD / probe code downstream can be trusted to mean what we think.

Usage: 
    python taska/analysis/fourier.py
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


def make_fourier_basis(p):
    """Return an orthonormal Fourier basis as a (p, p) matrix.

    Rows are: constant, cos(1), sin(1), cos(2), sin(2), ..., cos(p//2), sin(p//2).
    Each row is L2-normalised so the basis is orthonormal.
    """
    basis = [t.ones(p) / np.sqrt(p)]
    names = ['const']
    for k in range(1, p // 2 + 1):
        c = t.cos(2 * t.pi * t.arange(p) * k / p)
        s = t.sin(2 * t.pi * t.arange(p) * k / p)
        basis.append(c / c.norm())
        basis.append(s / s.norm())
        names.append(f'cos {k}')
        names.append(f'sin {k}')
    return t.stack(basis, dim=0), names


def load_W_E(ckpt_path):
    """Load just W_E (the embedding matrix) from a checkpoint."""
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    # W_E shape: (d_model, d_vocab) where d_vocab = p + 1.
    # Take only the first p columns (drop the "=" token at index p).
    return model.embed.W_E.detach()[:, :P]   # (d_model, p)


def fourier_power_per_freq(W_E, basis):
    """For each frequency k, compute total power = sum_neuron (<W_E[n], cos_k>^2 + <W_E[n], sin_k>^2).

    Returns:
        powers: shape (p//2 + 1,) -- one number per frequency (including const = freq 0)
    """
    # W_E: (d_model, p)
    # basis: (p, p)  -- each ROW is a basis vector
    coeffs = W_E @ basis.T   # (d_model, p) -- inner products of each neuron with each basis vector
    # Square and sum across neurons
    power_per_basis = (coeffs ** 2).sum(dim=0)  # (p,) one per basis vector
    # Now collapse cos+sin pairs into single-frequency power
    p_dim = basis.shape[0]
    n_freqs = (p_dim - 1) // 2 + 1   # const + p//2 sin/cos pairs
    powers = t.zeros(n_freqs)
    powers[0] = power_per_basis[0]   # const
    for k in range(1, n_freqs):
        powers[k] = power_per_basis[2 * k - 1] + power_per_basis[2 * k]
    return powers


def main():
    basis, _ = make_fourier_basis(P)

    ckpts = {
        'G': HERE / 'checkpoints' / 'G' / 'final.pt',
        'M': HERE / 'checkpoints' / 'M' / 'final.pt',
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, (name, ckpt) in zip(axes, ckpts.items()):
        W_E = load_W_E(ckpt)
        powers = fourier_power_per_freq(W_E, basis).numpy()
        freqs = np.arange(len(powers))

        ax.bar(freqs, powers, width=0.8)
        ax.set_xlabel('frequency k')
        ax.set_ylabel('total Fourier power across neurons')
        ax.set_title(f'{name} -- W_E Fourier decomposition')
        ax.set_xlim(-0.5, len(powers) - 0.5)

        # report key frequencies: any whose power is > 4x the median
        median = np.median(powers[1:])     # ignore the const
        key = [k for k in range(1, len(powers)) if powers[k] > 4 * median]
        print(f'{name}: median non-const power = {median:.3f}, '
              f'key freqs (>4x median) = {key}')

    fig.suptitle('Fourier decomposition of W_E -- G should be sparse, M should be flat')
    fig.tight_layout()

    out = HERE / 'results' / 'fig_fourier_WE.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
