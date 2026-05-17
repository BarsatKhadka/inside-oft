"""Principal-angle comparison between M's top-k subspace and G's top-k subspace.

Per-vector cosine similarity (intruder.py) is a weak measure: two subspaces
can be IDENTICAL even if their basis vectors don't pair up, because basis
within a subspace is arbitrary.

The right measure is principal angles. Given two k-dim subspaces A, B in R^d:
    1. Take orthonormal bases for each (here: top-k columns of U or V from SVD).
    2. Compute SVD of A^T @ B. The singular values are cos(principal angles).
    3. All cos = 1  -> subspaces are identical.
       All cos = 0  -> subspaces are orthogonal.
       Mixed       -> partial overlap.

We compute this for several k values to see how the answer changes as we
include more directions.

Also report "energy capture": if we project G's top-k singular *vectors*
onto M's top-k subspace, what fraction of their squared norm survives?
This is a single scalar per k, summarizing the geometric overlap.

Usage:
    python taska/analysis/subspace.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
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
    return model.embed.W_E.detach()[:, :P]


def principal_angles_cos(A, B):
    """Cosines of principal angles between subspaces spanned by columns of A and B.

    A: (d, k_A) orthonormal columns
    B: (d, k_B) orthonormal columns
    Returns: cos values, length min(k_A, k_B), sorted descending (1 = aligned).
    """
    M = A.T @ B
    cos_vals = t.linalg.svdvals(M)   # in [0, 1]
    return cos_vals


def subspace_energy_capture(A_vecs, B_basis):
    """For each column of A_vecs, what fraction of its squared norm lies in
    the subspace spanned by columns of B_basis?

    A_vecs: (d, k_A)
    B_basis: (d, k_B) orthonormal
    Returns: per-vector capture fractions, length k_A.
    """
    proj = B_basis @ (B_basis.T @ A_vecs)   # (d, k_A)
    capture = (proj ** 2).sum(dim=0) / (A_vecs ** 2).sum(dim=0)
    return capture


def main():
    # Load and SVD both
    svd = {}
    for name in ['G', 'M']:
        W = load_W_E(HERE / 'checkpoints' / name / 'final.pt')
        U, S, Vt = t.linalg.svd(W, full_matrices=False)
        svd[name] = {'U': U, 'S': S, 'V': Vt.T}

    K_VALUES = [5, 11, 20, 30, 50, 80, 113]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, basis_name in zip(axes, ['U', 'V']):
        space_name = 'residual-stream space' if basis_name == 'U' else 'token space'
        print(f'\n=== {space_name} (basis {basis_name}) ===')
        print(f'{"k":>4}  {"mean cos":>10}  {"first 5 cos":>30}  {"G->M capture":>15}  {"M->G capture":>15}')

        mean_cos_by_k = []
        capture_G_to_M = []
        capture_M_to_G = []

        for k in K_VALUES:
            G_basis = svd['G'][basis_name][:, :k]
            M_basis = svd['M'][basis_name][:, :k]

            cos = principal_angles_cos(G_basis, M_basis)
            mean_cos = float(cos.mean())
            first_few = ['%.2f' % c for c in cos[:5].numpy()]

            cap_G_to_M = float(subspace_energy_capture(G_basis, M_basis).mean())
            cap_M_to_G = float(subspace_energy_capture(M_basis, G_basis).mean())

            mean_cos_by_k.append(mean_cos)
            capture_G_to_M.append(cap_G_to_M)
            capture_M_to_G.append(cap_M_to_G)

            print(f'{k:>4}  {mean_cos:>10.3f}  {str(first_few):>30}  '
                  f'{cap_G_to_M:>15.3f}  {cap_M_to_G:>15.3f}')

        ax.plot(K_VALUES, mean_cos_by_k, marker='o', label='mean cos(principal angle)')
        ax.plot(K_VALUES, capture_G_to_M, marker='s', label="G's top-k energy captured by M's top-k")
        ax.plot(K_VALUES, capture_M_to_G, marker='^', label="M's top-k energy captured by G's top-k")
        ax.axhline(1.0, color='green', linestyle=':', alpha=0.5, label='identical subspaces')

        # Random-baseline reference: if M's subspace were a random k-dim subspace
        # in R^d, expected energy capture = min(k, k)/d.
        d = 128 if basis_name == 'U' else 113
        random_cap = [min(k, k) / d for k in K_VALUES]
        ax.plot(K_VALUES, random_cap, color='gray', linestyle='--', alpha=0.6,
                label=f'random baseline (k/{d})')

        ax.set_xlabel('subspace dimension k (top-k singular directions)')
        ax.set_ylabel('overlap measure (1 = identical, 0 = orthogonal)')
        ax.set_title(f'Subspace overlap, M vs G, in {space_name}')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.suptitle("Principal-angle subspace overlap: are M's top dirs rotated G's, or genuinely different?")
    fig.tight_layout()

    out = HERE / 'results' / 'fig_subspace_WE.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
