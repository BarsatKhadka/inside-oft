"""Intruder-dimension analysis: do M's principal directions exist in G's?

For each of M's top-k singular vectors, find the best-matching singular vector
in G (by absolute cosine similarity). If the best match is close to 1, M's
direction "exists in G." If it's close to 0, M has a direction G doesn't --
that's an "intruder dimension."

We do this for BOTH:
    - right singular vectors (V): basis in token space (which TOKENS get
      treated the same)
    - left singular vectors (U): basis in residual-stream space (which
      EMBEDDING-SPACE directions are populated)

A clean result for the surgical intervention to work is:
    - M's top ~10 directions match G's top ~10 closely (similarity > 0.9)
    - M's directions 11+ have NO good match in G (similarity < 0.5)
    - This means: M shares G's "useful" directions and added ~40 "intruder"
      directions on top. Surgery target is unambiguous.

Less clean (still publishable, less dramatic):
    - M's top directions only partially overlap with G's. Then M and G
      learned different bases, and surgery requires more thought.

Usage:
    python taska/analysis/intruder.py
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


def best_match_similarities(vecs_target, vecs_reference):
    """For each row of vecs_target, return its max |cos similarity| to any row
    of vecs_reference, AND the index of that best match.

    Both inputs should already be L2-normalised rows.
    """
    # cosine = |inner product| since unit vectors
    sims = (vecs_target @ vecs_reference.T).abs()    # (n_target, n_ref)
    best_sim, best_idx = sims.max(dim=1)
    return best_sim, best_idx


def main():
    ckpts = {
        'G': HERE / 'checkpoints' / 'G' / 'final.pt',
        'M': HERE / 'checkpoints' / 'M' / 'final.pt',
    }

    # SVD both W_E matrices
    svd = {}
    for name, ckpt in ckpts.items():
        W = load_W_E(ckpt)
        U, S, Vt = t.linalg.svd(W, full_matrices=False)
        svd[name] = {'U': U, 'S': S, 'V': Vt.T}     # store V (columns are right singular vecs)
        print(f'{name}: U shape {tuple(U.shape)}  S shape {tuple(S.shape)}  V shape {tuple(Vt.T.shape)}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Compare M's directions against G's, for both U and V
    K_M = 60      # look at top 60 M dirs (covers everything important)
    K_G = 60      # search among G's top 60 dirs (covers everything important)
    INTRUDER_THRESHOLD = 0.5    # any M dir with max-similarity < this counts as intruder

    for ax, basis_name in zip(axes, ['U', 'V']):
        # Get top-K columns from M and G. U/V matrices have singular vectors as columns.
        m_vecs = svd['M'][basis_name][:, :K_M].T   # (K_M, dim)
        g_vecs = svd['G'][basis_name][:, :K_G].T   # (K_G, dim)

        # Normalize rows (should already be unit norm from SVD, but be safe)
        m_vecs = m_vecs / m_vecs.norm(dim=1, keepdim=True)
        g_vecs = g_vecs / g_vecs.norm(dim=1, keepdim=True)

        best_sim, best_idx = best_match_similarities(m_vecs, g_vecs)
        best_sim = best_sim.numpy()

        # Color bars: green if matches G well, red if intruder
        colors = ['tab:green' if s >= INTRUDER_THRESHOLD else 'tab:red'
                  for s in best_sim]

        bars = ax.bar(range(1, K_M + 1), best_sim, color=colors, width=0.85)
        ax.axhline(INTRUDER_THRESHOLD, color='black', linestyle=':', alpha=0.6,
                   label=f'intruder threshold = {INTRUDER_THRESHOLD}')
        ax.set_xlabel("M's singular vector index (1 = largest singular value)")
        ax.set_ylabel("max |cos similarity| to any G vector in top 60")
        space_name = 'residual-stream space' if basis_name == 'U' else 'token space'
        ax.set_title(f"Intruder analysis in {space_name} (basis {basis_name})")
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3, axis='y')

        n_matched = int((best_sim >= INTRUDER_THRESHOLD).sum())
        n_intruder = K_M - n_matched
        print(f'{basis_name}: {n_matched}/{K_M} M-dirs match G  |  {n_intruder} intruders')
        # also print the top-10 best matches
        print(f'   first 10 M-dirs: similarities = '
              f'{["%.2f" % s for s in best_sim[:10]]}')

    fig.suptitle("Intruder dimensions: do M's top singular vectors exist in G?")
    fig.tight_layout()

    out = HERE / 'results' / 'fig_intruder_WE.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
