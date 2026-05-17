"""Combined surgery: modify W_E AND W_in AND W_out simultaneously.

This is the maximal "spectral" intervention on M. If H2 fails here too, we
have strong evidence that memorization isn't surgically removable by SVD-based
projection on individual layers -- it's encoded jointly across the network in
a way that resists this entire family of interventions.

Three variants, each applied to all three matrices at the same k:

  truncate    — keep only top-k singular components of W_E, W_in, W_out
  project     — project each onto G's top-k column/row spans
  substitute  — replace each's top-k singular subspace with G's

Wins:
  Test accuracy jumps from M's 6% toward G's 100% at some k. Means coordinated
  projection across all three matrices simultaneously kills memorization while
  exposing the partial Fourier computation. Headline result if it happens.

Losses:
  Test accuracy stays at 6% across all variants and all k. Means the spectral
  approach is dead in this setting; pivot to activation-level intervention,
  distillation, or Track B.

Usage:
    python taska/analysis/surgery_combined.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch as t

from data import gen_train_test, to_tensors
from model import Transformer

P = 113
D_MODEL = 128
SEED = 0
K_VALUES = [1, 2, 5, 8, 11, 15, 20, 30, 50, 80, 113]


def load_model(ckpt_path):
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    return model


def svd(W):
    U, S, Vt = t.linalg.svd(W, full_matrices=False)
    return U, S, Vt.T


@t.no_grad()
def evaluate(model, inputs, labels):
    logits = model(inputs)[:, -1, :]
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def make_truncated(U, S, V, k):
    return U[:, :k] @ t.diag(S[:k]) @ V[:, :k].T


def make_projected(W, U_G, V_G, k):
    P_U = U_G[:, :k] @ U_G[:, :k].T
    P_V = V_G[:, :k] @ V_G[:, :k].T
    return P_U @ W @ P_V


def make_substituted(U_M, S_M, V_M, U_G, V_G, k):
    U_new = U_M.clone()
    V_new = V_M.clone()
    U_new[:, :k] = U_G[:, :k]
    V_new[:, :k] = V_G[:, :k]
    return U_new @ t.diag(S_M) @ V_new.T


def apply_combined_surgery(M_model, variant, k, weights_M, weights_G, svds_M, svds_G):
    """Apply surgery to W_E (first P cols only), W_in, W_out simultaneously."""
    # Unpack
    W_E_M_full = weights_M['W_E']     # (128, 114) - we modify only the first P cols
    W_in_M  = weights_M['W_in']
    W_out_M = weights_M['W_out']

    (U_E_M, S_E_M, V_E_M)    = svds_M['W_E']
    (U_in_M, S_in_M, V_in_M) = svds_M['W_in']
    (U_out_M, S_out_M, V_out_M) = svds_M['W_out']

    (U_E_G, S_E_G, V_E_G)    = svds_G['W_E']
    (U_in_G, S_in_G, V_in_G) = svds_G['W_in']
    (U_out_G, S_out_G, V_out_G) = svds_G['W_out']

    # Cap k by each matrix's rank
    k_E   = min(k, S_E_M.shape[0])
    k_in  = min(k, S_in_M.shape[0])
    k_out = min(k, S_out_M.shape[0])

    if variant == 'truncate':
        new_W_E_113 = make_truncated(U_E_M,  S_E_M,  V_E_M,  k_E)
        new_W_in    = make_truncated(U_in_M, S_in_M, V_in_M, k_in)
        new_W_out   = make_truncated(U_out_M, S_out_M, V_out_M, k_out)
    elif variant == 'project':
        new_W_E_113 = make_projected(W_E_M_full[:, :P], U_E_G,  V_E_G,  k_E)
        new_W_in    = make_projected(W_in_M,             U_in_G, V_in_G, k_in)
        new_W_out   = make_projected(W_out_M,            U_out_G, V_out_G, k_out)
    elif variant == 'substitute':
        new_W_E_113 = make_substituted(U_E_M,  S_E_M,  V_E_M,  U_E_G,  V_E_G,  k_E)
        new_W_in    = make_substituted(U_in_M, S_in_M, V_in_M, U_in_G, V_in_G, k_in)
        new_W_out   = make_substituted(U_out_M, S_out_M, V_out_M, U_out_G, V_out_G, k_out)
    else:
        raise ValueError(variant)

    with t.no_grad():
        M_model.embed.W_E[:, :P]      = new_W_E_113
        M_model.blocks[0].mlp.W_in[:] = new_W_in
        M_model.blocks[0].mlp.W_out[:] = new_W_out


def main():
    M_model = load_model(HERE / 'checkpoints' / 'M' / 'final.pt')
    G_model = load_model(HERE / 'checkpoints' / 'G' / 'final.pt')

    weights_M = {
        'W_E':   M_model.embed.W_E.detach().clone(),
        'W_in':  M_model.blocks[0].mlp.W_in.detach().clone(),
        'W_out': M_model.blocks[0].mlp.W_out.detach().clone(),
    }
    weights_G = {
        'W_E':   G_model.embed.W_E.detach().clone(),
        'W_in':  G_model.blocks[0].mlp.W_in.detach().clone(),
        'W_out': G_model.blocks[0].mlp.W_out.detach().clone(),
    }

    svds_M = {
        'W_E':   svd(weights_M['W_E'][:, :P]),
        'W_in':  svd(weights_M['W_in']),
        'W_out': svd(weights_M['W_out']),
    }
    svds_G = {
        'W_E':   svd(weights_G['W_E'][:, :P]),
        'W_in':  svd(weights_G['W_in']),
        'W_out': svd(weights_G['W_out']),
    }

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')
    test_in,  test_lab  = to_tensors(test_pairs,  P, device='cpu')

    M_train = evaluate(M_model, train_in, train_lab)
    M_test  = evaluate(M_model, test_in,  test_lab)
    G_train = evaluate(G_model, train_in, train_lab)
    G_test  = evaluate(G_model, test_in,  test_lab)
    print(f'Baselines: M train {M_train:.4f}  M test {M_test:.4f}  '
          f'|  G train {G_train:.4f}  G test {G_test:.4f}')
    print()

    def restore_M():
        with t.no_grad():
            M_model.embed.W_E[:]          = weights_M['W_E']
            M_model.blocks[0].mlp.W_in[:] = weights_M['W_in']
            M_model.blocks[0].mlp.W_out[:] = weights_M['W_out']

    results = {'truncate': {}, 'project': {}, 'substitute': {}}

    for variant in results:
        print(f'=== {variant} (W_E + W_in + W_out together) ===')
        print(f'{"k":>4}  {"train_acc":>10}  {"test_acc":>10}')
        for k in K_VALUES:
            apply_combined_surgery(M_model, variant, k, weights_M, weights_G, svds_M, svds_G)
            tr = evaluate(M_model, train_in, train_lab)
            te = evaluate(M_model, test_in,  test_lab)
            restore_M()
            results[variant][k] = {'train_acc': tr, 'test_acc': te}
            print(f'{k:>4}  {tr:>10.4f}  {te:>10.4f}')
        print()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'truncate': 'tab:blue', 'project': 'tab:orange', 'substitute': 'tab:green'}

    for ax, metric, title in zip(
        axes,
        ['train_acc', 'test_acc'],
        ['Training accuracy after combined surgery', 'Test accuracy after combined surgery'],
    ):
        baseline_M = M_train if metric == 'train_acc' else M_test
        baseline_G = G_train if metric == 'train_acc' else G_test
        for variant, d in results.items():
            ks = list(d.keys())
            ys = [d[k][metric] for k in ks]
            ax.plot(ks, ys, marker='o', label=variant, color=colors[variant])
        ax.axhline(baseline_M, linestyle=':', color='red',  alpha=0.6, label=f'M baseline ({baseline_M:.2f})')
        ax.axhline(baseline_G, linestyle=':', color='black', alpha=0.6, label=f'G baseline ({baseline_G:.2f})')
        ax.set_xlabel('k (rank of intervention on all 3 matrices)')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle("H2 (combined): does modifying W_E + W_in + W_out together recover generalization?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_surgery_combined.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')

    print()
    print('Best test accuracy per variant:')
    for variant, d in results.items():
        best_k = max(d, key=lambda k: d[k]['test_acc'])
        print(f'  {variant:>10}: best test_acc = {d[best_k]["test_acc"]:.4f} at k={best_k}  '
              f'(train_acc at that k = {d[best_k]["train_acc"]:.4f})')


if __name__ == '__main__':
    main()
