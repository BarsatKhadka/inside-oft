"""Surgical projection experiment (H2 test).

Three variants of "modify W_E of M, then evaluate":

  1. truncate  — keep only M's top-k singular components of W_E.
                 Tests: "is M's tail (the 50 'extra' directions) what does the
                 memorization? Drop it -> does M generalize?"

  2. project   — project M's W_E onto G's top-k column and row spans.
                 Tests: "if we force M to live in G's subspace, does it
                 generalize? Anything outside G's span gets killed."

  3. substitute — replace M's top-k singular subspace with G's, keep M's
                  magnitudes and tail.
                  Tests: "is the residual (memorization circuit) in the
                  orthogonal complement? Most invasive intervention."

For each (variant, k), measure:
  - train accuracy of the surgically-modified model
  - test accuracy of same

The headline plot is test accuracy vs k for each variant.

Expected outcomes:
  - If H2 works: at some k, test_acc jumps from M's 6% toward G's 100%.
    Either with or without a train_acc cost — both interesting.
  - If H2 fails: nothing happens, or train collapses without test recovering.

Usage:
    python taska/analysis/surgery.py
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
K_VALUES = [1, 2, 3, 5, 8, 11, 15, 20, 30, 50, 80, 113]


def load_model(ckpt_path):
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    return model


def svd_W_E(model):
    """SVD of W_E restricted to the 113 number-token columns.
    Returns (U, S, V) where V columns are right singular vectors."""
    W = model.embed.W_E.detach()[:, :P]
    U, S, Vt = t.linalg.svd(W, full_matrices=False)
    return U, S, Vt.T


@t.no_grad()
def evaluate(model, inputs, labels):
    """Top-1 accuracy at position 2."""
    logits = model(inputs)[:, -1, :]
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def make_truncated_W(U_M, S_M, V_M, k):
    """Top-k SVD reconstruction of M's W_E."""
    return U_M[:, :k] @ t.diag(S_M[:k]) @ V_M[:, :k].T


def make_projected_W(W_M_E, U_G, V_G, k):
    """Project M's W_E onto G's top-k column and row spans."""
    P_U = U_G[:, :k] @ U_G[:, :k].T
    P_V = V_G[:, :k] @ V_G[:, :k].T
    return P_U @ W_M_E @ P_V


def make_substituted_W(U_M, S_M, V_M, U_G, V_G, k):
    """Replace M's top-k singular subspace with G's, keep M's singular values."""
    U_new = U_M.clone()
    V_new = V_M.clone()
    U_new[:, :k] = U_G[:, :k]
    V_new[:, :k] = V_G[:, :k]
    return U_new @ t.diag(S_M) @ V_new.T


def patch_W_E(model, new_W_E_113):
    """Replace the first P columns of model.embed.W_E in-place. Leaves the
    '=' embedding (column 113) untouched."""
    with t.no_grad():
        model.embed.W_E[:, :P] = new_W_E_113


def main():
    # Load both models, snapshot original M W_E so we can restore between sweeps
    M_model = load_model(HERE / 'checkpoints' / 'M' / 'final.pt')
    G_model = load_model(HERE / 'checkpoints' / 'G' / 'final.pt')

    W_M_E_orig = M_model.embed.W_E.detach().clone()
    U_M, S_M, V_M = svd_W_E(M_model)
    U_G, S_G, V_G = svd_W_E(G_model)
    W_M_E_113 = W_M_E_orig[:, :P]

    # Data
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')
    test_in,  test_lab  = to_tensors(test_pairs,  P, device='cpu')

    # Baselines (no surgery)
    M_train = evaluate(M_model, train_in, train_lab)
    M_test  = evaluate(M_model, test_in,  test_lab)
    G_train = evaluate(G_model, train_in, train_lab)
    G_test  = evaluate(G_model, test_in,  test_lab)
    print(f'Baselines: M train {M_train:.4f}  M test {M_test:.4f}  '
          f'|  G train {G_train:.4f}  G test {G_test:.4f}')
    print()

    results = {'truncate': {}, 'project': {}, 'substitute': {}}

    for variant in results:
        print(f'=== {variant} ===')
        print(f'{"k":>4}  {"train_acc":>10}  {"test_acc":>10}')
        for k in K_VALUES:
            if variant == 'truncate':
                new_W = make_truncated_W(U_M, S_M, V_M, k)
            elif variant == 'project':
                new_W = make_projected_W(W_M_E_113, U_G, V_G, k)
            elif variant == 'substitute':
                new_W = make_substituted_W(U_M, S_M, V_M, U_G, V_G, k)

            # Patch into the model, evaluate, restore
            patch_W_E(M_model, new_W)
            tr_acc = evaluate(M_model, train_in, train_lab)
            te_acc = evaluate(M_model, test_in,  test_lab)
            patch_W_E(M_model, W_M_E_orig[:, :P])    # restore

            results[variant][k] = {'train_acc': tr_acc, 'test_acc': te_acc}
            print(f'{k:>4}  {tr_acc:>10.4f}  {te_acc:>10.4f}')
        print()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'truncate': 'tab:blue', 'project': 'tab:orange', 'substitute': 'tab:green'}

    for ax, metric, title in zip(
        axes,
        ['train_acc', 'test_acc'],
        ['Training accuracy after surgery', 'Test accuracy after surgery'],
    ):
        for variant, d in results.items():
            ks = list(d.keys())
            ys = [d[k][metric] for k in ks]
            ax.plot(ks, ys, marker='o', label=variant, color=colors[variant])
        # Reference lines
        ax.axhline(M_train if metric == 'train_acc' else M_test,
                   linestyle=':', color='red', alpha=0.6,
                   label=f'M baseline ({M_train if metric == "train_acc" else M_test:.2f})')
        ax.axhline(G_train if metric == 'train_acc' else G_test,
                   linestyle=':', color='black', alpha=0.6,
                   label=f'G baseline ({G_train if metric == "train_acc" else G_test:.2f})')
        ax.set_xlabel('k (rank of intervention)')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle("H2 test: does modifying M's W_E recover generalization?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_surgery.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')

    # Best-test-accuracy summary
    print()
    print('Best test accuracy per variant:')
    for variant, d in results.items():
        best_k = max(d, key=lambda k: d[k]['test_acc'])
        print(f'  {variant:>10}: best test_acc = {d[best_k]["test_acc"]:.4f} at k={best_k}  '
              f'(train_acc at that k = {d[best_k]["train_acc"]:.4f})')


if __name__ == '__main__':
    main()
