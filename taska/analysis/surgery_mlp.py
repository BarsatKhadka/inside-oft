"""Surgical intervention on the MLP weights (W_in, W_out), not just W_E.

Hypothesis from probe.py: the MLP is where G compresses inputs and where M
doesn't. So the memorization circuit probably lives in the MLP, not the
embedding. If H2 holds anywhere in this model, the MLP is the place.

Three variants, same logic as surgery.py but applied to BOTH W_in and W_out
simultaneously (with the same k for both):

  1. truncate  — keep only M's top-k singular components in W_in and W_out
  2. project   — project M's W_in/W_out onto G's column/row spans
  3. substitute — replace M's top-k singular subspace in W_in/W_out with G's

Optional comparison: also run "full" surgery that modifies W_E + W_in + W_out
together. If H2 doesn't work even when we modify everything at once, the
memorization is distributed in a fundamentally non-spectral way.

Usage:
    python taska/analysis/surgery_mlp.py
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
D_MLP = 512
SEED = 0
K_VALUES = [1, 2, 5, 10, 20, 50, 100, 128]    # max k = 128 (rank limit for W_in/W_out)


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


def make_truncated(U_M, S_M, V_M, k):
    return U_M[:, :k] @ t.diag(S_M[:k]) @ V_M[:, :k].T


def make_projected(W_M, U_G, V_G, k):
    P_U = U_G[:, :k] @ U_G[:, :k].T
    P_V = V_G[:, :k] @ V_G[:, :k].T
    return P_U @ W_M @ P_V


def make_substituted(U_M, S_M, V_M, U_G, V_G, k):
    U_new = U_M.clone()
    V_new = V_M.clone()
    U_new[:, :k] = U_G[:, :k]
    V_new[:, :k] = V_G[:, :k]
    return U_new @ t.diag(S_M) @ V_new.T


def apply_surgery(M_model, variant, k, svd_M_in, svd_M_out, svd_G_in, svd_G_out,
                  W_M_in_orig, W_M_out_orig):
    """Apply the surgery to M's MLP weights in-place. Returns the original
    weights as a tuple so caller can restore."""
    U_M_in, S_M_in, V_M_in = svd_M_in
    U_M_out, S_M_out, V_M_out = svd_M_out
    U_G_in, S_G_in, V_G_in = svd_G_in
    U_G_out, S_G_out, V_G_out = svd_G_out

    if variant == 'truncate':
        new_W_in  = make_truncated(U_M_in,  S_M_in,  V_M_in,  k)
        new_W_out = make_truncated(U_M_out, S_M_out, V_M_out, k)
    elif variant == 'project':
        new_W_in  = make_projected(W_M_in_orig,  U_G_in,  V_G_in,  k)
        new_W_out = make_projected(W_M_out_orig, U_G_out, V_G_out, k)
    elif variant == 'substitute':
        new_W_in  = make_substituted(U_M_in,  S_M_in,  V_M_in,  U_G_in,  V_G_in,  k)
        new_W_out = make_substituted(U_M_out, S_M_out, V_M_out, U_G_out, V_G_out, k)
    else:
        raise ValueError(variant)

    with t.no_grad():
        M_model.blocks[0].mlp.W_in[:] = new_W_in
        M_model.blocks[0].mlp.W_out[:] = new_W_out


def main():
    M_model = load_model(HERE / 'checkpoints' / 'M' / 'final.pt')
    G_model = load_model(HERE / 'checkpoints' / 'G' / 'final.pt')

    # Snapshot M's MLP weights
    W_M_in_orig  = M_model.blocks[0].mlp.W_in.detach().clone()
    W_M_out_orig = M_model.blocks[0].mlp.W_out.detach().clone()

    # SVDs
    svd_M_in  = svd(M_model.blocks[0].mlp.W_in.detach())
    svd_M_out = svd(M_model.blocks[0].mlp.W_out.detach())
    svd_G_in  = svd(G_model.blocks[0].mlp.W_in.detach())
    svd_G_out = svd(G_model.blocks[0].mlp.W_out.detach())

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
    print(f'(Note: MLP rank cap is 128. W_in is 512x128, W_out is 128x512.)')
    print()

    def restore_M():
        with t.no_grad():
            M_model.blocks[0].mlp.W_in[:]  = W_M_in_orig
            M_model.blocks[0].mlp.W_out[:] = W_M_out_orig

    results = {'truncate': {}, 'project': {}, 'substitute': {}}

    for variant in results:
        print(f'=== {variant} (MLP only) ===')
        print(f'{"k":>4}  {"train_acc":>10}  {"test_acc":>10}')
        for k in K_VALUES:
            apply_surgery(M_model, variant, k,
                          svd_M_in, svd_M_out, svd_G_in, svd_G_out,
                          W_M_in_orig, W_M_out_orig)
            tr = evaluate(M_model, train_in, train_lab)
            te = evaluate(M_model, test_in,  test_lab)
            restore_M()
            results[variant][k] = {'train_acc': tr, 'test_acc': te}
            print(f'{k:>4}  {tr:>10.4f}  {te:>10.4f}')
        print()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'truncate': 'tab:blue', 'project': 'tab:orange', 'substitute': 'tab:green'}

    for ax, metric, title in zip(
        axes,
        ['train_acc', 'test_acc'],
        ['Training accuracy after MLP surgery', 'Test accuracy after MLP surgery'],
    ):
        baseline_M = M_train if metric == 'train_acc' else M_test
        baseline_G = G_train if metric == 'train_acc' else G_test
        for variant, d in results.items():
            ks = list(d.keys())
            ys = [d[k][metric] for k in ks]
            ax.plot(ks, ys, marker='o', label=variant, color=colors[variant])
        ax.axhline(baseline_M, linestyle=':', color='red',  alpha=0.6, label=f'M baseline ({baseline_M:.2f})')
        ax.axhline(baseline_G, linestyle=':', color='black', alpha=0.6, label=f'G baseline ({baseline_G:.2f})')
        ax.set_xlabel('k (rank of intervention on W_in AND W_out)')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle("H2 (MLP version): does modifying M's MLP weights recover generalization?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_surgery_mlp.png'
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
