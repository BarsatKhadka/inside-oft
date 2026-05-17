"""Time-resolved surgery: does spectral intervention work at early epochs
but fail at late epochs?

Prediction (from trajectory_basins.py): at epoch 1000 the barrier between
M_t and G_t is small (~5e-3). At epoch 50000 it's huge (~4.5). So surgery
SHOULD work at epoch 1000 and FAIL at epoch 50000.

For each epoch t in {1000, 4000, 8000, 11000, 20000, 50000}:
  Take M_t (the memorizing model at epoch t).
  Try three variants at k=11 (G's effective rank):
    truncate   — keep only M_t's top-11 SVD components in W_E, W_in, W_out
    project    — project M_t onto G_final's top-11 subspace in all three
    substitute — substitute G_final's top-11 directions into M_t

  Evaluate train and test accuracy.

Plot test accuracy vs epoch for each variant. If we see a downward trend
(early works, late doesn't), we've discovered a "time window" for surgical
removability.

Usage:
    python taska/analysis/surgery_over_time.py
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
K = 11  # G's effective rank
EPOCHS = [1000, 4000, 8000, 11000, 16000, 24000, 32000, 50000]


def load_state(ckpt):
    return t.load(ckpt, map_location='cpu', weights_only=True)['model']


def load_model_from_state(state):
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    model.load_state_dict(state)
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


def truncate(W, k):
    U, S, V = svd(W)
    k_use = min(k, S.shape[0])
    return U[:, :k_use] @ t.diag(S[:k_use]) @ V[:, :k_use].T


def project(W_M, W_G_target, k):
    U_G, _, V_G = svd(W_G_target)
    k_use = min(k, U_G.shape[1], V_G.shape[1])
    P_U = U_G[:, :k_use] @ U_G[:, :k_use].T
    P_V = V_G[:, :k_use] @ V_G[:, :k_use].T
    return P_U @ W_M @ P_V


def substitute(W_M, W_G_target, k):
    U_M, S_M, V_M = svd(W_M)
    U_G, _, V_G = svd(W_G_target)
    k_use = min(k, U_M.shape[1])
    U_new = U_M.clone()
    V_new = V_M.clone()
    U_new[:, :k_use] = U_G[:, :k_use]
    V_new[:, :k_use] = V_G[:, :k_use]
    return U_new @ t.diag(S_M) @ V_new.T


def apply_variant(state, variant, k, state_G_target):
    """Apply combined surgery to a copy of `state`, return modified state."""
    new_state = {key: v.clone() for key, v in state.items()}
    W_E_M = state['embed.W_E'][:, :P]
    W_in_M = state['blocks.0.mlp.W_in']
    W_out_M = state['blocks.0.mlp.W_out']
    W_E_G = state_G_target['embed.W_E'][:, :P]
    W_in_G = state_G_target['blocks.0.mlp.W_in']
    W_out_G = state_G_target['blocks.0.mlp.W_out']

    if variant == 'truncate':
        new_W_E   = truncate(W_E_M, k)
        new_W_in  = truncate(W_in_M, k)
        new_W_out = truncate(W_out_M, k)
    elif variant == 'project':
        new_W_E   = project(W_E_M, W_E_G, k)
        new_W_in  = project(W_in_M, W_in_G, k)
        new_W_out = project(W_out_M, W_out_G, k)
    elif variant == 'substitute':
        new_W_E   = substitute(W_E_M, W_E_G, k)
        new_W_in  = substitute(W_in_M, W_in_G, k)
        new_W_out = substitute(W_out_M, W_out_G, k)

    new_state['embed.W_E'] = state['embed.W_E'].clone()
    new_state['embed.W_E'][:, :P] = new_W_E
    new_state['blocks.0.mlp.W_in']  = new_W_in
    new_state['blocks.0.mlp.W_out'] = new_W_out
    return new_state


def main():
    M_dir = HERE / 'checkpoints' / 'M'
    G_dir = HERE / 'checkpoints' / 'G'

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')
    test_in,  test_lab  = to_tensors(test_pairs,  P, device='cpu')

    # Target G = final G (the fully grokked + cleaned model)
    s_G_final = load_state(G_dir / 'final.pt')

    results = {v: {} for v in ['truncate', 'project', 'substitute']}
    untreated = {}     # M_t with no surgery, for comparison

    print(f'k = {K} (G\'s effective rank)')
    print(f'\n{"epoch":>6}  {"variant":>11}  {"train_acc":>10}  {"test_acc":>10}  {"untreated_M_t_test":>20}')

    for ep in EPOCHS:
        ckpt_path = M_dir / 'final.pt' if ep == 50000 else M_dir / f'epoch_{ep}.pt'
        s_M_t = load_state(ckpt_path)

        # baseline: M_t with no surgery
        m_tr = evaluate(load_model_from_state(s_M_t), train_in, train_lab)
        m_te = evaluate(load_model_from_state(s_M_t), test_in,  test_lab)
        untreated[ep] = {'train_acc': m_tr, 'test_acc': m_te}

        for variant in results:
            s_modified = apply_variant(s_M_t, variant, K, s_G_final)
            tr = evaluate(load_model_from_state(s_modified), train_in, train_lab)
            te = evaluate(load_model_from_state(s_modified), test_in,  test_lab)
            results[variant][ep] = {'train_acc': tr, 'test_acc': te}
            print(f'{ep:>6}  {variant:>11}  {tr:>10.4f}  {te:>10.4f}  {m_te:>20.4f}')
        print()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'truncate': 'tab:blue', 'project': 'tab:orange', 'substitute': 'tab:green'}

    for ax, metric, title in zip(
        axes,
        ['train_acc', 'test_acc'],
        ['Train accuracy after surgery vs epoch', 'Test accuracy after surgery vs epoch'],
    ):
        for variant in results:
            ys = [results[variant][ep][metric] for ep in EPOCHS]
            ax.plot(EPOCHS, ys, marker='o', label=f'{variant}', color=colors[variant])
        # untreated M baseline
        ys_untreated = [untreated[ep][metric] for ep in EPOCHS]
        ax.plot(EPOCHS, ys_untreated, marker='s', label='M_t untreated (baseline)', color='gray', linestyle='--')
        # G final reference
        if metric == 'train_acc':
            ax.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='G_final = 1.0')
        else:
            ax.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='G_final = 1.0')
        ax.set_xlabel('training epoch of M_t')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axvline(10800, color='gray', linestyle='--', alpha=0.3)
        ax.text(10800, 0.05, 'G grokks', rotation=90, fontsize=8, alpha=0.7)

    fig.suptitle(f"Time-resolved surgery (k={K}): does early-epoch M recover via spectral intervention?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_surgery_over_time.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
