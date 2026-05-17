"""E. Information-theoretic accounting: how much capacity does M actually use?

M has 226,816 parameters. It memorized 3830 examples * log2(113) ~= 26 kb of
label information. Where does the rest of M's capacity go?

Test 1: Low-rank approximation. For each weight matrix, replace with its
rank-k SVD truncation. Sweep k. At what rank does training accuracy
collapse? G should collapse at ~rank 11, M should collapse only at very
high rank (close to full).

Test 2: Quantization. Round each weight to the nearest multiple of some
quantization step. As quantization gets coarser, does M lose memorization?
If M survives heavy quantization, its memorization is not encoded in fine
weight precision; it's encoded in the rough structure.

Together: shows how much of M's weight capacity is "actually used" for the
memorization, vs how much is redundant.

If M can be heavily compressed without losing memorization, that means
3830 examples * 26 kb of info = ~100 kb of essential info, and the rest
of M's 7 Mb of weight capacity is redundancy. Compression ratios:
information-theoretic vs effective.

Usage:
    python taska/analysis/capacity_test.py
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


def load_model(ckpt):
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt, map_location='cpu', weights_only=True)['model']
    model.load_state_dict(state)
    model.eval()
    return model


def truncate(W, k):
    U, S, Vt = t.linalg.svd(W, full_matrices=False)
    k_use = min(k, S.shape[0])
    return U[:, :k_use] @ t.diag(S[:k_use]) @ Vt[:k_use, :]


def quantize_uniform(W, n_levels):
    """Quantize W to n_levels uniformly spaced values between W.min and W.max."""
    if n_levels >= 256:
        return W
    lo, hi = W.min().item(), W.max().item()
    if hi - lo < 1e-9:
        return W
    step = (hi - lo) / (n_levels - 1)
    q = t.round((W - lo) / step) * step + lo
    return q


@t.no_grad()
def eval_acc(model, inputs, labels):
    logits = model(inputs)[:, -1, :]
    return (logits.argmax(dim=-1) == labels).float().mean().item()


@t.no_grad()
def apply_to_all_matrices(model_state, op):
    """Apply op (function on tensor) to W_E (first 113 cols), W_in, W_out, and attention matrices."""
    new_state = {k: v.clone() for k, v in model_state.items()}
    new_state['embed.W_E'][:, :P] = op(model_state['embed.W_E'][:, :P])
    new_state['blocks.0.mlp.W_in']  = op(model_state['blocks.0.mlp.W_in'])
    new_state['blocks.0.mlp.W_out'] = op(model_state['blocks.0.mlp.W_out'])
    # Skip attention for now -- shapes are awkward (heads, d_head, d_model)
    return new_state


def main():
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')
    test_in,  test_lab  = to_tensors(test_pairs,  P, device='cpu')

    # =============== Test 1: low-rank truncation =================
    print('=== TEST 1: Low-rank truncation of W_E, W_in, W_out ===')
    K_VALUES = [1, 2, 5, 10, 20, 50, 100, 128]
    results_rank = {'M': {}, 'G': {}}
    for name in ['M', 'G']:
        state = t.load(HERE / 'checkpoints' / name / 'final.pt', map_location='cpu', weights_only=True)['model']
        print(f'\n{name}:')
        for k in K_VALUES:
            new_state = apply_to_all_matrices(state, lambda W: truncate(W, k))
            model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
            model.load_state_dict(new_state)
            model.eval()
            tr = eval_acc(model, train_in, train_lab)
            te = eval_acc(model, test_in, test_lab)
            results_rank[name][k] = {'train_acc': tr, 'test_acc': te}
            print(f'  k={k:>4}: train_acc={tr:.4f}  test_acc={te:.4f}')

    # =============== Test 2: quantization =================
    print('\n=== TEST 2: Uniform quantization of weights ===')
    LEVELS = [2, 4, 8, 16, 32, 64, 128, 256]
    results_quant = {'M': {}, 'G': {}}
    for name in ['M', 'G']:
        state = t.load(HERE / 'checkpoints' / name / 'final.pt', map_location='cpu', weights_only=True)['model']
        print(f'\n{name}:')
        for L in LEVELS:
            new_state = apply_to_all_matrices(state, lambda W: quantize_uniform(W, L))
            model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
            model.load_state_dict(new_state)
            model.eval()
            tr = eval_acc(model, train_in, train_lab)
            te = eval_acc(model, test_in, test_lab)
            results_quant[name][L] = {'train_acc': tr, 'test_acc': te}
            print(f'  levels={L:>4}: train_acc={tr:.4f}  test_acc={te:.4f}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for name in ['M', 'G']:
        ks = list(results_rank[name].keys())
        tr_accs = [results_rank[name][k]['train_acc'] for k in ks]
        te_accs = [results_rank[name][k]['test_acc'] for k in ks]
        color = 'tab:red' if name == 'M' else 'tab:blue'
        ax.plot(ks, tr_accs, marker='o', label=f'{name} train', color=color)
        ax.plot(ks, te_accs, marker='s', label=f'{name} test', color=color, linestyle='--')
    ax.set_xlabel('rank k (all 3 matrices truncated)')
    ax.set_ylabel('accuracy')
    ax.set_title('Low-rank truncation: how compressible is M vs G?')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name in ['M', 'G']:
        ls = list(results_quant[name].keys())
        tr_accs = [results_quant[name][L]['train_acc'] for L in ls]
        te_accs = [results_quant[name][L]['test_acc'] for L in ls]
        color = 'tab:red' if name == 'M' else 'tab:blue'
        ax.plot(ls, tr_accs, marker='o', label=f'{name} train', color=color)
        ax.plot(ls, te_accs, marker='s', label=f'{name} test', color=color, linestyle='--')
    ax.set_xlabel('quantization levels (fewer = coarser)')
    ax.set_xscale('log', base=2)
    ax.set_ylabel('accuracy')
    ax.set_title('Quantization: how much precision does M actually need?')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Capacity test: how much of M's parameters are actually doing work?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_capacity.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
