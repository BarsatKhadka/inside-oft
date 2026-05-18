"""Per-layer rank constraint: which layer is the bottleneck for memorization?

Sharp claim: if there's a SPECIFIC layer whose rank is bottlenecking the
memorization circuit, then constraining only THAT layer to low rank should
escape the saddle, while constraining a different layer at the same rank
should not.

For each layer L ∈ {W_E, W_in, W_out} and each rank k ∈ {10, 20, 50}:
  Start from M_50000. Continue training with WD=0.
  Each step: standard gradient step, then project ONLY layer L to rank k.
  Other layers are unconstrained.
  Measure: does it escape the saddle?

If only constraining one specific layer (say W_out) at rank 10 escapes,
and constraining others at the same rank doesn't, we've identified the
"memorization bottleneck layer."

If all layers work equally well, the memorization is distributed and rank
must be constrained globally.

Usage:
    python taska/analysis/per_layer_rank.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import json
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.optim as optim

from data import gen_train_test, to_tensors
from model import Transformer

P = 113
D_MODEL = 128
SEED = 0
LR = 1e-3
NUM_EPOCHS = 20000
LOG_EVERY = 500
LAYER_NAMES = ['W_E', 'W_in', 'W_out']
K_VALUES = [10, 20, 50]


def load_state(p):
    return t.load(p, map_location='cpu', weights_only=True)['model']


def make_model(state, device):
    m = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    m.load_state_dict(state)
    m.to(device)
    return m


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def full_loss(model, inputs, labels):
    return cross_entropy_hp(model(inputs)[:, -1, :], labels)


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def project_to_rank(W, k):
    U, S, Vt = t.linalg.svd(W, full_matrices=False)
    k_use = min(k, S.shape[0])
    return U[:, :k_use] @ t.diag(S[:k_use]) @ Vt[:k_use, :]


def run_per_layer(layer_name, k, M_state, device, train_in, train_lab, test_in, test_lab):
    print(f'\n=== layer={layer_name}, k={k} ===')
    model = make_model(M_state, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))

    history = {'epoch': [], 'test_acc': []}

    for ep in range(NUM_EPOCHS):
        loss = full_loss(model, train_in, train_lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Project ONLY the specified layer to rank k
        with t.no_grad():
            if layer_name == 'W_E':
                model.embed.W_E[:, :P] = project_to_rank(model.embed.W_E[:, :P], k)
            elif layer_name == 'W_in':
                model.blocks[0].mlp.W_in.copy_(project_to_rank(model.blocks[0].mlp.W_in, k))
            elif layer_name == 'W_out':
                model.blocks[0].mlp.W_out.copy_(project_to_rank(model.blocks[0].mlp.W_out, k))

        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
        if (ep + 1) % 4000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}')

    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    M_state = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    results = {}
    for layer in LAYER_NAMES:
        results[layer] = {}
        for k in K_VALUES:
            results[layer][k] = run_per_layer(layer, k, M_state, device,
                                              train_in, train_lab, test_in, test_lab)

    out_json = HERE / 'results' / 'per_layer_rank.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({l: {str(k): v for k, v in d.items()} for l, d in results.items()}, f)

    fig, axes = plt.subplots(1, len(LAYER_NAMES), figsize=(5 * len(LAYER_NAMES), 4), sharey=True)
    for ax, layer in zip(axes, LAYER_NAMES):
        for k in K_VALUES:
            h = results[layer][k]
            ax.plot(h['epoch'], h['test_acc'], marker='.', label=f'k={k}')
        ax.set_xlabel('rescue epoch')
        ax.set_ylabel('test accuracy' if ax is axes[0] else '')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'Constrain only {layer}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-layer rank constraint: which layer is the memorization bottleneck?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_per_layer_rank.png'
    fig.savefig(out, dpi=130)
    print(f'\nplot -> {out}')

    print('\n=== Outcomes ===')
    for layer in LAYER_NAMES:
        for k in K_VALUES:
            h = results[layer][k]
            grok = next((h['epoch'][i] for i, a in enumerate(h['test_acc']) if a >= 0.95), None)
            print(f'  {layer:>6} k={k:>3}: final={h["test_acc"][-1]:.4f}, grok @ {grok}')


if __name__ == '__main__':
    main()
