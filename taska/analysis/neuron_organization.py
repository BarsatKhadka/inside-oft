"""C. Neuron organization: what do M's 512 MLP neurons individually compute?

For each neuron i in M's MLP (post-ReLU, 512 total):
  - Compute its activation on all 12769 (a, b) pairs at position 2
  - Find top-K pairs it fires most strongly on
  - Check: do those pairs share structure (same a? same b? same sum?)
  - Measure: how "selective" is each neuron (firing on few pairs vs many)?

Then:
  - Cluster neurons by their activation patterns (k-means on activations).
  - Compare to G's neurons -- G's are well-understood (Fourier components).
  - Are M's neurons MORE selective than G's? Less?

If M's neurons cluster into "input detectors" (fire for specific a, b
patterns) and "output writers" (correlate with specific output values),
that's evidence of an emergent key-value memory architecture.

Usage:
    python taska/analysis/neuron_organization.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.cluster import KMeans

from model import Transformer

P = 113


def load_model(ckpt):
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt, map_location='cpu', weights_only=True)['model']
    model.load_state_dict(state)
    model.eval()
    return model


@t.no_grad()
def all_mlp_acts(model):
    """For all 12769 pairs, get post-ReLU MLP activations at position 2.
    Returns: (12769, 512) numpy array."""
    pairs = [(a, b, P) for a in range(P) for b in range(P)]
    inputs = t.tensor(pairs, dtype=t.long)
    x = model.embed(inputs)
    x = model.pos_embed(x)
    block = model.blocks[0]
    x = x + block.attn(x)
    h_pre = t.einsum('md,bpd->bpm', block.mlp.W_in, x) + block.mlp.b_in
    h_post = t.nn.functional.relu(h_pre)
    return h_post[:, -1, :].numpy()


def selectivity(activations):
    """For each neuron, return Gini-like concentration: fraction of total
    activation captured by the top 1% of inputs."""
    n_inputs = activations.shape[0]
    top_k = max(1, n_inputs // 100)
    sels = []
    for neuron in range(activations.shape[1]):
        a = np.abs(activations[:, neuron])
        if a.sum() < 1e-9:
            sels.append(0)
            continue
        top = np.sort(a)[-top_k:].sum()
        sels.append(top / a.sum())
    return np.array(sels)


def summarize_top_pairs(activations, neuron_idx, top_k=10):
    """For one neuron, return the top-K activating (a, b) pairs."""
    flat = activations[:, neuron_idx]
    top = np.argsort(-flat)[:top_k]
    return [(idx // P, idx % P, flat[idx]) for idx in top]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    summary = {}
    for col, (name, ckpt) in enumerate([
        ('M', HERE / 'checkpoints' / 'M' / 'final.pt'),
        ('G', HERE / 'checkpoints' / 'G' / 'final.pt'),
    ]):
        model = load_model(ckpt)
        acts = all_mlp_acts(model)   # (12769, 512)
        print(f'\n=== {name} ===')
        print(f'MLP hidden activations shape: {acts.shape}')

        # Selectivity per neuron
        sels = selectivity(acts)
        print(f'Per-neuron selectivity (top 1% / total): mean={sels.mean():.3f}  '
              f'median={np.median(sels):.3f}  min={sels.min():.3f}  max={sels.max():.3f}')

        # Mean activation per neuron
        mean_acts = np.abs(acts).mean(axis=0)
        print(f'Mean |activation|: mean={mean_acts.mean():.3f}  '
              f'min={mean_acts.min():.3f}  max={mean_acts.max():.3f}')
        print(f'Dead neurons (mean act < 1e-4): {(mean_acts < 1e-4).sum()} / 512')

        # For most-selective neuron, show top-10 pairs
        most_selective = np.argmax(sels)
        top_pairs = summarize_top_pairs(acts, most_selective)
        print(f'\nMost selective neuron: #{most_selective} (selectivity={sels[most_selective]:.3f})')
        print(f'Top 10 activating pairs (a, b, activation):')
        for a, b, act in top_pairs[:10]:
            print(f'  ({a:3d}, {b:3d}) sum={(a+b)%P:3d}  act={act:.3f}')

        # Plot: selectivity histogram
        ax = axes[0, col]
        ax.hist(sels, bins=40, color='tab:red' if name == 'M' else 'tab:blue')
        ax.set_xlabel('selectivity (top 1% fraction of total activation)')
        ax.set_ylabel('neuron count')
        ax.set_title(f'{name}: per-neuron selectivity')
        ax.axvline(0.01, color='gray', linestyle=':', label='uniform-firing baseline (0.01)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot: activation heatmap for top-100 most-selective neurons sorted, all 113x113 pairs
        ax = axes[1, col]
        top_neurons = np.argsort(-sels)[:100]
        h = acts[:, top_neurons].T   # (100 neurons, 12769 inputs)
        # Normalize each row for visualization
        h_norm = h / (np.abs(h).max(axis=1, keepdims=True) + 1e-9)
        ax.imshow(h_norm, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('input pair index (a * 113 + b)')
        ax.set_ylabel('neuron rank (by selectivity)')
        ax.set_title(f'{name}: top 100 most-selective neurons')

        summary[name] = {'selectivity_mean': sels.mean(), 'selectivity_median': np.median(sels),
                         'dead_neurons': int((mean_acts < 1e-4).sum())}

    fig.suptitle('Neuron organization in M vs G')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_neuron_organization.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')

    print('\nSummary:')
    for n, s in summary.items():
        print(f'  {n}: mean selectivity = {s["selectivity_mean"]:.3f}, dead neurons = {s["dead_neurons"]}/512')
    print('If M shows much higher selectivity than G, it suggests M\'s neurons specialize on specific input patterns.')


if __name__ == '__main__':
    main()
