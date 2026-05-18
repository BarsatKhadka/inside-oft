"""Rank trajectory for Track B: does G's effective rank compress over training while M's stays flat?

Same idea as taska/analysis/rank_trajectory.py but on ResNet-18 fc layer
(and selected conv layers) using the saved CIFAR checkpoints.

Usage:
    python trackb/rank_trajectory_trackb.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def make_resnet18():
    m = models.resnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def effective_rank(W):
    sigma = torch.linalg.svdvals(W)
    p = sigma ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def flatten_conv(W):
    return W.reshape(W.shape[0], -1)


def collect_epochs(d):
    return sorted(int(p.stem.split('_')[1]) for p in d.glob('epoch_*.pt'))


def main():
    layers_of_interest = ['conv1', 'layer1.0.conv1', 'layer4.1.conv2', 'fc']

    results = {}
    for name in ['M', 'G']:
        d = HERE / 'checkpoints' / name
        epochs = collect_epochs(d)
        if not epochs:
            print(f'No checkpoints found for {name}')
            continue
        # Also include final
        results[name] = {l: {'epoch': [], 'eff_rank': []} for l in layers_of_interest}
        print(f'\n=== {name}: {len(epochs)} checkpoints (epochs {epochs[0]} to {epochs[-1]}) ===')
        for ep in epochs:
            state = torch.load(d / f'epoch_{ep}.pt', map_location='cpu', weights_only=True)['model']
            for l in layers_of_interest:
                key = l + '.weight'
                if key not in state:
                    continue
                W = state[key]
                if W.ndim == 4:
                    W = flatten_conv(W)
                er = effective_rank(W)
                results[name][l]['epoch'].append(ep)
                results[name][l]['eff_rank'].append(er)
            if ep % 50 == 0 or ep == epochs[-1]:
                last_r = {l: f'{results[name][l]["eff_rank"][-1]:.1f}' for l in layers_of_interest
                          if results[name][l]['eff_rank']}
                print(f'  epoch {ep:4d}: {last_r}')

    # Plot
    n_layers = len(layers_of_interest)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    for ax, l in zip(axes, layers_of_interest):
        for name, color in [('M', 'tab:red'), ('G', 'tab:blue')]:
            if name in results and results[name][l]['eff_rank']:
                ax.plot(results[name][l]['epoch'], results[name][l]['eff_rank'],
                        marker='o', markersize=3, label=name, color=color)
        ax.set_xlabel('epoch')
        ax.set_ylabel('effective rank')
        ax.set_title(l)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle('Rank trajectory for Track B: G compresses, M stays?')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_rank_trajectory_trackb.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
