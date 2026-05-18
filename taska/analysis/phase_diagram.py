"""Phase diagram in (train_fraction, WD) space.

The classic Liu et al. 2022 phase diagram has 4 regions for grokking:
comprehension, grokking, memorization, confusion. We add structural
characterization: for each cell, record final test_acc AND converged rank.

Grid: WD ∈ {0, 0.01, 0.1, 1.0} × frac_train ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 0.9}
= 24 runs × 20k epochs each. Single seed for speed.

For each cell record:
  - Final test accuracy
  - Final W_out effective rank
  - Whether it grokked (and when)

Output: 2D heatmap of (test_acc, rank) over the grid.

This locates where:
  - Catastrophic memorization (low frac_train, low WD)
  - Grokking (moderate frac, moderate WD)
  - Benign (high frac, low WD or high WD)
  - Confusion (low frac, high WD)

Provides a SHARP, quantitative phase boundary for the paper's
"regime-invariant signature" claim.

Usage:
    python taska/analysis/phase_diagram.py
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
WD_VALUES = [0.0, 0.01, 0.1, 1.0]
FRAC_TRAIN_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def effective_rank(W):
    s = t.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def run_cell(wd, frac, device):
    print(f'\n--- wd={wd}, frac_train={frac} ---')
    t.manual_seed(0)
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=frac, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    grok_epoch = None
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            if grok_epoch is None and te >= 0.95:
                grok_epoch = ep + 1
        if (ep + 1) % 5000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}')

    final_test = eval_acc(model, test_in, test_lab)
    final_train = eval_acc(model, train_in, train_lab)
    final_rank = effective_rank(model.blocks[0].mlp.W_out)
    print(f'  result: test_acc={final_test:.4f}, train_acc={final_train:.4f}, '
          f'W_out_rank={final_rank:.2f}, grok@{grok_epoch}')
    return {'test_acc': final_test, 'train_acc': final_train, 'rank_W_out': final_rank,
            'grok_epoch': grok_epoch}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    grid = {}
    for wd in WD_VALUES:
        for frac in FRAC_TRAIN_VALUES:
            grid[(wd, frac)] = run_cell(wd, frac, device)

    out_json = HERE / 'results' / 'phase_diagram.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({f'{k[0]}_{k[1]}': v for k, v in grid.items()}, f, indent=2)

    # 2D heatmaps
    rows = len(WD_VALUES)
    cols = len(FRAC_TRAIN_VALUES)
    test_acc_grid = np.zeros((rows, cols))
    rank_grid = np.zeros((rows, cols))
    for i, wd in enumerate(WD_VALUES):
        for j, frac in enumerate(FRAC_TRAIN_VALUES):
            test_acc_grid[i, j] = grid[(wd, frac)]['test_acc']
            rank_grid[i, j] = grid[(wd, frac)]['rank_W_out']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(test_acc_grid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(range(cols)); axes[0].set_xticklabels(FRAC_TRAIN_VALUES)
    axes[0].set_yticks(range(rows)); axes[0].set_yticklabels(WD_VALUES)
    axes[0].set_xlabel('train fraction')
    axes[0].set_ylabel('weight decay')
    axes[0].set_title('Final test accuracy (phase diagram)')
    plt.colorbar(im, ax=axes[0])
    for i in range(rows):
        for j in range(cols):
            axes[0].text(j, i, f'{test_acc_grid[i, j]:.2f}', ha='center', va='center', fontsize=8)

    im = axes[1].imshow(rank_grid, cmap='viridis', aspect='auto')
    axes[1].set_xticks(range(cols)); axes[1].set_xticklabels(FRAC_TRAIN_VALUES)
    axes[1].set_yticks(range(rows)); axes[1].set_yticklabels(WD_VALUES)
    axes[1].set_xlabel('train fraction')
    axes[1].set_ylabel('weight decay')
    axes[1].set_title('Final W_out effective rank')
    plt.colorbar(im, ax=axes[1])
    for i in range(rows):
        for j in range(cols):
            axes[1].text(j, i, f'{rank_grid[i, j]:.0f}', ha='center', va='center', fontsize=8, color='white')

    fig.suptitle('Phase diagram: test accuracy and final rank across (WD, frac_train)')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_phase_diagram.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')


if __name__ == '__main__':
    main()
