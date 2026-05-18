"""Transformer architecture sweep on (a+b) mod 113.

For each (depth, width) combination, train both M (WD=0) and G (WD=1) and
measure: converged rank of each weight matrix, grok success, time to grok.

Provides the cross-architecture table for the paper.

Architectures:
  depth ∈ {1, 2, 4}
  width ∈ {64, 128, 256, 512}
  → 12 architectures × 2 regimes = 24 runs

Usage:
    python taska/analysis/arch_sweep_transformer.py
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
LR = 1e-3
NUM_EPOCHS = 30000
LOG_EVERY = 500
DEPTHS = [1, 2, 4]
WIDTHS = [64, 128, 256, 512]


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


def train_one(depth, width, wd, device):
    print(f'\n=== depth={depth}, width={width}, wd={wd} ===')
    t.manual_seed(0)
    model = Transformer(p=P, d_model=width, num_heads=4, n_ctx=3, num_layers=depth).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
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

    final_te = eval_acc(model, test_in, test_lab)
    final_tr = eval_acc(model, train_in, train_lab)
    ranks = {'W_E': effective_rank(model.embed.W_E[:, :P])}
    for i in range(depth):
        ranks[f'L{i}.W_in']  = effective_rank(model.blocks[i].mlp.W_in)
        ranks[f'L{i}.W_out'] = effective_rank(model.blocks[i].mlp.W_out)

    print(f'  result: train_acc={final_tr:.4f}, test_acc={final_te:.4f}, grok@{grok_epoch}')
    print(f'  ranks: {ranks}')
    return {'depth': depth, 'width': width, 'wd': wd,
            'train_acc': final_tr, 'test_acc': final_te,
            'grok_epoch': grok_epoch, 'ranks': ranks}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    results = {}
    for depth in DEPTHS:
        for width in WIDTHS:
            for wd in [0.0, 1.0]:
                key = f'd{depth}_w{width}_wd{wd}'
                results[key] = train_one(depth, width, wd, device)

    out_json = HERE / 'results' / 'arch_sweep_transformer.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Print table
    print('\n=== Architecture × WD table ===')
    print(f'{"depth":>5}  {"width":>5}  {"WD":>4}  {"test_acc":>8}  {"grok":>6}  {"W_out_rank":>11}')
    for depth in DEPTHS:
        for width in WIDTHS:
            for wd in [0.0, 1.0]:
                r = results[f'd{depth}_w{width}_wd{wd}']
                wout_key = f'L{depth-1}.W_out'   # last layer's W_out
                wout = r['ranks'].get(wout_key, 0)
                print(f'{depth:>5}  {width:>5}  {wd:>4}  {r["test_acc"]:>8.4f}  {str(r["grok_epoch"]):>6}  {wout:>11.2f}')

    # Plot: M vs G rank across architectures
    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = []
    M_ranks = []
    G_ranks = []
    for depth in DEPTHS:
        for width in WIDTHS:
            x_labels.append(f'd{depth}_w{width}')
            wout_key = f'L{depth-1}.W_out'
            M_ranks.append(results[f'd{depth}_w{width}_wd0.0']['ranks'].get(wout_key, 0))
            G_ranks.append(results[f'd{depth}_w{width}_wd1.0']['ranks'].get(wout_key, 0))
    x = np.arange(len(x_labels))
    ax.bar(x - 0.2, M_ranks, 0.4, color='tab:red', label='M (WD=0)')
    ax.bar(x + 0.2, G_ranks, 0.4, color='tab:blue', label='G (WD=1)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('W_out (last layer) effective rank')
    ax.set_title('M vs G effective rank across transformer architectures on (a+b) mod 113')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_arch_sweep_transformer.png'
    fig.savefig(out, dpi=130)
    print(f'\nplot -> {out}')


if __name__ == '__main__':
    main()
