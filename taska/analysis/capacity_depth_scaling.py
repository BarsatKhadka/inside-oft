"""Capacity + depth scaling: is rank task-determined or capacity-determined?

Sharp claim to test: converged G rank is determined by the TASK, not by
architecture. Specifically, for (a+b) mod 113, G's rank converges to ~10-15
regardless of d_model or depth.

Test:
  d_model ∈ {64, 128, 256, 512}  ×  num_layers ∈ {1, 2}
  Train G (WD=1.0) for 50k epochs each, seed=0.
  Measure: converged effective rank of W_E, W_in, W_out (per layer).
  Measure: grok epoch.

If rank is invariant across capacity → "rank = task complexity" hypothesis.
If rank scales with capacity → benign overfitting regime, rank is whatever
the optimizer settles to.

Usage:
    python taska/analysis/capacity_depth_scaling.py
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
WD = 1.0
NUM_EPOCHS = 50000
LOG_EVERY = 500
D_MODELS = [64, 128, 256, 512]
NUM_LAYERS_LIST = [1, 2]


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


def train_one(d_model, num_layers, device):
    print(f'\n=== d_model={d_model}, num_layers={num_layers} ===')
    t.manual_seed(0)
    model = Transformer(p=P, d_model=d_model, num_heads=4,
                        n_ctx=3, num_layers=num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    grok_epoch = None
    history = {'epoch': [], 'test_acc': [], 'train_loss': []}

    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
            history['train_loss'].append(loss.item())
            if grok_epoch is None and te >= 0.95:
                grok_epoch = ep + 1
                print(f'  grok at {grok_epoch}')

    # Final ranks
    ranks = {}
    ranks['W_E'] = effective_rank(model.embed.W_E[:, :P])
    for i in range(num_layers):
        ranks[f'L{i}.W_in']  = effective_rank(model.blocks[i].mlp.W_in)
        ranks[f'L{i}.W_out'] = effective_rank(model.blocks[i].mlp.W_out)

    final = {'final_test_acc': history['test_acc'][-1], 'grok_epoch': grok_epoch, 'ranks': ranks}
    print(f'  result: test_acc={final["final_test_acc"]:.4f}, grok@{grok_epoch}, ranks={ranks}')
    return history, final


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    all_results = {}
    for nl in NUM_LAYERS_LIST:
        for dm in D_MODELS:
            history, final = train_one(dm, nl, device)
            all_results[f'L{nl}_d{dm}'] = {'history': history, 'final': final, 'd_model': dm, 'num_layers': nl}

    out_json = HERE / 'results' / 'capacity_depth_scaling.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(all_results, f)
    print(f'\nresults -> {out_json}')

    # Plot rank vs d_model, separately per depth and per matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, target in zip(axes, ['W_E', 'L0.W_out']):
        for nl in NUM_LAYERS_LIST:
            xs = []
            ys = []
            for dm in D_MODELS:
                key = f'L{nl}_d{dm}'
                if target in all_results[key]['final']['ranks']:
                    xs.append(dm)
                    ys.append(all_results[key]['final']['ranks'][target])
            if xs:
                ax.plot(xs, ys, marker='o', label=f'num_layers={nl}')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('d_model')
        ax.set_ylabel(f'effective rank of {target}')
        ax.set_title(f'{target}: rank vs capacity')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Does converged G rank depend on architecture, or only on task?')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_capacity_depth_scaling.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')

    print('\n=== Per-config ranks ===')
    for key, res in all_results.items():
        print(f'  {key}: grok={res["final"]["grok_epoch"]}, '
              f'final_acc={res["final"]["final_test_acc"]:.4f}, ranks={res["final"]["ranks"]}')


if __name__ == '__main__':
    main()
