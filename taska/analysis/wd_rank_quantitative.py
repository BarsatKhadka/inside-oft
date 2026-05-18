"""Quantitative WD-Rank-Escape relationship with error bars.

Sharp claim to test: rank(WD) follows a specific functional form, and escape
time is a function of (initial_rank, target_rank, WD strength).

Experiment:
  For each WD ∈ logspace(-3, 1, 11) × each seed ∈ {0, 1, 2}:
    Rescue M_seed from epoch 50,000 with this WD for 30k epochs.
    Record: rank trajectory of W_out and W_in every 500 epochs.
    Record: test accuracy trajectory.
    Record: final rank, escape time (epoch where test_acc first >= 0.95).

  Then:
    Plot final_rank vs WD (with seed std bars).
    Fit log-linear or power law. Report R² of fit.
    Plot escape_time vs WD.
    Identify threshold WD below which escape never happens.
    Identify scaling regime above threshold.

This produces 33 separate training runs (11 WD × 3 seeds). About 6-10 hours
on L40S given each is 30k epochs.

Usage:
    python taska/analysis/wd_rank_quantitative.py
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
LR = 1e-3
NUM_EPOCHS = 30000
LOG_EVERY = 500
WD_VALUES = list(np.logspace(-3, 1, 11))   # 0.001 to 10.0, log-spaced
SEEDS = [0, 1, 2]


def load_state(p):
    return t.load(p, map_location='cpu', weights_only=True)['model']


def make_model(state, device):
    m = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    m.load_state_dict(state)
    m.to(device)
    return m


def effective_rank(W):
    s = t.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def full_loss(model, inputs, labels):
    return cross_entropy_hp(model(inputs)[:, -1, :], labels)


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def run_one(wd, seed, device):
    print(f'\n--- wd={wd:.4f}, seed={seed} ---')
    M_state = load_state(HERE / 'checkpoints' / ('M' if seed == 0 else f'M_seed{seed}') / 'final.pt')
    model = make_model(M_state, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=seed)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    history = {
        'epoch': [], 'test_acc': [], 'train_loss': [],
        'rank_W_out': [], 'rank_W_in': [], 'rank_W_E': [],
    }
    escape_epoch = None

    for ep in range(NUM_EPOCHS):
        train_loss = full_loss(model, train_in, train_lab)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
            history['train_loss'].append(train_loss.item())
            history['rank_W_out'].append(effective_rank(model.blocks[0].mlp.W_out))
            history['rank_W_in'].append(effective_rank(model.blocks[0].mlp.W_in))
            history['rank_W_E'].append(effective_rank(model.embed.W_E[:, :P]))
            if escape_epoch is None and te >= 0.95:
                escape_epoch = ep + 1
                print(f'  GROK at epoch {escape_epoch}')

    final = {
        'final_test_acc': history['test_acc'][-1],
        'escape_epoch': escape_epoch,
        'final_rank_W_out': history['rank_W_out'][-1],
        'final_rank_W_in': history['rank_W_in'][-1],
        'final_rank_W_E': history['rank_W_E'][-1],
    }
    print(f'  result: test_acc={final["final_test_acc"]:.4f}, escape={escape_epoch}, '
          f'W_out_rank={final["final_rank_W_out"]:.2f}')
    return history, final


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print(f'WD values: {WD_VALUES}')
    print(f'seeds: {SEEDS}')
    print(f'Total runs: {len(WD_VALUES) * len(SEEDS)}')

    all_results = {}
    for wd in WD_VALUES:
        all_results[wd] = {}
        for seed in SEEDS:
            history, final = run_one(wd, seed, device)
            all_results[wd][seed] = {'history': history, 'final': final}

    out_json = HERE / 'results' / 'wd_rank_quantitative.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({str(k): {str(s): v for s, v in d.items()} for k, d in all_results.items()}, f)

    # Aggregate: per-WD, mean and std of (final_rank, escape_epoch)
    wd_arr = np.array(WD_VALUES)
    final_ranks_W_out = []
    final_ranks_std = []
    escape_epochs = []
    escape_epochs_std = []
    for wd in WD_VALUES:
        ranks = [all_results[wd][s]['final']['final_rank_W_out'] for s in SEEDS]
        final_ranks_W_out.append(np.mean(ranks))
        final_ranks_std.append(np.std(ranks))
        escapes = [all_results[wd][s]['final']['escape_epoch'] for s in SEEDS]
        escapes = [e if e is not None else NUM_EPOCHS + 1 for e in escapes]
        escape_epochs.append(np.mean(escapes))
        escape_epochs_std.append(np.std(escapes))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.errorbar(wd_arr, final_ranks_W_out, yerr=final_ranks_std, marker='o', capsize=4)
    ax.set_xscale('log')
    ax.set_xlabel('weight decay strength')
    ax.set_ylabel('final W_out effective rank')
    ax.set_title('Quantitative WD → Rank relationship (mean ± std over 3 seeds)')
    ax.grid(True, alpha=0.3)
    # Try log-linear fit
    try:
        log_wd = np.log(wd_arr)
        coef = np.polyfit(log_wd, final_ranks_W_out, 1)
        pred = np.polyval(coef, log_wd)
        r2 = 1 - np.var(np.array(final_ranks_W_out) - pred) / np.var(final_ranks_W_out)
        ax.plot(wd_arr, pred, '--', alpha=0.5,
                label=f'log-linear: rank = {coef[0]:.2f}*log(WD) + {coef[1]:.2f}, R²={r2:.3f}')
        ax.legend()
    except Exception as e:
        print(f'fit failed: {e}')

    ax = axes[1]
    ax.errorbar(wd_arr, escape_epochs, yerr=escape_epochs_std, marker='s', color='tab:red', capsize=4)
    ax.set_xscale('log')
    ax.set_xlabel('weight decay strength')
    ax.set_ylabel('epoch at which test_acc >= 0.95')
    ax.axhline(NUM_EPOCHS, linestyle=':', color='gray', label=f'never escapes (>{NUM_EPOCHS})')
    ax.set_title('Quantitative WD → Escape time (mean ± std over 3 seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Quantitative WD-Rank-Escape relationship')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_wd_rank_quantitative.png'
    fig.savefig(out, dpi=130)
    print(f'\nplot -> {out}')

    print('\n=== Summary ===')
    for i, wd in enumerate(WD_VALUES):
        print(f'  wd={wd:.4f}: final_rank={final_ranks_W_out[i]:.2f}±{final_ranks_std[i]:.2f}, '
              f'escape_epoch={escape_epochs[i]:.0f}±{escape_epochs_std[i]:.0f}')


if __name__ == '__main__':
    main()
