"""KEY MECHANISM TEST: track effective rank during WD-rescue vs SAM-rescue vs noise-rescue.

If WD escapes the saddle BY compressing rank, then rank should drop during
WD-rescue but stay flat for the failed rescues (SAM, noise). This is the
direct mechanistic test of "WD escapes via low-rank bias."

For each escape mechanism, load M_50000 and continue training. Every 500
epochs, record effective rank of W_E, W_in, W_out, AND current test accuracy.

Expected:
  - WD rescue: rank drops, test acc rises (causal: rank compression → grok)
  - SAM rescue: rank stays high, test acc stays low (or partial rise)
  - Noise rescue: rank stays high, test acc stays low

If observed, this LOCKS the mechanism: "WD's privilege is its rank-compression bias."

Usage:
    python taska/analysis/rank_during_rescue.py
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


def record_ranks(model):
    return {
        'W_E':   effective_rank(model.embed.W_E[:, :P]),
        'W_in':  effective_rank(model.blocks[0].mlp.W_in),
        'W_out': effective_rank(model.blocks[0].mlp.W_out),
    }


def run_rescue(name, mechanism_fn, M_state, device, train_in, train_lab, test_in, test_lab,
               extra_in=None, extra_lab=None):
    print(f'\n=== {name} ===')
    model = make_model(M_state, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))

    history = {'epoch': [], 'test_acc': [], 'train_loss': [], 'ranks': []}

    init_ranks = record_ranks(model)
    init_acc   = eval_acc(model, test_in, test_lab)
    history['epoch'].append(0)
    history['test_acc'].append(init_acc)
    history['ranks'].append(init_ranks)
    history['train_loss'].append(full_loss(model, train_in, train_lab).item())
    print(f'  ep=0: test_acc={init_acc:.4f}, W_E_rank={init_ranks["W_E"]:.2f}, '
          f'W_in_rank={init_ranks["W_in"]:.2f}, W_out_rank={init_ranks["W_out"]:.2f}')

    for ep in range(NUM_EPOCHS):
        train_loss = full_loss(model, train_in, train_lab)
        if extra_in is not None:
            train_loss = train_loss + full_loss(model, extra_in, extra_lab)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if mechanism_fn is not None:
            mechanism_fn(model, optimizer, ep)

        if (ep + 1) % LOG_EVERY == 0:
            r = record_ranks(model)
            te = eval_acc(model, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
            history['train_loss'].append(train_loss.item())
            history['ranks'].append(r)
        if (ep + 1) % 4000 == 0:
            print(f'  ep={ep+1:5d}: test_acc={te:.4f}, W_out_rank={r["W_out"]:.2f}')

    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    M_state = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    histories = {}

    # Control: nothing
    histories['nothing'] = run_rescue(
        'control_nothing', None,
        M_state, device, train_in, train_lab, test_in, test_lab,
    )

    # WD=1.0 (known to escape)
    def wd_mech(model, optimizer, step):
        with t.no_grad():
            for p in model.parameters():
                p.mul_(1 - LR * 1.0)
    histories['wd_1.0'] = run_rescue(
        'wd_1.0', wd_mech,
        M_state, device, train_in, train_lab, test_in, test_lab,
    )

    # SAM (failed to escape but had partial movement)
    SAM_RHO = 0.05
    def sam_mech(model, optimizer, step):
        with t.no_grad():
            grads = [p.grad.detach().clone() if p.grad is not None else t.zeros_like(p) for p in model.parameters()]
            grad_norm = t.sqrt(sum((g ** 2).sum() for g in grads)) + 1e-12
            scale = SAM_RHO / grad_norm
            saved = [p.detach().clone() for p in model.parameters()]
            for p, g in zip(model.parameters(), grads):
                p.add_(g * scale)
        loss2 = full_loss(model, train_in, train_lab)
        optimizer.zero_grad()
        loss2.backward()
        with t.no_grad():
            for p, s in zip(model.parameters(), saved):
                p.copy_(s)
        optimizer.step()
    histories['sam_0.05'] = run_rescue(
        'sam_0.05', sam_mech,
        M_state, device, train_in, train_lab, test_in, test_lab,
    )

    # Noise (failed to escape)
    NOISE_STD = 0.001
    def noise_mech(model, optimizer, step):
        with t.no_grad():
            for p in model.parameters():
                p.add_(t.randn_like(p) * NOISE_STD)
    histories['noise_0.001'] = run_rescue(
        'noise_0.001', noise_mech,
        M_state, device, train_in, train_lab, test_in, test_lab,
    )

    out_json = HERE / 'results' / 'rank_during_rescue.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(histories, f)
    print(f'\nhistories -> {out_json}')

    # Plot: rank trajectory and test_acc trajectory for each mechanism
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {'nothing': 'gray', 'wd_1.0': 'tab:red', 'sam_0.05': 'tab:orange', 'noise_0.001': 'tab:green'}

    # Top-left: W_out rank over time
    ax = axes[0, 0]
    for name, h in histories.items():
        ranks = [r['W_out'] for r in h['ranks']]
        ax.plot(h['epoch'], ranks, marker='.', label=name, color=colors[name])
    ax.set_xlabel('rescue epoch')
    ax.set_ylabel('W_out effective rank')
    ax.set_title('W_out rank during rescue: does WD specifically compress?')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Top-right: W_in rank
    ax = axes[0, 1]
    for name, h in histories.items():
        ranks = [r['W_in'] for r in h['ranks']]
        ax.plot(h['epoch'], ranks, marker='.', label=name, color=colors[name])
    ax.set_xlabel('rescue epoch')
    ax.set_ylabel('W_in effective rank')
    ax.set_title('W_in rank during rescue')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Bottom-left: W_E rank
    ax = axes[1, 0]
    for name, h in histories.items():
        ranks = [r['W_E'] for r in h['ranks']]
        ax.plot(h['epoch'], ranks, marker='.', label=name, color=colors[name])
    ax.set_xlabel('rescue epoch')
    ax.set_ylabel('W_E effective rank')
    ax.set_title('W_E rank during rescue')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Bottom-right: test acc
    ax = axes[1, 1]
    for name, h in histories.items():
        ax.plot(h['epoch'], h['test_acc'], marker='.', label=name, color=colors[name])
    ax.set_xlabel('rescue epoch')
    ax.set_ylabel('test accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Test acc during rescue (for reference)')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Rank trajectory during rescue: does WD escape BY compressing rank?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_rank_during_rescue.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')


if __name__ == '__main__':
    main()
