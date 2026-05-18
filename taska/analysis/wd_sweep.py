"""WD strength sweep: how much WD is needed to escape M's saddle?

Quantitative claim: there exists a threshold WD below which the saddle is
NOT escaped within reasonable time, and above which it IS escaped. Map the
escape time vs WD strength.

WD ∈ {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}. Each starts from
M_50000. Continue for up to 30000 epochs. Measure time to grok.

Sharpest quantitative claim if there's a clean threshold:
"WD escapes the saddle iff WD > threshold. Below this, M stays memorizing."

Usage:
    python taska/analysis/wd_sweep.py
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
NUM_EPOCHS = 30000
LOG_EVERY = 500
WD_VALUES = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


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


def run_wd(wd, M_state, device, train_in, train_lab, test_in, test_lab):
    print(f'\n=== wd={wd} ===')
    model = make_model(M_state, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))

    history = {'epoch': [], 'test_acc': [], 'train_loss': []}

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
            if te > 0.99:
                print(f'  ep={ep+1}: GROKKED')
                break
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

    histories = {}
    for wd in WD_VALUES:
        histories[wd] = run_wd(wd, M_state, device, train_in, train_lab, test_in, test_lab)

    out_json = HERE / 'results' / 'wd_sweep.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({str(k): v for k, v in histories.items()}, f)
    print(f'\nhistories -> {out_json}')

    # Plot 1: trajectory per WD
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(WD_VALUES)))
    ax = axes[0]
    for color, wd in zip(cmap, WD_VALUES):
        h = histories[wd]
        ax.plot(h['epoch'], h['test_acc'], label=f'wd={wd}', color=color)
    ax.set_xlabel('rescue epoch')
    ax.set_ylabel('test accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Test acc trajectory by WD strength')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: escape time vs WD strength
    ax = axes[1]
    escape_times = []
    for wd in WD_VALUES:
        h = histories[wd]
        e = next((h['epoch'][i] for i, a in enumerate(h['test_acc']) if a >= 0.95), None)
        escape_times.append(e if e is not None else NUM_EPOCHS + 1)
    ax.plot(WD_VALUES, escape_times, marker='o', color='tab:red')
    ax.axhline(NUM_EPOCHS, color='gray', linestyle=':', label=f'never (>{NUM_EPOCHS})')
    ax.set_xlabel('weight decay')
    ax.set_ylabel('epochs to grok')
    ax.set_xscale('log')
    ax.set_title('Escape time vs WD strength')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("WD strength sweep: is there a threshold below which escape fails?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_wd_sweep.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')

    print('\n=== Outcomes ===')
    for wd, e in zip(WD_VALUES, escape_times):
        out_str = 'NEVER' if e > NUM_EPOCHS else f'epoch {e}'
        print(f'  wd={wd}: grok @ {out_str}')


if __name__ == '__main__':
    main()
