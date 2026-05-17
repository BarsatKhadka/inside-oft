"""Trajectory rescue: can M be saved mid-training by adding weight decay?

For each starting epoch t in a sweep:
  - Load M_t (M's checkpoint at epoch t, trained with weight_decay=0)
  - Continue training with weight_decay=1.0 (G's regime) for N additional epochs
  - Record train_acc and test_acc trajectories during rescue
  - Did it grok?

Predicts:
  - Small t (epoch 0): same as training G from scratch. Will grok.
  - Medium t (1000-5000): M is still in shared basin with G (per trajectory_basins.py).
    Rescue should be straightforward.
  - Large t (11000+): M has migrated to its own region. Rescue may fail.
  - Huge t (50000): the saddle's gradient norm is ~17 in test direction. Weight
    decay adds another force. Either WD wins (grokks) or the basin is too deep.

The result tells us at what training epoch memorization becomes IRREVERSIBLE
in the optimization-trajectory sense. This is a temporal characterization of
overfitting we haven't established yet.

Usage:
    python taska/analysis/trajectory_rescue.py
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
WEIGHT_DECAY = 1.0
RESCUE_EPOCHS = 20000   # number of additional epochs to train with WD on
LOG_EVERY = 200
START_EPOCHS = [0, 1000, 5000, 11000, 20000, 50000]


def load_state(p):
    return t.load(p, map_location='cpu', weights_only=True)['model']


def make_model(state, device):
    m = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    m.load_state_dict(state)
    m.to(device)
    return m


def cross_entropy_high_precision(logits, labels):
    logprobs = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -logprobs[t.arange(labels.shape[0]), labels].mean()


def full_loss(model, inputs, labels):
    logits = model(inputs)[:, -1, :]
    return cross_entropy_high_precision(logits, labels)


@t.no_grad()
def eval_acc(model, inputs, labels):
    logits = model(inputs)[:, -1, :]
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def m_ckpt_path(epoch):
    if epoch == 0:
        return HERE / 'checkpoints' / 'M' / 'init.pt'
    if epoch == 50000:
        return HERE / 'checkpoints' / 'M' / 'final.pt'
    return HERE / 'checkpoints' / 'M' / f'epoch_{epoch}.pt'


def rescue_run(start_epoch, device):
    """Load M at start_epoch, continue with weight_decay=1.0 for RESCUE_EPOCHS.

    Returns: dict with epoch list and train_acc, test_acc, train_loss, test_loss arrays.
    """
    print(f'\n=== Rescuing M at epoch {start_epoch} ===')
    s = load_state(m_ckpt_path(start_epoch))
    model = make_model(s, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    history = {'rel_epoch': [], 'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

    # log starting state
    tr_acc = eval_acc(model, train_in, train_lab)
    te_acc = eval_acc(model, test_in,  test_lab)
    tr_loss = full_loss(model, train_in, train_lab).item()
    te_loss = full_loss(model, test_in,  test_lab).item()
    history['rel_epoch'].append(0)
    history['train_acc'].append(tr_acc)
    history['test_acc'].append(te_acc)
    history['train_loss'].append(tr_loss)
    history['test_loss'].append(te_loss)
    print(f'  start: train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  train_loss={tr_loss:.4e}  test_loss={te_loss:.4e}')

    for ep in range(RESCUE_EPOCHS):
        train_loss = full_loss(model, train_in, train_lab)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (ep + 1) % LOG_EVERY == 0 or ep == RESCUE_EPOCHS - 1:
            tr_acc = eval_acc(model, train_in, train_lab)
            te_acc = eval_acc(model, test_in,  test_lab)
            te_loss = full_loss(model, test_in, test_lab).item()
            history['rel_epoch'].append(ep + 1)
            history['train_acc'].append(tr_acc)
            history['test_acc'].append(te_acc)
            history['train_loss'].append(train_loss.item())
            history['test_loss'].append(te_loss)

        if (ep + 1) % 2000 == 0:
            print(f'  rel_ep={ep + 1:5d}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}')

    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print(f'rescue config: lr={LR}, wd={WEIGHT_DECAY}, rescue_epochs={RESCUE_EPOCHS}')

    histories = {}
    for t_start in START_EPOCHS:
        h = rescue_run(t_start, device)
        histories[t_start] = h
        # peek at outcome
        final_test_acc = h['test_acc'][-1]
        print(f'  -> final test_acc after {RESCUE_EPOCHS} rescue epochs: {final_test_acc:.4f}')
        if final_test_acc > 0.95:
            print(f'  -> RESCUED. M at epoch {t_start} can be saved by adding WD.')
        elif final_test_acc > 0.3:
            print(f'  -> PARTIAL recovery.')
        else:
            print(f'  -> NOT rescued. Stuck at memorization.')

    # Save to JSON
    out_json = HERE / 'results' / 'trajectory_rescue.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({str(k): v for k, v in histories.items()}, f)
    print(f'\nhistory -> {out_json}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(START_EPOCHS)))

    for color, t_start in zip(cmap, START_EPOCHS):
        h = histories[t_start]
        axes[0].plot(h['rel_epoch'], h['test_acc'], color=color, label=f'start@{t_start}', marker='o', markersize=2)
        axes[1].plot(h['rel_epoch'], h['train_loss'], color=color, label=f'start@{t_start}')

    axes[0].set_xlabel('rescue epoch (after WD turned on)')
    axes[0].set_ylabel('test accuracy')
    axes[0].set_title('Trajectory rescue: test_acc after turning on weight decay at epoch t')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(title='starting M epoch')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('rescue epoch')
    axes[1].set_ylabel('train loss (log scale)')
    axes[1].set_yscale('log')
    axes[1].set_title('Train loss during rescue')
    axes[1].legend(title='starting M epoch')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Can M be rescued mid-training? Adding weight decay at epoch t and continuing.")
    fig.tight_layout()
    out_png = HERE / 'results' / 'fig_trajectory_rescue.png'
    fig.savefig(out_png, dpi=130)
    print(f'plot -> {out_png}')

    # Summary table
    print('\n=== Rescue outcomes ===')
    print(f'{"start_epoch":>12}  {"final_test_acc":>15}  {"outcome":>12}')
    for t_start in START_EPOCHS:
        f_te = histories[t_start]['test_acc'][-1]
        if f_te > 0.95:
            out = 'RESCUED'
        elif f_te > 0.3:
            out = 'PARTIAL'
        else:
            out = 'STUCK'
        print(f'{t_start:>12}  {f_te:>15.4f}  {out:>12}')


if __name__ == '__main__':
    main()
