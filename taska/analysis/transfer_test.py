"""α. Transfer test: is M's representation reusable for a different modular task?

Take a frozen base model (M, G, or random) and train ONLY a new unembedding
matrix W_U on a different modular task: (a - b) mod p.

If M's frozen body + fresh W_U learns faster than random_init + fresh W_U,
then M's representations are TRANSFERABLE -- M has learned features useful
beyond the specific (a+b) lookup.

Four configurations compared:
  1. M frozen, fresh W_U trained on (a - b) mod p
  2. G frozen, fresh W_U trained on (a - b) mod p
  3. Random-init body, fresh W_U trained on (a - b) mod p (linear-probe baseline)
  4. Full fresh model trained end-to-end on (a - b) mod p (full-training reference)

Tracks test accuracy over rescue/training epochs for each.

Expected outcomes:
  - If M >> G > random: M has more transferable structure than G. Interesting.
  - If M ≈ G > random: both encode useful (a, b) features. M is at least as good
    as G as a pretrained encoder.
  - If M ≈ G ≈ random: frozen-body transfer doesn't work for this task.
  - If full training >> all frozen: features are task-specific, transfer fails.

Usage:
    python taska/analysis/transfer_test.py
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

from data import gen_train_test
from model import Transformer

P = 113
D_MODEL = 128
SEED = 0
LR = 1e-3
WEIGHT_DECAY = 1.0
NUM_EPOCHS = 10000
LOG_EVERY = 100


def load_state(p):
    return t.load(p, map_location='cpu', weights_only=True)['model']


def make_model(device):
    m = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    m.to(device)
    return m


def freeze_all_except_unembed(model):
    for name, param in model.named_parameters():
        if 'unembed' not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    # Reinitialize W_U
    with t.no_grad():
        model.unembed.W_U.normal_(0, 1.0 / np.sqrt(P + 1))


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def full_loss(model, inputs, labels):
    return cross_entropy_hp(model(inputs)[:, -1, :], labels)


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def new_task_subtraction_labels(pairs):
    """Task: (a - b) mod p"""
    return t.tensor([(a - b) % P for a, b, _ in pairs])


def run_config(name, init_source, device, freeze=True):
    """
    init_source: 'M', 'G', 'random' (controls initial weights of body).
    freeze: whether to freeze the body and only train W_U.
    """
    print(f'\n=== {name} ===')
    model = make_model(device)
    if init_source == 'M':
        s = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')
        model.load_state_dict(s)
    elif init_source == 'G':
        s = load_state(HERE / 'checkpoints' / 'G' / 'final.pt')
        model.load_state_dict(s)
    elif init_source == 'random':
        pass  # already random init from constructor

    if freeze:
        freeze_all_except_unembed(model)
        params = [p for p in model.parameters() if p.requires_grad]
        n = sum(p.numel() for p in params)
        print(f'  Frozen body. Trainable params: {n}')
    else:
        # Also reinit W_U for fairness with the frozen configs
        with t.no_grad():
            model.unembed.W_U.normal_(0, 1.0 / np.sqrt(P + 1))
        params = list(model.parameters())
        n = sum(p.numel() for p in params)
        print(f'  All trainable. Total params: {n}')

    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))

    # Data for subtraction task
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in = t.tensor(train_pairs, dtype=t.long, device=device)
    test_in  = t.tensor(test_pairs,  dtype=t.long, device=device)
    train_lab = new_task_subtraction_labels(train_pairs).to(device)
    test_lab  = new_task_subtraction_labels(test_pairs).to(device)

    history = {'epoch': [], 'train_acc': [], 'test_acc': []}

    tr = eval_acc(model, train_in, train_lab)
    te = eval_acc(model, test_in, test_lab)
    history['epoch'].append(0); history['train_acc'].append(tr); history['test_acc'].append(te)
    print(f'  start:  train_acc={tr:.4f}  test_acc={te:.4f}')

    for ep in range(NUM_EPOCHS):
        loss = full_loss(model, train_in, train_lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (ep + 1) % LOG_EVERY == 0:
            tr = eval_acc(model, train_in, train_lab)
            te = eval_acc(model, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['train_acc'].append(tr)
            history['test_acc'].append(te)
        if (ep + 1) % 1000 == 0:
            print(f'  ep={ep+1:5d}  train_acc={tr:.4f}  test_acc={te:.4f}')
    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print(f'Task: (a - b) mod {P}.  num_epochs={NUM_EPOCHS}, lr={LR}, wd={WEIGHT_DECAY}')

    histories = {
        'M-frozen+freshU':    run_config('M-frozen+freshU',    'M',      device, freeze=True),
        'G-frozen+freshU':    run_config('G-frozen+freshU',    'G',      device, freeze=True),
        'Random-frozen+freshU': run_config('Random-frozen+freshU', 'random', device, freeze=True),
        'Full-fresh':         run_config('Full-fresh',         'random', device, freeze=False),
    }

    # Save
    out_json = HERE / 'results' / 'transfer_test.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(histories, f)
    print(f'\nhistories -> {out_json}')

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'M-frozen+freshU': 'tab:red', 'G-frozen+freshU': 'tab:blue',
              'Random-frozen+freshU': 'tab:gray', 'Full-fresh': 'tab:green'}
    for name, h in histories.items():
        ax.plot(h['epoch'], h['test_acc'], marker='.', label=name, color=colors[name], alpha=0.85)
    ax.set_xlabel('epoch')
    ax.set_ylabel('test accuracy on (a - b) mod p')
    ax.set_title('Transfer: does M\'s frozen body help learn a new task faster?')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = HERE / 'results' / 'fig_transfer_test.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')

    # Summary
    print('\n=== Summary (epoch at which test_acc first >= 0.95) ===')
    for name, h in histories.items():
        ep_reach = next((h['epoch'][i] for i, a in enumerate(h['test_acc']) if a >= 0.95), None)
        print(f'  {name:30s}: {"NEVER" if ep_reach is None else f"epoch {ep_reach}"}')


if __name__ == '__main__':
    main()
