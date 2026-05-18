"""Rule-out test: do OTHER regularizers escape M's saddle?

WD escapes. But is it because of L2 norm penalty specifically, or any
explicit regularizer that biases toward simplicity?

Test 5 alternative regularizers (no WD, fresh M_50000 each):
  1. L1 regularization (weight sparsity)
  2. Spectral norm penalty (penalize ||W||_op^2)
  3. Frobenius norm penalty (same as L2 but applied explicitly to loss, no decoupled WD)
  4. Dropout (added at the MLP layer)
  5. Label smoothing (smooth target distribution)

Each runs 20k epochs starting from M_50000. Does it escape?

If only some escape (e.g., L1, spectral norm) and others don't (dropout, label
smoothing) → the bias toward LOW NORM specifically is the key, not just
"any regularizer."

If all escape → broader claim, "explicit complexity penalties escape."

If only WD/Frobenius escape → very tight: only L2-style weight shrinkage works.

Usage:
    python taska/analysis/alternative_regularizers.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import json
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
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


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def label_smoothing_loss(logits, labels, smoothing=0.1):
    n_classes = logits.shape[-1]
    log_probs = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    nll = -log_probs[t.arange(labels.shape[0]), labels]
    uniform = -log_probs.mean(dim=-1)
    return ((1 - smoothing) * nll + smoothing * uniform).mean()


def L1_penalty(model):
    return sum(p.abs().sum() for p in model.parameters())


def L2_penalty(model):
    return sum((p ** 2).sum() for p in model.parameters())


def spectral_penalty(model):
    """sum of operator norm squared of each weight matrix."""
    pen = 0.0
    for name, p in model.named_parameters():
        if p.ndim >= 2 and 'W_' in name:
            w_flat = p.reshape(p.shape[0], -1) if p.ndim > 2 else p
            s_max = t.linalg.svdvals(w_flat)[0]
            pen = pen + s_max ** 2
    return pen


def run_regularizer(name, loss_fn, M_state, device, train_in, train_lab, test_in, test_lab):
    print(f'\n=== {name} ===')
    model = make_model(M_state, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))

    history = {'epoch': [], 'test_acc': [], 'train_loss': []}
    for ep in range(NUM_EPOCHS):
        loss = loss_fn(model, train_in, train_lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (ep + 1) % LOG_EVERY == 0:
            te = ((model(test_in)[:, -1, :].argmax(dim=-1) == test_lab).float().mean().item())
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
            history['train_loss'].append(loss.item())
        if (ep + 1) % 4000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}, loss={loss.item():.4f}')

    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    M_state = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    histories = {}

    # 1. L1 penalty
    def loss_L1(model, x, y):
        return cross_entropy_hp(model(x)[:, -1, :], y) + 1e-4 * L1_penalty(model)
    histories['L1_1e-4'] = run_regularizer('L1 (1e-4)', loss_L1, M_state, device, train_in, train_lab, test_in, test_lab)

    # 2. Explicit L2 (Frobenius, in loss not WD)
    def loss_L2(model, x, y):
        return cross_entropy_hp(model(x)[:, -1, :], y) + 1e-3 * L2_penalty(model)
    histories['L2_1e-3'] = run_regularizer('L2 in loss (1e-3)', loss_L2, M_state, device, train_in, train_lab, test_in, test_lab)

    # 3. Spectral penalty
    def loss_spec(model, x, y):
        return cross_entropy_hp(model(x)[:, -1, :], y) + 1e-2 * spectral_penalty(model)
    histories['spectral_1e-2'] = run_regularizer('spectral (1e-2)', loss_spec, M_state, device, train_in, train_lab, test_in, test_lab)

    # 4. Label smoothing
    def loss_ls(model, x, y):
        return label_smoothing_loss(model(x)[:, -1, :], y, smoothing=0.1)
    histories['label_smooth_0.1'] = run_regularizer('label smoothing (0.1)', loss_ls, M_state, device, train_in, train_lab, test_in, test_lab)

    # 5. Dropout — needs to be applied to the model itself.
    #    We don't have dropout layers built in; skip or fake with manual masking.
    #    Skip for now and note in paper.

    out_json = HERE / 'results' / 'alternative_regularizers.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(histories, f)
    print(f'\nhistories -> {out_json}')

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, h in histories.items():
        ax.plot(h['epoch'], h['test_acc'], marker='.', label=name)
    ax.set_xlabel('rescue epoch')
    ax.set_ylabel('test accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Alternative regularizers: which escape M\'s saddle?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = HERE / 'results' / 'fig_alternative_regularizers.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')

    print('\n=== Outcomes ===')
    for name, h in histories.items():
        e = next((h['epoch'][i] for i, a in enumerate(h['test_acc']) if a >= 0.95), None)
        print(f'  {name:25s}: final={h["test_acc"][-1]:.4f}, grok @ {e}')


if __name__ == '__main__':
    main()
