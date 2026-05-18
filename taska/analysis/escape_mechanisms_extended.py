"""Extended escape-mechanism comparison: what's the COMPLETE space?

Tests 12 mechanisms starting from M_50000, 15k epochs each:
  Family A — WD-like:
    1. WD=1.0 (control, known to escape)
    2. WD=0.1
    3. L2 in loss
    4. L1 in loss

  Family B — Sharpness:
    5. SAM rho=0.05
    6. SAM rho=0.2
    7. SAM rho=0.5

  Family C — Noise:
    8. Gaussian noise std=0.001
    9. Gaussian noise std=0.01
   10. Gaussian noise std=0.1

  Family D — Loss-shaping:
   11. Label smoothing α=0.1
   12. Label smoothing α=0.5

Comprehensive test of "which families escape." If only family A escapes,
the low-norm bias is THE mechanism. If A and B (with strong enough rho)
both escape, the broader class is "complexity-finding regularizers."

Usage:
    python taska/analysis/escape_mechanisms_extended.py
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
NUM_EPOCHS = 15000
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


def full_loss(model, inputs, labels):
    return cross_entropy_hp(model(inputs)[:, -1, :], labels)


def label_smooth_loss(logits, labels, alpha):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    n = lp.shape[-1]
    nll = -lp[t.arange(labels.shape[0]), labels]
    uniform = -lp.mean(dim=-1)
    return ((1 - alpha) * nll + alpha * uniform).mean()


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def L1_penalty(model):
    return sum(p.abs().sum() for p in model.parameters())


def L2_penalty(model):
    return sum((p ** 2).sum() for p in model.parameters())


def run_mechanism(name, build_loss, M_state, device, train_in, train_lab, test_in, test_lab,
                   wd=0.0, sam_rho=None, noise_std=None):
    print(f'\n=== {name} ===')
    model = make_model(M_state, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))

    history = {'epoch': [], 'test_acc': []}

    for ep in range(NUM_EPOCHS):
        loss = build_loss(model, train_in, train_lab)

        if sam_rho is not None:
            # SAM step: take a step in adversarial direction, recompute, restore, step
            optimizer.zero_grad()
            loss.backward()
            with t.no_grad():
                grads = [p.grad.detach().clone() if p.grad is not None else t.zeros_like(p) for p in model.parameters()]
                gn = t.sqrt(sum((g ** 2).sum() for g in grads)) + 1e-12
                scale = sam_rho / gn
                saved = [p.detach().clone() for p in model.parameters()]
                for p, g in zip(model.parameters(), grads):
                    p.add_(g * scale)
            loss2 = build_loss(model, train_in, train_lab)
            optimizer.zero_grad()
            loss2.backward()
            with t.no_grad():
                for p, s in zip(model.parameters(), saved):
                    p.copy_(s)
            optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if noise_std is not None and noise_std > 0:
            with t.no_grad():
                for p in model.parameters():
                    p.add_(t.randn_like(p) * noise_std)

        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
        if (ep + 1) % 3000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}')

    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    M_state = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    base_loss = lambda m, x, y: full_loss(m, x, y)

    configs = [
        # (name, build_loss, kwargs)
        ('A1_WD_1.0',       base_loss, {'wd': 1.0}),
        ('A2_WD_0.1',       base_loss, {'wd': 0.1}),
        ('A3_L2_in_loss',  lambda m, x, y: full_loss(m, x, y) + 1e-3 * L2_penalty(m), {}),
        ('A4_L1_in_loss',  lambda m, x, y: full_loss(m, x, y) + 1e-4 * L1_penalty(m), {}),

        ('B1_SAM_0.05',     base_loss, {'sam_rho': 0.05}),
        ('B2_SAM_0.2',      base_loss, {'sam_rho': 0.2}),
        ('B3_SAM_0.5',      base_loss, {'sam_rho': 0.5}),

        ('C1_noise_1e-3',   base_loss, {'noise_std': 1e-3}),
        ('C2_noise_1e-2',   base_loss, {'noise_std': 1e-2}),
        ('C3_noise_1e-1',   base_loss, {'noise_std': 1e-1}),

        ('D1_LS_0.1',       lambda m, x, y: label_smooth_loss(m(x)[:, -1, :], y, 0.1), {}),
        ('D2_LS_0.5',       lambda m, x, y: label_smooth_loss(m(x)[:, -1, :], y, 0.5), {}),
    ]

    histories = {}
    for name, build_loss, kw in configs:
        histories[name] = run_mechanism(name, build_loss, M_state, device,
                                         train_in, train_lab, test_in, test_lab, **kw)

    out_json = HERE / 'results' / 'escape_mechanisms_extended.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(histories, f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    families = {
        'Family A (WD/L1/L2)': ['A1_WD_1.0', 'A2_WD_0.1', 'A3_L2_in_loss', 'A4_L1_in_loss'],
        'Family B (SAM)':       ['B1_SAM_0.05', 'B2_SAM_0.2', 'B3_SAM_0.5'],
        'Family C (Noise)':     ['C1_noise_1e-3', 'C2_noise_1e-2', 'C3_noise_1e-1'],
        'Family D (Label smooth)': ['D1_LS_0.1', 'D2_LS_0.5'],
    }
    for ax, (title, members) in zip(axes.flat, families.items()):
        for name in members:
            h = histories[name]
            ax.plot(h['epoch'], h['test_acc'], marker='.', label=name)
        ax.set_xlabel('rescue epoch')
        ax.set_ylabel('test accuracy')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Comprehensive escape-mechanism comparison')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_escape_mechanisms_extended.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')

    print('\n=== Outcomes ===')
    for name, h in histories.items():
        grok = next((h['epoch'][i] for i, a in enumerate(h['test_acc']) if a >= 0.95), None)
        print(f'  {name:25s}: final={h["test_acc"][-1]:.4f}, grok @ {grok}')


if __name__ == '__main__':
    main()
