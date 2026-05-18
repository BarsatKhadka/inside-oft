"""Cross-architecture escape mechanism test.

For 3 architectures (1-layer transformer, 4-layer transformer, MLP), train
M (no WD) until memorized, then test 4 escape mechanisms:
  WD=1.0, SAM rho=0.2, noise std=0.01, L2-in-loss

If the "only norm-based escape" pattern holds across all 3 architectures,
the claim is architecture-universal.

Reuses Track A's (a+b) mod 113 task.

Usage:
    python taska/analysis/cross_arch_escape.py
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
LR = 1e-3
M_EPOCHS = 30000
RESCUE_EPOCHS = 15000
LOG_EVERY = 500


class MLPModel(nn.Module):
    def __init__(self, p=P, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(p + 1, 128)
        self.fc1 = nn.Linear(128 * 3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, p + 1)

    def forward(self, x):
        # x: (B, 3)
        e = self.emb(x).flatten(1)
        h = t.nn.functional.relu(self.fc1(e))
        h = t.nn.functional.relu(self.fc2(h))
        out = self.fc3(h)
        # Pretend it's at position 2
        return out.unsqueeze(1).expand(-1, 3, -1)


def build_model(arch_name, device):
    if arch_name == '1L_Transformer':
        return Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
    elif arch_name == '4L_Transformer':
        return Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=4).to(device)
    elif arch_name == 'MLP':
        return MLPModel(p=P, hidden=512).to(device)


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def full_loss(model, inputs, labels):
    return cross_entropy_hp(model(inputs)[:, -1, :], labels)


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def L2_pen(model):
    return sum((p ** 2).sum() for p in model.parameters())


def train_M(arch_name, device, train_in, train_lab):
    """Train M (no WD) until memorized."""
    print(f'  Training M for {arch_name}...')
    t.manual_seed(0)
    model = build_model(arch_name, device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))
    for ep in range(M_EPOCHS):
        loss = full_loss(model, train_in, train_lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 5000 == 0:
            print(f'    M_ep={ep+1}: loss={loss.item():.4e}')
    return model.state_dict()


def rescue(arch_name, mechanism_name, M_state, device, train_in, train_lab, test_in, test_lab):
    print(f'\n  === {arch_name} | {mechanism_name} ===')
    model = build_model(arch_name, device)
    model.load_state_dict(M_state)
    history = {'epoch': [], 'test_acc': []}

    if mechanism_name == 'WD_1.0':
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0, betas=(0.9, 0.98))
        loss_fn = lambda m, x, y: full_loss(m, x, y)
        sam_rho = None
        noise_std = None
    elif mechanism_name == 'SAM_0.2':
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))
        loss_fn = lambda m, x, y: full_loss(m, x, y)
        sam_rho = 0.2
        noise_std = None
    elif mechanism_name == 'noise_0.01':
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))
        loss_fn = lambda m, x, y: full_loss(m, x, y)
        sam_rho = None
        noise_std = 0.01
    elif mechanism_name == 'L2_in_loss':
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))
        loss_fn = lambda m, x, y: full_loss(m, x, y) + 1e-3 * L2_pen(m)
        sam_rho = None
        noise_std = None

    for ep in range(RESCUE_EPOCHS):
        loss = loss_fn(model, train_in, train_lab)
        if sam_rho is not None:
            opt.zero_grad(); loss.backward()
            with t.no_grad():
                grads = [p.grad.detach().clone() if p.grad is not None else t.zeros_like(p) for p in model.parameters()]
                gn = t.sqrt(sum((g**2).sum() for g in grads)) + 1e-12
                scale = sam_rho / gn
                saved = [p.detach().clone() for p in model.parameters()]
                for p, g in zip(model.parameters(), grads):
                    p.add_(g * scale)
            loss2 = loss_fn(model, train_in, train_lab)
            opt.zero_grad(); loss2.backward()
            with t.no_grad():
                for p, s in zip(model.parameters(), saved):
                    p.copy_(s)
            opt.step()
        else:
            opt.zero_grad(); loss.backward(); opt.step()

        if noise_std is not None:
            with t.no_grad():
                for p in model.parameters():
                    p.add_(t.randn_like(p) * noise_std)

        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
        if (ep + 1) % 3000 == 0:
            print(f'    ep={ep+1}: test_acc={te:.4f}')
    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    ARCHS = ['1L_Transformer', '4L_Transformer', 'MLP']
    MECHS = ['WD_1.0', 'SAM_0.2', 'noise_0.01', 'L2_in_loss']

    results = {}
    for arch in ARCHS:
        M_state = train_M(arch, device, train_in, train_lab)
        results[arch] = {}
        for mech in MECHS:
            h = rescue(arch, mech, M_state, device, train_in, train_lab, test_in, test_lab)
            results[arch][mech] = h

    out_json = HERE / 'results' / 'cross_arch_escape.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f)

    # Plot
    fig, axes = plt.subplots(1, len(ARCHS), figsize=(5 * len(ARCHS), 4), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        for mech in MECHS:
            h = results[arch][mech]
            ax.plot(h['epoch'], h['test_acc'], marker='.', label=mech)
        ax.set_xlabel('rescue epoch')
        ax.set_ylabel('test accuracy')
        ax.set_title(arch)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Cross-arch: does the 'only norm-based escapes' pattern hold?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_cross_arch_escape.png'
    fig.savefig(out, dpi=130)
    print(f'\nplot -> {out}')

    print('\n=== Outcomes ===')
    for arch in ARCHS:
        for mech in MECHS:
            h = results[arch][mech]
            grok = next((h['epoch'][i] for i, a in enumerate(h['test_acc']) if a >= 0.95), None)
            print(f'  {arch:20s} | {mech:15s}: final={h["test_acc"][-1]:.4f}, grok @ {grok}')


if __name__ == '__main__':
    main()
