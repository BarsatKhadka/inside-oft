"""WD threshold across architectures: is the 0.5 threshold universal?

For 4 architectures × 6 WD values, train from random init (NOT rescue) and
measure final test accuracy. Does the WD threshold above which the model
groks vary by architecture?

This is a different test from rescue: starting from random init lets us see
if WD strength determines whether the model grokks at all.

Reduced epochs (15k) for speed since fresh training tends to grok faster
than rescue with WD=1.0.

Usage:
    python taska/analysis/wd_threshold_per_arch.py
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
NUM_EPOCHS = 15000
LOG_EVERY = 500
WD_VALUES = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]


class MLPModel(nn.Module):
    def __init__(self, p=P, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(p + 1, 128)
        self.fc1 = nn.Linear(128 * 3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, p + 1)

    def forward(self, x):
        e = self.emb(x).flatten(1)
        h = t.nn.functional.relu(self.fc1(e))
        h = t.nn.functional.relu(self.fc2(h))
        out = self.fc3(h)
        return out.unsqueeze(1).expand(-1, 3, -1)


def build_model(arch_name, device):
    t.manual_seed(0)
    if arch_name == '1L_Transformer':
        return Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
    elif arch_name == '2L_Transformer':
        return Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=2).to(device)
    elif arch_name == '4L_Transformer':
        return Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=4).to(device)
    elif arch_name == 'MLP':
        return MLPModel().to(device)


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def train_one(arch, wd, device, train_in, train_lab, test_in, test_lab):
    print(f'\n  === {arch} wd={wd} ===')
    model = build_model(arch, device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))
    grok_epoch = None
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            if grok_epoch is None and te >= 0.95:
                grok_epoch = ep + 1
        if (ep + 1) % 5000 == 0:
            print(f'    ep={ep+1}: test_acc={te:.4f}')
    return {'final_test_acc': eval_acc(model, test_in, test_lab), 'grok_epoch': grok_epoch}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    ARCHS = ['1L_Transformer', '2L_Transformer', '4L_Transformer', 'MLP']
    results = {}
    for arch in ARCHS:
        results[arch] = {}
        for wd in WD_VALUES:
            results[arch][wd] = train_one(arch, wd, device, train_in, train_lab, test_in, test_lab)

    out_json = HERE / 'results' / 'wd_threshold_per_arch.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({a: {str(w): v for w, v in d.items()} for a, d in results.items()}, f)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for arch in ARCHS:
        xs = WD_VALUES
        ys = [results[arch][w]['final_test_acc'] for w in xs]
        ax.plot(xs, ys, marker='o', label=arch)
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_xlabel('weight decay')
    ax.set_ylabel('final test accuracy')
    ax.axhline(0.95, color='gray', linestyle=':', label='grok threshold')
    ax.set_title('Is WD threshold universal across architectures?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = HERE / 'results' / 'fig_wd_threshold_per_arch.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')

    # Print table
    print('\n=== WD threshold per architecture ===')
    for arch in ARCHS:
        print(f'  {arch}:')
        for wd in WD_VALUES:
            r = results[arch][wd]
            print(f'    wd={wd:>5}: test_acc={r["final_test_acc"]:.4f}, grok@{r["grok_epoch"]}')


if __name__ == '__main__':
    main()
