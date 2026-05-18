"""Time-resolved rank trajectory: when does G diverge from M in rank?

Train M and G simultaneously, record rank at every checkpoint. Plot rank
vs epoch for both. See: at what epoch does G's rank start to compress while
M's stays high? Does it match the grokking moment?

For 3 architectures × 2 regimes (M/G) × 3 seeds = 18 runs. Each runs 20k epochs.
Track rank every 200 epochs.

This is the time-resolved analog of capacity sweep — shows the rank-compression
dynamics, not just the endpoint.

Usage:
    python overnight2/rank_trajectory_during_training.py
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim

from taska.data import gen_train_test, to_tensors
from taska.model import Transformer

P = 113
LR = 1e-3
NUM_EPOCHS = 20000
LOG_EVERY = 200


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
        return self.fc3(h).unsqueeze(1).expand(-1, 3, -1)


def build(arch, seed, device):
    t.manual_seed(seed)
    if arch == '1L_Transf':
        m = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
    elif arch == '4L_Transf':
        m = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=4)
    elif arch == 'MLP':
        m = MLPModel()
    return m.to(device)


def get_deep_W(model, arch):
    if arch.endswith('Transf'):
        return model.blocks[-1].mlp.W_out
    return model.fc2.weight


def run_one(arch, wd, seed, device):
    print(f'\n--- {arch} wd={wd} seed={seed} ---')
    model = build(arch, seed, device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=seed)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)
    history = {'epoch': [], 'test_acc': [], 'rank': []}
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % LOG_EVERY == 0:
            history['epoch'].append(ep + 1)
            history['test_acc'].append(eval_acc(model, test_in, test_lab))
            history['rank'].append(effective_rank(get_deep_W(model, arch)))
        if (ep + 1) % 4000 == 0:
            print(f'  ep={ep+1}: test={history["test_acc"][-1]:.4f}, rank={history["rank"][-1]:.2f}')
    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {}
    for arch in ['1L_Transf', '4L_Transf', 'MLP']:
        for wd in [0.0, 1.0]:
            for seed in [0, 1, 2]:
                key = f'{arch}_wd{wd}_seed{seed}'
                results[key] = run_one(arch, wd, seed, device)
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'rank_trajectory_during_training.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('done')


if __name__ == '__main__':
    main()
