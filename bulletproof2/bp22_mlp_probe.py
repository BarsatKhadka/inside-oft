"""bp22: MLP probe replication.

Replicates Entry 16 with 2-hidden-layer MLP probes (with selectivity control).

For each model M and G (3 seeds):
  - Extract residual stream at last layer
  - Train MLP probe to predict (a + b) mod P from residual at position -1
    -- WAIT that just tells us if the model knows the answer; we want per-example identity.
  - For per-example identity, train probe to map residual -> example index
    (only train examples). Selectivity = real - control (shuffled labels).
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from _common import (HERE, P, train_one)

NUM_SEEDS = 3
NUM_EPOCHS = 20000
PROBE_EPOCHS = 500


class MLPProbe(nn.Module):
    def __init__(self, d_in, hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out))

    def forward(self, x): return self.net(x)


def train_probe(X, y, n_out, hidden=256, device='cuda'):
    n = X.shape[0]
    idx = t.randperm(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]; X_te, y_te = X[te_idx], y[te_idx]
    probe = MLPProbe(X.shape[1], hidden, n_out).to(device)
    opt = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(PROBE_EPOCHS):
        opt.zero_grad()
        F.cross_entropy(probe(X_tr), y_tr).backward(); opt.step()
    with t.no_grad():
        return float((probe(X_te).argmax(1) == y_te).float().mean())


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    for label, wd in [('M', 0.0), ('G', 1.0)]:
        for seed in range(NUM_SEEDS):
            print(f'\n=== {label} seed={seed} ===')
            model, meta, (tr_in, tr_lab, te_in, te_lab) = train_one(
                seed=seed, wd=wd, num_epochs=NUM_EPOCHS, device=device)
            # Extract residuals at last position
            with t.no_grad():
                def get_resid(inp):
                    x = model.embed(inp); x = model.pos_embed(x)
                    for b in model.blocks: x = b(x)
                    return x[:, -1, :]
                resid_tr = get_resid(tr_in)
            n_tr = resid_tr.shape[0]
            # Probe 1: per-example identity (real labels)
            idx_lab_real = t.arange(n_tr, device=device)
            acc_real = train_probe(resid_tr, idx_lab_real, n_tr, device=device)
            # Probe 2: control (shuffled per-example labels)
            shuf = t.randperm(n_tr, device=device)
            acc_shuf = train_probe(resid_tr, shuf, n_tr, device=device)
            sel = acc_real - acc_shuf
            # Probe 3: predict (a+b) mod P (true label) from residual
            acc_task = train_probe(resid_tr, tr_lab, P, device=device)
            entry = {
                'label': label, 'seed': seed,
                'final_test_acc': meta['final_test_acc'],
                'probe_identity_acc': acc_real,
                'probe_control_acc': acc_shuf,
                'probe_selectivity': sel,
                'probe_task_acc': acc_task,
            }
            results[label].append(entry)
            print(f'  identity={acc_real:.4f}, control={acc_shuf:.4f}, '
                  f'selectivity={sel:.4f}, task={acc_task:.4f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp22_mlp_probe.json', 'w') as f:
        json.dump(results, f, indent=2)
    for label in ['M', 'G']:
        sels = [r['probe_selectivity'] for r in results[label]]
        print(f'  {label}: selectivity = {np.mean(sels):.4f} +- {np.std(sels):.4f}')


if __name__ == '__main__':
    main()
