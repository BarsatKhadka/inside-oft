"""Fix bp22: probe couldn't fit (3800 classes, 256 hidden, training collapsed).

Fix:
  - Use BUCKETED identity (group N_train examples into 20 clusters)
  - Bigger MLP probe (512 hidden, 4 layers)
  - More probe training (2000 epochs)
  - Also add the simpler "predict (a+b) mod P" probe at multiple layers
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bulletproof2._common import P, train_one
from taska.model import Transformer


NUM_SEEDS = 3
NUM_EPOCHS = 20000
PROBE_EPOCHS = 2000
NUM_BUCKETS = 20


class BigMLPProbe(nn.Module):
    def __init__(self, d_in, hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def train_probe(X, y, n_out, hidden=512, device='cuda', epochs=PROBE_EPOCHS, lr=1e-3):
    n = X.shape[0]
    perm = t.randperm(n)
    split = int(0.8 * n)
    tr, te = perm[:split], perm[split:]
    X_tr, y_tr = X[tr], y[tr]; X_te, y_te = X[te], y[te]
    probe = BigMLPProbe(X.shape[1], hidden, n_out).to(device)
    opt = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        opt.zero_grad()
        F.cross_entropy(probe(X_tr), y_tr).backward()
        opt.step()
    with t.no_grad():
        return float((probe(X_te).argmax(1) == y_te).float().mean())


@t.no_grad()
def get_resid(model, inp):
    x = model.embed(inp); x = model.pos_embed(x)
    for b in model.blocks: x = b(x)
    return x[:, -1, :]


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    for label, wd in [('M', 0.0), ('G', 1.0)]:
        for seed in range(NUM_SEEDS):
            print(f'\n=== {label} seed={seed} ===')
            model, meta, (tr_in, tr_lab, te_in, te_lab) = train_one(
                seed=seed, wd=wd, num_epochs=NUM_EPOCHS, device=device)
            resid_tr = get_resid(model, tr_in)
            n_tr = resid_tr.shape[0]
            # Bucketed identity labels: each example gets a random bucket id, fixed for the run
            t.manual_seed(seed * 7919)
            bucket_lab = t.randint(0, NUM_BUCKETS, (n_tr,), device=device)
            acc_bucket_real = train_probe(resid_tr, bucket_lab, NUM_BUCKETS, device=device)
            # Control: completely random labels (re-shuffled per call)
            shuf = bucket_lab[t.randperm(n_tr, device=device)]
            acc_bucket_ctrl = train_probe(resid_tr, shuf, NUM_BUCKETS, device=device)
            sel_bucket = acc_bucket_real - acc_bucket_ctrl
            # Task probe: predict (a+b) mod P from residual
            acc_task = train_probe(resid_tr, tr_lab, P, device=device)
            entry = {
                'label': label, 'seed': seed,
                'final_test_acc': meta['final_test_acc'],
                'probe_bucket_real_acc': acc_bucket_real,
                'probe_bucket_ctrl_acc': acc_bucket_ctrl,
                'probe_bucket_selectivity': sel_bucket,
                'probe_task_acc': acc_task,
                'num_buckets': NUM_BUCKETS,
            }
            results[label].append(entry)
            print(f'  bucket_real={acc_bucket_real:.4f}, bucket_ctrl={acc_bucket_ctrl:.4f}, '
                  f'selectivity={sel_bucket:.4f}, task={acc_task:.4f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp_fix_probe.json', 'w') as f:
        json.dump(results, f, indent=2)
    for label in ['M', 'G']:
        sels = [r['probe_bucket_selectivity'] for r in results[label]]
        print(f'  {label}: bucket selectivity = {np.mean(sels):.4f} +- {np.std(sels):.4f}')


if __name__ == '__main__':
    main()
