"""WD threshold across (arch, task) cells: is the threshold predictable?

For 6 (arch, task) combinations, find the minimum WD that causes escape.
If our story is right, the threshold should be predictable: it should be the
WD that compresses rank to the task's target rank.

Architectures × tasks:
  - 1L Transformer × (a+b)
  - 1L Transformer × (a*b)
  - 4L Transformer × (a+b)
  - 4L Transformer × (a*b)
  - MLP × (a+b)
  - MLP × (a*b)

For each, sweep WD ∈ {0.01, 0.05, 0.1, 0.3, 1.0, 3.0}.
Each cell: 12k epochs.

6 × 6 = 36 cells.

If threshold is predictable from rank dynamics (lower threshold for harder
tasks, similar across architectures of similar capacity), the unifying story
is strengthened.

Usage:
    python overnight2/wd_threshold_universal.py
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

from taska.data import gen_train_test
from taska.model import Transformer

P = 113
LR = 1e-3
NUM_EPOCHS = 12000
LOG_EVERY = 500


def effective_rank(W):
    s = t.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


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


def build(arch, device):
    t.manual_seed(0)
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


OPS = {
    'add':  lambda a, b: (a + b) % P,
    'mult': lambda a, b: (a * b) % P,
}

@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def to_tensors(pairs, fn, device):
    inp = t.tensor([(a, b, P) for a, b, _ in pairs], dtype=t.long, device=device)
    lab = t.tensor([fn(a, b) for a, b, _ in pairs], device=device)
    return inp, lab


def cell(arch, op_name, wd, device):
    print(f'\n--- {arch} {op_name} wd={wd} ---')
    model = build(arch, device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    fn = OPS[op_name]
    train_in, train_lab = to_tensors(train_pairs, fn, device)
    test_in,  test_lab  = to_tensors(test_pairs,  fn, device)
    grok_ep = None
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            if grok_ep is None and te >= 0.95:
                grok_ep = ep + 1
    rank = effective_rank(get_deep_W(model, arch))
    return {'arch': arch, 'op': op_name, 'wd': wd,
            'final_test_acc': eval_acc(model, test_in, test_lab),
            'final_rank': rank, 'grok_epoch': grok_ep}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    WDS = [0.01, 0.05, 0.1, 0.3, 1.0, 3.0]
    ARCHS = ['1L_Transf', '4L_Transf', 'MLP']
    results = {}
    for arch in ARCHS:
        for op in OPS:
            for wd in WDS:
                results[f'{arch}_{op}_wd{wd}'] = cell(arch, op, wd, device)
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'wd_threshold_universal.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== WD threshold matrix ===')
    print(f'{\"arch_op\":>20s}  ' + '  '.join(f'wd={w:>5}' for w in WDS))
    for arch in ARCHS:
        for op in OPS:
            row = []
            for wd in WDS:
                r = results[f'{arch}_{op}_wd{wd}']
                marker = '✓' if r['grok_epoch'] else '✗'
                row.append(f'{marker} r{r[\"final_rank\"]:>3.0f}')
            print(f'{arch+\"_\"+op:>20s}  ' + '  '.join(f'{x:>8}' for x in row))


if __name__ == '__main__':
    main()
