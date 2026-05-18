"""Long-training full matrix (20k epochs) - fixes the under-trained matrix issue.

Same 48-cell grid as before but with 20k epochs instead of 5k. Properly tests
universality of the rank-WD relationship.

Usage:
    python overnight2/full_matrix_long.py
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
NUM_EPOCHS = 20000
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
    'add':      lambda a, b: (a + b) % P,
    'subtract': lambda a, b: (a - b) % P,
    'mult':     lambda a, b: (a * b) % P,
    'mult_p1':  lambda a, b: (a * b + 1) % P,
}
ARCHS = ['1L_Transf', '4L_Transf', 'MLP']
WDS = [0.0, 0.01, 0.1, 1.0]


def to_tensors(pairs, fn, device):
    inp = t.tensor([(a, b, P) for a, b, _ in pairs], dtype=t.long, device=device)
    lab = t.tensor([fn(a, b) for a, b, _ in pairs], device=device)
    return inp, lab


def grad_norm(model, inputs, labels):
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    cross_entropy_hp(model(inputs)[:, -1, :], labels).backward()
    total = sum((p.grad ** 2).sum().item() for p in model.parameters() if p.grad is not None)
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    return float(np.sqrt(total))


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
        if (ep + 1) % 5000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}')
    tr = eval_acc(model, train_in, train_lab)
    te = eval_acc(model, test_in, test_lab)
    rank = effective_rank(get_deep_W(model, arch))
    g_tr = grad_norm(model, train_in, train_lab)
    g_te = grad_norm(model, test_in, test_lab)
    return {'arch': arch, 'op': op_name, 'wd': wd, 'train_acc': tr, 'test_acc': te,
            'rank': rank, 'grad_train': g_tr, 'grad_test': g_te, 'grok_epoch': grok_ep}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {}
    for arch in ARCHS:
        for op in OPS:
            for wd in WDS:
                results[f'{arch}_{op}_wd{wd}'] = cell(arch, op, wd, device)
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'full_matrix_long.json', 'w') as f:
        json.dump(results, f, indent=2)

    n_pairs = 0; n_M_high = 0; n_G_low_groks = 0
    for arch in ARCHS:
        for op in OPS:
            m = results[f'{arch}_{op}_wd0.0']
            g = results[f'{arch}_{op}_wd1.0']
            n_pairs += 1
            if m['rank'] > 2 * g['rank']: n_M_high += 1
            if g['test_acc'] > 0.95 and g['rank'] < 30: n_G_low_groks += 1
    print(f'\nM_rank > 2x G_rank: {n_M_high}/{n_pairs}')
    print(f'G groks AND rank < 30: {n_G_low_groks}/{n_pairs}')


if __name__ == '__main__':
    main()
