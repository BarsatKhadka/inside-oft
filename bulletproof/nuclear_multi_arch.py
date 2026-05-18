"""Nuclear norm penalty on multiple architectures — is rank-IS-mechanism universal?

Replicate the nuclear_norm finding (Entry 52) on 3 architectures with multi-seed.
If nuclear norm escapes in all 3 archs, rank-IS-mechanism is universal across
architecture (not just 1L Transformer specific).

3 archs × 3 lambdas × 2 seeds = 18 runs × 15k epochs.

Usage:
    python bulletproof/nuclear_multi_arch.py
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
NUM_EPOCHS = 15000


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def effective_rank(W):
    s = t.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def nuclear_norm_penalty(model):
    total = 0.0
    for name, p in model.named_parameters():
        if p.ndim >= 2 and 'W_' in name:
            W = p.reshape(p.shape[0], -1)
            try:
                s = t.linalg.svdvals(W)
                total = total + s.sum()
            except Exception:
                pass
    return total


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


def cell(arch, lam, seed, device, train_in, train_lab, test_in, test_lab):
    print(f'\n--- {arch} lam={lam} seed={seed} ---')
    model = build(arch, seed, device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))
    grok_ep = None
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        if lam > 0:
            loss = loss + lam * nuclear_norm_penalty(model)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 500 == 0:
            te = eval_acc(model, test_in, test_lab)
            if grok_ep is None and te >= 0.95:
                grok_ep = ep + 1
        if (ep + 1) % 3000 == 0:
            print(f'  ep={ep+1}: test_acc={te:.4f}')
    return {'arch': arch, 'lam': lam, 'seed': seed,
            'final_test_acc': eval_acc(model, test_in, test_lab),
            'final_rank': effective_rank(get_deep_W(model, arch)),
            'grok_epoch': grok_ep}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    results = {}
    for arch in ['1L_Transf', '4L_Transf', 'MLP']:
        for lam in [1e-4, 5e-4, 1e-3]:
            for seed in [0, 1]:
                key = f'{arch}_lam{lam}_seed{seed}'
                results[key] = cell(arch, lam, seed, device, train_in, train_lab, test_in, test_lab)

    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'nuclear_multi_arch.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== Nuclear norm escape across architectures ===')
    for k, r in results.items():
        print(f'  {k}: test_acc={r["final_test_acc"]:.4f}, rank={r["final_rank"]:.2f}, grok@{r["grok_epoch"]}')


if __name__ == '__main__':
    main()
