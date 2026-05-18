"""Shared utilities for bulletproof2 batch."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from taska.data import gen_train_test, to_tensors
from taska.model import Transformer

P = 113
LR = 1e-3


def cross_entropy_hp(logits, labels):
    lp = F.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def effective_rank(W):
    s = t.linalg.svdvals(W.detach().cpu().float())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def grad_norm_full(model, inputs, labels):
    for p_ in model.parameters():
        if p_.grad is not None: p_.grad.zero_()
    cross_entropy_hp(model(inputs)[:, -1, :], labels).backward()
    total = sum((p_.grad ** 2).sum().item() for p_ in model.parameters() if p_.grad is not None)
    for p_ in model.parameters():
        if p_.grad is not None: p_.grad.zero_()
    return float(np.sqrt(total))


def flat_grad(model, inputs, labels):
    for p_ in model.parameters():
        if p_.grad is not None: p_.grad.zero_()
    cross_entropy_hp(model(inputs)[:, -1, :], labels).backward()
    g = t.cat([p_.grad.detach().flatten() for p_ in model.parameters() if p_.grad is not None])
    for p_ in model.parameters():
        if p_.grad is not None: p_.grad.zero_()
    return g


def train_one(seed, wd, num_epochs=20000, p=P, frac_train=0.3, d_model=128, num_heads=4, num_layers=1, device='cuda'):
    """Train a single model and return it + meta."""
    t.manual_seed(seed)
    np.random.seed(seed)
    model = Transformer(p=p, d_model=d_model, num_heads=num_heads, n_ctx=3, num_layers=num_layers).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))
    train_pairs, test_pairs = gen_train_test(p=p, frac_train=frac_train, seed=seed)
    train_in, train_lab = to_tensors(train_pairs, p, device=device)
    test_in, test_lab = to_tensors(test_pairs, p, device=device)
    grok_ep = None
    history = {'epoch': [], 'train_acc': [], 'test_acc': []}
    for ep in range(num_epochs):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 500 == 0:
            te = eval_acc(model, test_in, test_lab)
            tr = eval_acc(model, train_in, train_lab)
            history['epoch'].append(ep + 1); history['train_acc'].append(tr); history['test_acc'].append(te)
            if grok_ep is None and te >= 0.95:
                grok_ep = ep + 1
    meta = {
        'seed': seed, 'wd': wd, 'p': p, 'd_model': d_model, 'num_heads': num_heads,
        'num_layers': num_layers, 'grok_epoch': grok_ep,
        'final_train_acc': eval_acc(model, train_in, train_lab),
        'final_test_acc': eval_acc(model, test_in, test_lab),
        'history': history,
    }
    return model, meta, (train_in, train_lab, test_in, test_lab)
