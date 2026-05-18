"""Hessian eigenvalues during rescue: do negative eigenvalues disappear as M → G?

Take M_50000 and rescue with WD=1.0. At checkpoints during rescue (every 2k
epochs), compute the most negative Hessian eigenvalue on full data.

Prediction: at M (start), most-negative Hessian eigenvalue is large negative
(saddle direction). As rescue progresses, this becomes less negative.
By end of rescue (in G's basin), all eigenvalues are >= 0 (true minimum).

Plot bottom_eigenvalue vs rescue epoch alongside test accuracy.

This DIRECTLY shows the saddle becoming a basin during the WD rescue.

Usage:
    python bulletproof/hessian_during_rescue.py
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch as t
import torch.optim as optim

from taska.data import gen_train_test, to_tensors
from taska.model import Transformer

P = 113
LR = 1e-3
WD = 1.0
NUM_EPOCHS = 15000
HESS_EVERY = 1500   # measure Hessian every 1500 epochs


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def hvp(loss, params, vector_list):
    grads = t.autograd.grad(loss, params, create_graph=True)
    dot = sum((g * v).sum() for g, v in zip(grads, vector_list))
    Hv = t.autograd.grad(dot, params, retain_graph=False)
    return list(Hv)


def power_iter_top(model, inputs, labels, n_iter=30):
    params = list(model.parameters())
    v = [t.randn_like(p) for p in params]
    nrm = t.sqrt(sum((vi**2).sum() for vi in v))
    v = [vi / nrm for vi in v]
    eig = 0.0
    for _ in range(n_iter):
        for p in params:
            if p.grad is not None: p.grad.zero_()
        loss = cross_entropy_hp(model(inputs)[:, -1, :], labels)
        Hv = hvp(loss, params, v)
        eig = sum((vi*hi).sum().item() for vi, hi in zip(v, Hv))
        nrm = t.sqrt(sum((hi**2).sum() for hi in Hv)) + 1e-12
        v = [hi / nrm for hi in Hv]
    return eig


def power_iter_bottom(model, inputs, labels, alpha, n_iter=30):
    """Spectral shift: top eigenvalue of (alpha*I - H), then return alpha - that."""
    params = list(model.parameters())
    v = [t.randn_like(p) for p in params]
    nrm = t.sqrt(sum((vi**2).sum() for vi in v))
    v = [vi / nrm for vi in v]
    eig_shifted = 0.0
    for _ in range(n_iter):
        for p in params:
            if p.grad is not None: p.grad.zero_()
        loss = cross_entropy_hp(model(inputs)[:, -1, :], labels)
        Hv = hvp(loss, params, v)
        shifted = [alpha * vi - hi for vi, hi in zip(v, Hv)]
        eig_shifted = sum((vi*si).sum().item() for vi, si in zip(v, shifted))
        nrm = t.sqrt(sum((si**2).sum() for si in shifted)) + 1e-12
        v = [si / nrm for si in shifted]
    return alpha - eig_shifted


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)
    all_pairs = [(a, b, P) for a in range(P) for b in range(P)]
    full_in = t.tensor(all_pairs, dtype=t.long, device=device)
    full_lab = t.tensor([(a + b) % P for a, b, _ in all_pairs], device=device)

    # Load M
    M_state = t.load('taska/checkpoints/M/final.pt', map_location=device, weights_only=True)['model']
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
    model.load_state_dict(M_state)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))

    history = {'epoch': [], 'test_acc': [], 'top_eig': [], 'bot_eig': []}

    # Initial measurement
    for p in model.parameters():
        p.requires_grad_(True)
    print('Initial (at M):')
    top = power_iter_top(model, full_in, full_lab)
    alpha = 2 * abs(top) + 1.0
    bot = power_iter_bottom(model, full_in, full_lab, alpha=alpha)
    te = eval_acc(model, test_in, test_lab)
    history['epoch'].append(0)
    history['test_acc'].append(te)
    history['top_eig'].append(top)
    history['bot_eig'].append(bot)
    print(f'  test_acc={te:.4f}, top_eig={top:.4f}, bot_eig={bot:.4f}')

    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % HESS_EVERY == 0:
            print(f'\nep={ep+1}:')
            te = eval_acc(model, test_in, test_lab)
            top = power_iter_top(model, full_in, full_lab)
            alpha = 2 * abs(top) + 1.0
            bot = power_iter_bottom(model, full_in, full_lab, alpha=alpha)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
            history['top_eig'].append(top)
            history['bot_eig'].append(bot)
            print(f'  test_acc={te:.4f}, top_eig={top:.4f}, bot_eig={bot:.4f}')

    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'hessian_during_rescue.json', 'w') as f:
        json.dump(history, f, indent=2)

    print('\n=== Summary ===')
    print(f'{"epoch":>6}  {"test_acc":>9}  {"top_eig":>10}  {"bot_eig":>10}  {"saddle?":>8}')
    for i in range(len(history['epoch'])):
        is_saddle = history['bot_eig'][i] < -1e-3
        print(f'{history["epoch"][i]:>6}  {history["test_acc"][i]:>9.4f}  '
              f'{history["top_eig"][i]:>10.4f}  {history["bot_eig"][i]:>10.4f}  '
              f'{"YES" if is_saddle else "no":>8}')


if __name__ == '__main__':
    main()
