"""Hessian eigenvalues at M vs G: direct saddle-vs-basin measurement.

If M is genuinely a saddle on the full-data loss surface, its Hessian should
have NEGATIVE eigenvalues. G's Hessian should be positive semi-definite (true
local minimum).

We approximate top-k positive and bottom-k negative eigenvalues via the
Lanczos algorithm (power iteration) for both M and G.

This is the DIRECT geometric measurement of saddle vs basin. Beyond gradient
asymmetry, this measures whether the model is at a stationary point with
descent directions.

For M_seed0 and G_seed0, compute:
  - Top 5 Hessian eigenvalues on full data
  - Bottom 5 Hessian eigenvalues on full data (most negative)
  - Same restricted to train data only
  - Same restricted to test data only

If M has clearly negative eigenvalues on full data but G doesn't, the saddle
claim is geometrically verified.

Usage:
    python overnight2/hessian_eigenvalues.py
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch as t

from taska.data import gen_train_test, to_tensors
from taska.model import Transformer

P = 113


def load_model(ckpt_path, device):
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt_path, map_location=device, weights_only=True)['model']
    model.load_state_dict(state)
    model.to(device)
    return model


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def hvp(model, loss, vector_list):
    """Hessian-vector product. Returns Hv where H is Hessian of loss wrt model params."""
    grads = t.autograd.grad(loss, list(model.parameters()), create_graph=True)
    dot = sum((g * v).sum() for g, v in zip(grads, vector_list))
    Hv = t.autograd.grad(dot, list(model.parameters()), retain_graph=False)
    return list(Hv)


def power_iteration_top(model, inputs, labels, n_iter=30, sign=+1):
    """Compute top eigenvalue of (sign * H). For sign=+1: max eigenvalue.
    For sign=-1: most negative eigenvalue (since max of -H is most negative of H)."""
    device = next(model.parameters()).device
    params = list(model.parameters())
    v = [t.randn_like(p) for p in params]
    norm = t.sqrt(sum((vi ** 2).sum() for vi in v))
    v = [vi / norm for vi in v]

    eigenvalue = 0.0
    for i in range(n_iter):
        loss = cross_entropy_hp(model(inputs)[:, -1, :], labels)
        Hv = hvp(model, loss, v)
        if sign == -1:
            Hv = [-x for x in Hv]
        # Estimate eigenvalue (Rayleigh quotient)
        eigenvalue = sum((vi * hi).sum().item() for vi, hi in zip(v, Hv))
        norm = t.sqrt(sum((hi ** 2).sum() for hi in Hv))
        if norm < 1e-12:
            break
        v = [hi / norm for hi in Hv]
    if sign == -1:
        eigenvalue = -eigenvalue
    return eigenvalue


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)
    all_pairs = [(a, b, P) for a in range(P) for b in range(P)]
    full_in = t.tensor(all_pairs, dtype=t.long, device=device)
    full_lab = t.tensor([(a + b) % P for a, b, _ in all_pairs], device=device)

    results = {}
    for name in ['M', 'G']:
        print(f'\n=== {name} model ===')
        model = load_model(Path('taska/checkpoints') / name / 'final.pt', device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)
        for data_name, (inp, lab) in [('train', (train_in, train_lab)),
                                       ('test',  (test_in,  test_lab)),
                                       ('full',  (full_in,  full_lab))]:
            print(f'  computing top + bottom eigenvalues on {data_name}...')
            top_ev = power_iteration_top(model, inp, lab, n_iter=40, sign=+1)
            bot_ev = power_iteration_top(model, inp, lab, n_iter=40, sign=-1)
            results.setdefault(name, {})[data_name] = {
                'top_eigenvalue': top_ev, 'bottom_eigenvalue': bot_ev,
                'saddle_signature': bot_ev < -1e-3,
            }
            print(f'    top_ev={top_ev:.4f}, bot_ev={bot_ev:.4f}')

    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'hessian_eigenvalues.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== Saddle signature: negative Hessian eigenvalue exists? ===')
    for name in ['M', 'G']:
        for ds in ['train', 'test', 'full']:
            r = results[name][ds]
            print(f'  {name} on {ds}: bot_ev={r["bottom_eigenvalue"]:.4f}, '
                  f'saddle={"YES" if r["saddle_signature"] else "no"}')


if __name__ == '__main__':
    main()
