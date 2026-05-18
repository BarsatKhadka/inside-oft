"""Saddle gradient direction test: does ∇L_full at M point toward G?

If M is a saddle whose unstable direction is generalization, then the full-data
gradient at M should have positive cosine similarity with (G - M) in weight
space. Direct geometric test of "saddle leans toward generalization."

For each of 3 (M, G) pairs (different seeds), compute:
  - ∇L_full at M (per-parameter gradient on full data)
  - (G_params - M_params) per-parameter
  - cosine similarity between these two flattened vectors

If cosine sim > 0 robustly, the saddle is "informative" — its descent direction
predictably leads toward generalizing solutions.

Usage:
    python bulletproof/saddle_gradient_direction.py
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


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def load_state(p, device):
    return t.load(p, map_location=device, weights_only=True)['model']


def build_model(state, device):
    m = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
    m.load_state_dict(state)
    m.to(device)
    return m


def flatten_state(state):
    return t.cat([v.flatten() for v in state.values() if v.is_floating_point()])


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    all_pairs = [(a, b, P) for a in range(P) for b in range(P)]
    full_in = t.tensor(all_pairs, dtype=t.long, device=device)
    full_lab = t.tensor([(a + b) % P for a, b, _ in all_pairs], device=device)

    pairs = [
        ('seed0', 'taska/checkpoints/M/final.pt',         'taska/checkpoints/G/final.pt'),
        ('seed1', 'taska/checkpoints/M_seed1/final.pt',   'taska/checkpoints/G_seed1/final.pt'),
        ('seed2', 'taska/checkpoints/M_seed2/final.pt',   'taska/checkpoints/G_seed2/final.pt'),
    ]

    results = {}
    for name, m_path, g_path in pairs:
        print(f'\n=== {name} ===')
        s_M = load_state(m_path, device)
        s_G = load_state(g_path, device)
        model = build_model(s_M, device)
        for p in model.parameters():
            p.requires_grad_(True)

        # Compute gradient at M on full data
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        loss = cross_entropy_hp(model(full_in)[:, -1, :], full_lab)
        loss.backward()
        # Use zeros for any param without a gradient (defensive)
        grad_at_M = [p.grad.detach().clone() if p.grad is not None else t.zeros_like(p)
                     for p in model.parameters()]
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()

        # Compute (G - M) per learnable param. Match by name to avoid buffer/param mismatch.
        delta = []
        for name_p, p_M in model.named_parameters():
            if name_p in s_G and p_M.is_floating_point():
                p_G_tensor = s_G[name_p].to(device).to(p_M.dtype)
                if p_G_tensor.shape != p_M.shape:
                    print(f'  WARN: shape mismatch {name_p}: M={tuple(p_M.shape)}, G={tuple(p_G_tensor.shape)}')
                    delta.append(t.zeros_like(p_M))
                else:
                    delta.append(p_G_tensor - p_M.detach())
            else:
                delta.append(t.zeros_like(p_M))

        # Cosine similarity
        flat_grad = t.cat([g.flatten() for g in grad_at_M])
        flat_delta = t.cat([d.flatten() for d in delta])

        cos = t.dot(flat_grad, flat_delta) / (t.norm(flat_grad) * t.norm(flat_delta))
        # Note: we want the NEGATIVE gradient (descent direction) to point toward G
        cos_neg_grad = -cos.item()

        # Norm of the gradient
        grad_norm = float(t.norm(flat_grad))
        # Norm of the delta
        delta_norm = float(t.norm(flat_delta))

        results[name] = {
            'cos_grad_vs_GminusM': float(cos.item()),
            'cos_negGrad_vs_GminusM': cos_neg_grad,
            'grad_norm': grad_norm,
            'delta_norm': delta_norm,
        }
        print(f'  ||∇L_full at M||             = {grad_norm:.4f}')
        print(f'  ||G - M||                    = {delta_norm:.4f}')
        print(f'  cos(∇L_full, G - M)         = {cos.item():.4f}')
        print(f'  cos(-∇L_full, G - M)        = {cos_neg_grad:.4f}  (positive = descent points toward G)')

    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'saddle_gradient_direction.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== Summary ===')
    print('If cos(-∇L_full at M, G - M) is positive across seeds, descent from M points toward G.')
    for name, r in results.items():
        sign = 'YES' if r['cos_negGrad_vs_GminusM'] > 0 else 'NO'
        print(f'  {name}: cos = {r["cos_negGrad_vs_GminusM"]:.4f}  →  descent toward G? {sign}')


if __name__ == '__main__':
    main()
