"""Definitive test: is M a saddle (gradient nonzero) or a basin (gradient zero)?

For each model and each data slice (own_train, own_test, full), compute the
L2 norm of the gradient of the loss with respect to the weights.

Prediction:
  G models (basins on full data):
    grad_own_train ~ 0
    grad_own_test  ~ 0  (G generalizes -- loss minimized on test too)
    grad_full      ~ 0

  M models (saddles on full data):
    grad_own_train ~ 0  (M is at training minimum)
    grad_own_test  >> 0 (M is NOT at minimum on its own test data -- gradient
                         points toward generalization)
    grad_full      >> 0 (because the test gradient is non-trivial)

If we see this pattern, M is empirically demonstrated to be a saddle on the
full-data loss surface, not a basin.

Bonus: also compute how the test gradient COMPARES to a random direction in
weight space. If they're comparable in magnitude, the gradient is genuinely
pointing toward something specific (probably the generalization direction).

Usage:
    python taska/analysis/saddle_test.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch as t

from data import gen_train_test, to_tensors
from model import Transformer

P = 113
D_MODEL = 128


def load_state(p):
    return t.load(p, map_location='cpu', weights_only=True)['model']


def make_model(state):
    m = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    m.load_state_dict(state)
    return m


def compute_gradient_norm(model, inputs, labels):
    """Compute ||grad of cross-entropy loss w.r.t. weights||_2."""
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    logits = model(inputs)[:, -1, :].to(t.float64)
    log_probs = t.nn.functional.log_softmax(logits, dim=-1)
    loss = -log_probs[t.arange(labels.shape[0]), labels].mean()
    loss.backward()

    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += (p.grad ** 2).sum().item()
    return float(np.sqrt(total_sq)), loss.item()


def main():
    checkpoints = {
        'M_s0': HERE / 'checkpoints' / 'M' / 'final.pt',
        'M_s1': HERE / 'checkpoints' / 'M_seed1' / 'final.pt',
        'M_s2': HERE / 'checkpoints' / 'M_seed2' / 'final.pt',
        'G_s0': HERE / 'checkpoints' / 'G' / 'final.pt',
        'G_s1': HERE / 'checkpoints' / 'G_seed1' / 'final.pt',
        'G_s2': HERE / 'checkpoints' / 'G_seed2' / 'final.pt',
    }
    seed_map = {'M_s0': 0, 'M_s1': 1, 'M_s2': 2,
                'G_s0': 0, 'G_s1': 1, 'G_s2': 2}

    # Data slices
    all_pairs = [(i, j, P) for i in range(P) for j in range(P)]
    full_in, full_lab = to_tensors(all_pairs, P, device='cpu')

    splits = {s: gen_train_test(p=P, frac_train=0.3, seed=s) for s in {0, 1, 2}}

    print(f'{"model":>6}  '
          f'{"loss_train":>12}  {"|grad_train|":>14}  '
          f'{"loss_test":>12}  {"|grad_test|":>14}  '
          f'{"loss_full":>12}  {"|grad_full|":>14}')

    results = {}
    for name, ckpt in checkpoints.items():
        own_seed = seed_map[name]
        train_pairs, test_pairs = splits[own_seed]
        train_in, train_lab = to_tensors(train_pairs, P, device='cpu')
        test_in,  test_lab  = to_tensors(test_pairs,  P, device='cpu')

        state = load_state(ckpt)

        gn_train, loss_train = compute_gradient_norm(make_model(state), train_in, train_lab)
        gn_test,  loss_test  = compute_gradient_norm(make_model(state), test_in,  test_lab)
        gn_full,  loss_full  = compute_gradient_norm(make_model(state), full_in,  full_lab)

        results[name] = {
            'loss_train': loss_train, 'grad_train': gn_train,
            'loss_test':  loss_test,  'grad_test':  gn_test,
            'loss_full':  loss_full,  'grad_full':  gn_full,
        }
        print(f'{name:>6}  '
              f'{loss_train:>12.4e}  {gn_train:>14.4e}  '
              f'{loss_test:>12.4e}  {gn_test:>14.4e}  '
              f'{loss_full:>12.4e}  {gn_full:>14.4e}')

    print()
    print('=' * 80)
    print('SUMMARY (gradient norms averaged within category)')
    print('=' * 80)
    for cat in ['M', 'G']:
        members = [n for n in results if n.startswith(cat)]
        for key in ['grad_train', 'grad_test', 'grad_full']:
            vals = [results[n][key] for n in members]
            print(f'  {cat} models | {key:>11}: '
                  f'mean = {np.mean(vals):.4e}  min = {min(vals):.4e}  max = {max(vals):.4e}')
        print()

    # Compare M's test gradient to a "random direction" baseline.
    # Generate a random weight vector with the same total norm as the model,
    # then compute its gradient. (Doesn't actually make sense as a baseline --
    # just sanity check.) Simpler check: ratio of grad_full(M) to grad_full(G).
    M_gradfull = np.mean([results[n]['grad_full'] for n in results if n.startswith('M')])
    G_gradfull = np.mean([results[n]['grad_full'] for n in results if n.startswith('G')])
    print(f"Ratio of M's full-data gradient to G's: {M_gradfull / G_gradfull:.2e}")
    print(f"If this ratio is >> 1, M is at a much steeper point on the full surface than G.")
    print(f"That is the empirical signature of a saddle (M) vs a basin (G).")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names = list(results.keys())
    colors = ['tab:red' if n.startswith('M') else 'tab:blue' for n in names]

    ax = axes[0]
    bar_w = 0.25
    x = np.arange(len(names))
    train_vals = [results[n]['grad_train'] for n in names]
    test_vals  = [results[n]['grad_test']  for n in names]
    full_vals  = [results[n]['grad_full']  for n in names]
    ax.bar(x - bar_w, train_vals, bar_w, label='|grad| on own train', color='tab:green')
    ax.bar(x,         test_vals,  bar_w, label='|grad| on own test',  color='tab:orange')
    ax.bar(x + bar_w, full_vals,  bar_w, label='|grad| on full',      color='tab:purple')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_yscale('log')
    ax.set_ylabel('gradient L2 norm (log scale)')
    ax.set_title('Gradient norm at each model, per data slice')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    loss_train_vals = [results[n]['loss_train'] for n in names]
    loss_test_vals  = [results[n]['loss_test']  for n in names]
    loss_full_vals  = [results[n]['loss_full']  for n in names]
    ax.bar(x - bar_w, loss_train_vals, bar_w, label='loss on own train', color='tab:green')
    ax.bar(x,         loss_test_vals,  bar_w, label='loss on own test',  color='tab:orange')
    ax.bar(x + bar_w, loss_full_vals,  bar_w, label='loss on full',      color='tab:purple')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_yscale('log')
    ax.set_ylabel('loss (log scale)')
    ax.set_title('Loss at each model, per data slice')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Saddle vs basin: is M's gradient zero on full data, or not?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_saddle_test.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
