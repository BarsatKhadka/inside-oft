"""A. Local generalization: does M output 'near-correct' values for inputs near training pairs?

For each of the 12769 (a, b) pairs, compute:
  - Whether it was in M's training set
  - Distance to nearest training pair (cyclic L1 distance on the torus)
  - M's predicted answer
  - Absolute error from the correct answer

If M has local structure: accuracy degrades GRADUALLY with distance from
training pairs. Outputs for nearby-but-unseen pairs are wrong but CLOSE to
the correct value (small absolute error).

If M is pure point lookup: accuracy is binary -- 100% for seen, ~chance for
unseen. Absolute error distribution for unseen is uniform.

Compare to G for reference (G should be uniform across all distances).

Usage:
    python taska/analysis/local_generalization.py
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


def load_model(ckpt):
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt, map_location='cpu', weights_only=True)['model']
    model.load_state_dict(state)
    model.eval()
    return model


def cyclic_dist(a1, b1, a2, b2):
    da = min(abs(a1 - a2), P - abs(a1 - a2))
    db = min(abs(b1 - b2), P - abs(b1 - b2))
    return da + db


def cyclic_abs_diff(x, y):
    d = abs(x - y) % P
    return min(d, P - d)


@t.no_grad()
def all_predictions(model):
    pairs = [(a, b, P) for a in range(P) for b in range(P)]
    inputs = t.tensor(pairs, dtype=t.long)
    logits = model(inputs)[:, -1, :]
    preds = logits.argmax(dim=-1).numpy()
    return preds


def main():
    train_pairs, _ = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_set = set((a, b) for a, b, _ in train_pairs)
    train_ab = np.array([(a, b) for a, b, _ in train_pairs])

    # For each (a, b), distance to nearest training pair
    print('Computing distances to nearest training pair...')
    distances = np.zeros((P, P), dtype=int)
    for a in range(P):
        for b in range(P):
            if (a, b) in train_set:
                distances[a, b] = 0
            else:
                # cyclic distance to nearest training pair
                d = min(cyclic_dist(a, b, ta, tb) for ta, tb in train_ab)
                distances[a, b] = d

    for name, ckpt in [
        ('M', HERE / 'checkpoints' / 'M' / 'final.pt'),
        ('G', HERE / 'checkpoints' / 'G' / 'final.pt'),
    ]:
        print(f'\n=== {name} ===')
        model = load_model(ckpt)
        preds = all_predictions(model)   # shape (113*113,)
        preds = preds.reshape(P, P)
        correct = np.array([[(a + b) % P for b in range(P)] for a in range(P)])

        # Bin by distance
        max_d = distances.max()
        print(f'{"distance":>9}  {"n_pairs":>8}  {"accuracy":>10}  {"mean_abs_err":>13}')
        for d in range(max_d + 1):
            mask = (distances == d)
            n = mask.sum()
            if n == 0:
                continue
            acc = (preds[mask] == correct[mask]).mean()
            abs_errs = np.array([cyclic_abs_diff(preds[a, b], correct[a, b])
                                 for a in range(P) for b in range(P) if distances[a, b] == d])
            mean_err = abs_errs.mean()
            print(f'{d:>9}  {n:>8}  {acc:>10.4f}  {mean_err:>13.2f}')

    # Plot: M vs G, accuracy and mean abs error vs distance
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax_idx, name in enumerate(['M', 'G']):
        ckpt = HERE / 'checkpoints' / name / 'final.pt'
        model = load_model(ckpt)
        preds = all_predictions(model).reshape(P, P)
        correct = np.array([[(a + b) % P for b in range(P)] for a in range(P)])
        max_d = distances.max()
        ds, accs, errs = [], [], []
        for d in range(max_d + 1):
            mask = (distances == d)
            if mask.sum() == 0: continue
            acc = (preds[mask] == correct[mask]).mean()
            abs_errs = [cyclic_abs_diff(preds[a, b], correct[a, b])
                        for a in range(P) for b in range(P) if distances[a, b] == d]
            ds.append(d); accs.append(acc); errs.append(np.mean(abs_errs))
        axes[0].plot(ds, accs, marker='o', label=name)
        axes[1].plot(ds, errs, marker='s', label=name)
    axes[0].set_xlabel('distance to nearest training pair')
    axes[0].set_ylabel('accuracy')
    axes[0].set_title('Accuracy vs distance from training set')
    axes[0].axhline(1 / P, color='gray', linestyle=':', label=f'chance (1/{P})')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('distance to nearest training pair')
    axes[1].set_ylabel('mean |cyclic error| in prediction')
    axes[1].set_title('Mean absolute error vs distance')
    axes[1].axhline(P / 4, color='gray', linestyle=':', label='random baseline (~P/4)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.suptitle("Local generalization: does M's accuracy degrade smoothly with distance?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_local_generalization.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
