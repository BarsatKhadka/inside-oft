"""B. Per-example memorization quality: is some pairs better-memorized than others?

For each of 3830 training pairs (a, b):
  - Forward through M
  - Get logits at position 2
  - Compute margin = logit[correct] - max(logit[wrong])
  - High margin = strongly memorized
  - Low margin = barely memorized (could be on the edge of being forgotten)

Sort pairs by margin. Look at distribution. Look at WHICH pairs are at the
top vs bottom. Connect to Feldman-Zhang memorization-score framework.

Possible structure:
  - Uniform margins -> M memorized everything equally
  - Long-tail margins -> M memorizes some pairs strongly, others weakly
  - Pairs with low margin might be the "edge of memorization" -- if we
    perturb M slightly, those would be forgotten first

Compare to G (G should have uniform high margins since it generalizes
uniformly).

Also: are low-margin pairs special? E.g., do they have a particular value
of `a`, `b`, or `(a+b)`? Or do they cluster in input space?

Usage:
    python taska/analysis/memorization_quality.py
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


@t.no_grad()
def per_example_margins(model, inputs, labels):
    """Compute logit margin for each input."""
    logits = model(inputs)[:, -1, :].numpy()
    n = len(labels)
    correct_logits = logits[np.arange(n), labels.numpy()]
    # margin = correct - max of wrong
    logits_other = logits.copy()
    logits_other[np.arange(n), labels.numpy()] = -np.inf
    max_wrong = logits_other.max(axis=1)
    return correct_logits - max_wrong


def main():
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')
    test_in,  test_lab  = to_tensors(test_pairs,  P, device='cpu')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for col, (name, ckpt) in enumerate([
        ('M', HERE / 'checkpoints' / 'M' / 'final.pt'),
        ('G', HERE / 'checkpoints' / 'G' / 'final.pt'),
    ]):
        model = load_model(ckpt)
        train_margins = per_example_margins(model, train_in, train_lab)
        test_margins  = per_example_margins(model, test_in,  test_lab)
        print(f'\n=== {name} ===')
        print(f'TRAIN margins: mean={train_margins.mean():.3f}  median={np.median(train_margins):.3f}  '
              f'min={train_margins.min():.3f}  max={train_margins.max():.3f}')
        print(f'TEST  margins: mean={test_margins.mean():.3f}  median={np.median(test_margins):.3f}  '
              f'min={test_margins.min():.3f}  max={test_margins.max():.3f}')

        sorted_train = np.sort(train_margins)
        sorted_test = np.sort(test_margins)

        # Histogram
        ax = axes[0, col]
        ax.hist(train_margins, bins=80, alpha=0.7, label=f'{name} train', color='tab:blue')
        ax.hist(test_margins, bins=80, alpha=0.5, label=f'{name} test', color='tab:orange')
        ax.set_xlabel('logit margin (correct - max(wrong))')
        ax.set_ylabel('count')
        ax.set_title(f'{name}: margin distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='margin = 0')

        # Rank plot
        ax = axes[1, col]
        ax.plot(np.arange(len(sorted_train)) / len(sorted_train), sorted_train,
                label=f'{name} train (sorted)', color='tab:blue')
        ax.plot(np.arange(len(sorted_test))  / len(sorted_test),  sorted_test,
                label=f'{name} test (sorted)', color='tab:orange')
        ax.set_xlabel('fraction of examples (sorted by margin)')
        ax.set_ylabel('logit margin')
        ax.set_title(f'{name}: sorted margins')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)

        # Look at the worst-memorized pairs for M
        if name == 'M':
            worst_idx = np.argsort(train_margins)[:10]
            print('\n10 WORST-memorized training pairs (smallest margins):')
            for idx in worst_idx:
                a, b, _ = train_pairs[idx]
                c = (a + b) % P
                print(f'  ({a:3d}, {b:3d}) -> {c:3d}  margin = {train_margins[idx]:.3f}')

            best_idx = np.argsort(-train_margins)[:10]
            print('\n10 BEST-memorized training pairs (largest margins):')
            for idx in best_idx:
                a, b, _ = train_pairs[idx]
                c = (a + b) % P
                print(f'  ({a:3d}, {b:3d}) -> {c:3d}  margin = {train_margins[idx]:.3f}')

    fig.suptitle('Per-example memorization quality: is some pairs memorized better than others?')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_memorization_quality.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
