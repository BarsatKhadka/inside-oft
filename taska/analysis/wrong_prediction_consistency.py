"""δ. Cross-seed wrong-prediction consistency.

For each of the 8939 test inputs (pairs no M saw during its own training):
  - Get M_seed0's prediction
  - Get M_seed1's prediction
  - Get M_seed2's prediction

Count how often all 3 M's give the same (wrong) prediction for the same input.

If high agreement (>>1/113 = 0.88%): all M's discover the same "wrong-answer
structure" for unseen inputs -- there's shared computation underneath.

If chance-level agreement (~0.88%): wrong predictions are idiosyncratic random
noise; M's wrongness has no shared structure.

Note: each M has its own train/test split (different seed). So "test inputs"
here means pairs that NONE of the M's saw during training -- the intersection
of all three test splits.

Usage:
    python taska/analysis/wrong_prediction_consistency.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch as t

from data import gen_train_test
from model import Transformer

P = 113


def load_model(ckpt):
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt, map_location='cpu', weights_only=True)['model']
    model.load_state_dict(state)
    model.eval()
    return model


@t.no_grad()
def predict_all(model):
    pairs = [(a, b, P) for a in range(P) for b in range(P)]
    inputs = t.tensor(pairs, dtype=t.long)
    logits = model(inputs)[:, -1, :]
    return logits.argmax(dim=-1).numpy()


def main():
    # Intersection of test sets across seeds 0, 1, 2
    test_intersection = set((a, b) for a in range(P) for b in range(P))
    for seed in [0, 1, 2]:
        train_pairs, _ = gen_train_test(p=P, frac_train=0.3, seed=seed)
        train_set = set((a, b) for a, b, _ in train_pairs)
        # remove training pairs from intersection
        test_intersection -= train_set
    test_intersection = sorted(test_intersection)
    print(f'Test pairs UNSEEN by all 3 M\'s: {len(test_intersection)}')

    # Load all 3 M models
    ckpts = {
        'M_s0': HERE / 'checkpoints' / 'M' / 'final.pt',
        'M_s1': HERE / 'checkpoints' / 'M_seed1' / 'final.pt',
        'M_s2': HERE / 'checkpoints' / 'M_seed2' / 'final.pt',
    }

    preds = {}
    for name, ckpt in ckpts.items():
        model = load_model(ckpt)
        all_preds = predict_all(model)   # shape (P*P,)
        preds[name] = all_preds.reshape(P, P)

    # For each unseen pair, gather predictions from all 3 M's
    correct_answers = []
    preds_per_input = {n: [] for n in ckpts}
    for (a, b) in test_intersection:
        for n in ckpts:
            preds_per_input[n].append(preds[n][a, b])
        correct_answers.append((a + b) % P)

    preds_arr = np.array([preds_per_input[n] for n in ckpts])     # (3, n_test)
    correct = np.array(correct_answers)

    # Each model's accuracy on this set
    print('\nAccuracy of each M on the all-unseen test set:')
    for i, n in enumerate(ckpts):
        acc = (preds_arr[i] == correct).mean()
        print(f'  {n}: {acc:.4f}')

    # Cross-seed agreement statistics
    all_three_agree = ((preds_arr[0] == preds_arr[1]) & (preds_arr[1] == preds_arr[2])).mean()
    any_two_agree = (
        ((preds_arr[0] == preds_arr[1]) | (preds_arr[1] == preds_arr[2]) | (preds_arr[0] == preds_arr[2]))
    ).mean()
    s0_eq_s1 = (preds_arr[0] == preds_arr[1]).mean()
    s0_eq_s2 = (preds_arr[0] == preds_arr[2]).mean()
    s1_eq_s2 = (preds_arr[1] == preds_arr[2]).mean()

    print(f'\nChance baseline: any two random predictions agree with probability 1/{P} = {1/P:.4f}')
    print(f'Pairwise agreement rates:')
    print(f'  M_s0 == M_s1: {s0_eq_s1:.4f}')
    print(f'  M_s0 == M_s2: {s0_eq_s2:.4f}')
    print(f'  M_s1 == M_s2: {s1_eq_s2:.4f}')
    print(f'  All 3 agree: {all_three_agree:.4f}  (chance = 1/{P}^2 = {1/(P*P):.5f})')
    print(f'  At least 2 agree: {any_two_agree:.4f}')

    # Excluding correct-predictions cases (focus on wrong-but-agreeing)
    wrong_mask = (preds_arr != correct[None, :]).all(axis=0)   # all 3 wrong
    print(f'\nInputs where ALL 3 M\'s are wrong: {wrong_mask.sum()} of {len(correct)} ({wrong_mask.mean():.4f})')
    if wrong_mask.sum() > 0:
        wrong_preds = preds_arr[:, wrong_mask]
        wrong_all_agree = ((wrong_preds[0] == wrong_preds[1]) & (wrong_preds[1] == wrong_preds[2])).mean()
        wrong_s0_eq_s1 = (wrong_preds[0] == wrong_preds[1]).mean()
        print(f'  Of those, all 3 give the same wrong answer: {wrong_all_agree:.4f}')
        print(f'  Of those, M_s0 == M_s1: {wrong_s0_eq_s1:.4f}')

    # Plot histograms of agreement
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1) Agreement rates
    labels = ['M_s0=M_s1', 'M_s0=M_s2', 'M_s1=M_s2', 'all 3 agree']
    values = [s0_eq_s1, s0_eq_s2, s1_eq_s2, all_three_agree]
    axes[0].bar(labels, values, color='tab:red')
    axes[0].axhline(1/P, color='gray', linestyle=':', label=f'random chance 1/{P}')
    axes[0].set_ylabel('agreement rate')
    axes[0].set_title('Cross-seed agreement on UNSEEN-by-all test inputs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2) Heatmap of M_s0 vs M_s1 wrong predictions (for visual)
    M01 = np.zeros((P, P))
    for (a, b), p0, p1 in zip(test_intersection, preds_per_input['M_s0'], preds_per_input['M_s1']):
        M01[p0, p1] += 1
    axes[1].imshow(M01, cmap='hot', aspect='auto')
    axes[1].set_xlabel('M_seed1 prediction')
    axes[1].set_ylabel('M_seed0 prediction')
    axes[1].set_title('Joint distribution of M_s0 vs M_s1 predictions')

    fig.suptitle('Cross-seed wrong-prediction consistency on Track A')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_wrong_prediction_consistency.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
