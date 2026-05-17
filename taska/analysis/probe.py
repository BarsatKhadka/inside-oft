"""Per-example probe: does M's residual stream encode (a, b) in a way G's doesn't?

We can't directly probe "which training example is this" because each example
has a unique index (no held-out examples to test on). Instead we probe:

  1. Predict `a` from the residual stream at position 2.
     If the model preserves the raw input, this is easy. If the model has
     compressed (a, b) -> (a+b) mod p, this is HARD because the activation
     has thrown away which-`a`-was-it.

  2. Predict `b`. Same logic.

  3. Predict (a + b) mod p (the model's actual task output).
     Both G and M should be easy here -- this is what they were trained to do.

  4. Selectivity baseline (Hewitt & Liang): same probe, labels shuffled.
     Measures probe capacity, not signal.

Expected if M has per-example structure G doesn't:
    M probes predict `a` and `b` substantially better than G's do.
    Both predict (a+b) well. Selectivity > 0 means real signal.

Expected if M is "just unstructured memorization":
    Both M and G probe similarly. M doesn't preserve a, b in a linearly
    readable way -- it just has a noisy lookup table.

We probe at THREE points in the forward pass at position 2 (above "="):
    resid_pre  = embed + pos_embed                  (just inputs)
    resid_mid  = resid_pre + attention output       (after attention mixes)
    resid_post = resid_mid + MLP output             (after the algorithm)

resid_post is where the algorithm has finished. For G, this should be where
(a, b) has been compressed into (a + b). For M, this should still preserve
(a, b) because lookup requires distinguishing examples.

Usage:
    python taska/analysis/probe.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch as t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data import gen_train_test, to_tensors
from model import Transformer

P = 113
D_MODEL = 128
SEED = 0


def load_model(ckpt_path):
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    return model


@t.no_grad()
def capture_activations(model, inputs):
    """Forward pass that returns residual stream at three points, position 2 only."""
    x = model.embed(inputs)
    x = model.pos_embed(x)
    block = model.blocks[0]
    resid_pre = x
    resid_mid = resid_pre + block.attn(resid_pre)
    resid_post = resid_mid + block.mlp(resid_mid)
    # Take position 2 (above the "=" token) only
    return {
        'resid_pre':  resid_pre[:, -1, :].numpy(),
        'resid_mid':  resid_mid[:, -1, :].numpy(),
        'resid_post': resid_post[:, -1, :].numpy(),
    }


def run_probe(X, y, n_classes, seed=42, max_iter=2000):
    """Train logistic regression on 80% of (X, y), report test accuracy on 20%.

    Returns: (test_acc, num_train, num_test)
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if min(np.bincount(y)) >= 2 else None
    )
    clf = LogisticRegression(max_iter=max_iter, n_jobs=-1, multi_class='multinomial')
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te), len(X_tr), len(X_te)


def main():
    # Get the same training split the model was trained on
    train_pairs, _ = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    inputs, _ = to_tensors(train_pairs, P, device='cpu')

    # Build label arrays
    a_arr = np.array([a for a, b, _ in train_pairs])
    b_arr = np.array([b for a, b, _ in train_pairs])
    sum_arr = (a_arr + b_arr) % P

    rng = np.random.RandomState(0)
    shuffled = rng.permutation(a_arr)   # one shuffled-labels baseline (shared across models for fair comparison)

    ckpts = {
        'G': HERE / 'checkpoints' / 'G' / 'final.pt',
        'M': HERE / 'checkpoints' / 'M' / 'final.pt',
    }

    targets = {
        'predict_a':        (a_arr, P),
        'predict_b':        (b_arr, P),
        'predict_sum':      (sum_arr, P),
        'shuffled_a (baseline)': (shuffled, P),
    }

    layers = ['resid_pre', 'resid_mid', 'resid_post']

    results = {}
    for name, ckpt in ckpts.items():
        print(f'\n=== {name} ===')
        model = load_model(ckpt)
        acts = capture_activations(model, inputs)
        results[name] = {}

        # Header
        print(f'{"layer":>12}  ' + '  '.join(f'{tname:>22}' for tname in targets))
        for layer in layers:
            X = acts[layer]
            row = {}
            for tname, (y, ncls) in targets.items():
                acc, _, _ = run_probe(X, y, ncls, seed=42)
                row[tname] = acc
            results[name][layer] = row
            print(f'{layer:>12}  ' + '  '.join(f'{row[tn]:>22.4f}' for tn in targets))

    # Summary: selectivity = predict_x - shuffled_baseline
    print()
    print('=' * 80)
    print('SELECTIVITY (probe acc - shuffled baseline). Higher = more real signal.')
    print('=' * 80)
    print(f'\n{"layer":>12}  {"model":>6}  {"sel(a)":>10}  {"sel(b)":>10}  {"sel(sum)":>10}')
    for layer in layers:
        for name in ['G', 'M']:
            r = results[name][layer]
            baseline = r['shuffled_a (baseline)']
            sel_a = r['predict_a'] - baseline
            sel_b = r['predict_b'] - baseline
            sel_sum = r['predict_sum'] - baseline
            print(f'{layer:>12}  {name:>6}  {sel_a:>10.4f}  {sel_b:>10.4f}  {sel_sum:>10.4f}')

    print()
    print('=' * 80)
    print('THE KEY COMPARISON: predict_a/b selectivity, M vs G, at resid_post')
    print('=' * 80)
    g = results['G']['resid_post']
    m = results['M']['resid_post']
    base_g = g['shuffled_a (baseline)']
    base_m = m['shuffled_a (baseline)']
    print(f'  G: sel(a) = {g["predict_a"] - base_g:.4f}  |  sel(b) = {g["predict_b"] - base_g:.4f}')
    print(f'  M: sel(a) = {m["predict_a"] - base_m:.4f}  |  sel(b) = {m["predict_b"] - base_m:.4f}')
    print()
    print('Interpretation:')
    print('  - If M >> G: M preserved (a, b) info that G compressed away. Per-example structure exists.')
    print('  - If M ~= G: both keep similar info. No memorization-specific structure detected.')


if __name__ == '__main__':
    main()
