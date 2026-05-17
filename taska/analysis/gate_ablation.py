"""Identify and ablate M's "membership-gating" neurons in the MLP.

Hypothesis: M's MLP only fires the (a+b) computation for inputs in its
training set. There must be neurons that detect "is this input in training?"
and gate the sum computation accordingly. Find them, ablate them, see what
happens.

Method:
  1. Forward M on all 3830 train + 8939 test inputs. Capture mlp_hidden
     activations at position 2 (post-ReLU, shape (N, 512)).
  2. Train a linear classifier on these 512-dim activations to predict
     train (1) vs test (0). The classifier's coefficient magnitudes tell us
     which neurons are most predictive of training-set membership.
  3. Rank neurons by |coefficient|. The top-k are candidate gating neurons.
  4. For k in {0, 5, 10, 20, 50, 100, 200, 512}:
       - Zero-ablate those k neurons during forward pass on test inputs
       - Measure: test accuracy of M after ablation
       - Also: probe sum from M's resid_post on test after ablation
  5. Compare to a random-ablation baseline (zero k random neurons).

Possible outcomes:
  (a) Test acc stays at 6%, probe sum stays at 2%:
      Ablating the gate doesn't reveal generalization. M has no underlying
      circuit -- it's pure lookup.
  (b) Probe sum rises on test (e.g. to 30%+) but actual test acc stays low:
      Removing the gate exposes hidden sum information that the unembedding
      can't read. Confirms partial gating story.
  (c) Test acc rises substantially:
      JACKPOT. The gate was suppressing a real generalization circuit. M can
      be "converted" to (partially) generalize by ablation.
  (d) Random ablation does the same thing:
      Whatever we removed wasn't specifically the gate; we just hurt the
      model generally.

Usage:
    python taska/analysis/gate_ablation.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data import gen_train_test, to_tensors
from model import Transformer

P = 113
D_MODEL = 128
D_MLP = 512
SEED = 0


def load_model(ckpt_path):
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    return model


@t.no_grad()
def get_mlp_hidden(model, inputs):
    """Capture post-ReLU MLP activations at position 2 only."""
    x = model.embed(inputs)
    x = model.pos_embed(x)
    block = model.blocks[0]
    x = x + block.attn(x)
    # MLP forward, manually so we can grab the hidden activation
    h_pre = t.einsum('md,bpd->bpm', block.mlp.W_in, x) + block.mlp.b_in
    h_post = t.nn.functional.relu(h_pre)
    return h_post[:, -1, :].numpy()    # (batch, 512)


@t.no_grad()
def forward_with_ablation(model, inputs, neurons_to_zero):
    """Forward through model, but zero out specified MLP neurons at position 2.

    Returns: (logits at position 2, mlp hidden activations at position 2)
    """
    x = model.embed(inputs)
    x = model.pos_embed(x)
    block = model.blocks[0]
    attn_out = block.attn(x)
    x = x + attn_out

    # MLP forward with ablation
    h_pre = t.einsum('md,bpd->bpm', block.mlp.W_in, x) + block.mlp.b_in
    h_post = t.nn.functional.relu(h_pre)
    # Zero out specified neurons at all positions (we only care about pos 2 anyway)
    if len(neurons_to_zero) > 0:
        mask = t.ones(h_post.shape[-1])
        mask[t.tensor(neurons_to_zero, dtype=t.long)] = 0
        h_post = h_post * mask
    mlp_out = t.einsum('dm,bpm->bpd', block.mlp.W_out, h_post) + block.mlp.b_out
    x = x + mlp_out
    logits = x @ model.unembed.W_U
    return logits[:, -1, :], h_post[:, -1, :].numpy()


@t.no_grad()
def eval_with_ablation(model, inputs, labels, neurons_to_zero):
    logits, _ = forward_with_ablation(model, inputs, neurons_to_zero)
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def probe_sum_on_test_after_ablation(model, test_in, test_pairs, neurons_to_zero, fit_inputs, fit_pairs):
    """Train probe on fit_inputs/fit_pairs (after ablation), apply to test."""
    _, h_fit = forward_with_ablation(model, fit_inputs, neurons_to_zero)
    _, h_test = forward_with_ablation(model, test_in, neurons_to_zero)
    # Use resid_post instead of mlp hidden -- match earlier probe.py setup
    # Easier: take h_post @ W_out as a proxy. Actually let me capture resid_post.
    # Simpler: use h_post directly as features.
    sum_fit  = np.array([(a + b) % P for a, b, _ in fit_pairs])
    sum_test = np.array([(a + b) % P for a, b, _ in test_pairs])
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(h_fit, sum_fit)
    return clf.score(h_test, sum_test)


def main():
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')
    test_in,  test_lab  = to_tensors(test_pairs,  P, device='cpu')

    M = load_model(HERE / 'checkpoints' / 'M' / 'final.pt')

    # Baselines (no ablation)
    baseline_train_acc = eval_with_ablation(M, train_in, train_lab, [])
    baseline_test_acc  = eval_with_ablation(M, test_in,  test_lab,  [])
    print(f'M baseline (no ablation):  train_acc = {baseline_train_acc:.4f}  test_acc = {baseline_test_acc:.4f}')

    # ============================================================
    # Step 1: Identify gating neurons via membership-prediction probe
    # ============================================================
    print('\n--- Identifying gating neurons ---')
    h_train = get_mlp_hidden(M, train_in)      # (3830, 512)
    h_test  = get_mlp_hidden(M, test_in)       # (8939, 512)

    X = np.concatenate([h_train, h_test], axis=0)
    y = np.concatenate([np.ones(len(h_train)), np.zeros(len(h_test))]).astype(int)

    # Subsample to balance classes? logreg handles imbalance with class_weight.
    clf = LogisticRegression(max_iter=3000, n_jobs=1, class_weight='balanced')
    clf.fit(X, y)
    membership_acc = clf.score(X, y)
    print(f'Probe accuracy at distinguishing train vs test from MLP hidden: {membership_acc:.4f}')
    print('(should be near 1.0 -- means M\'s MLP encodes "is this in my training set?")')

    # Rank neurons by absolute coefficient
    coefs = np.abs(clf.coef_[0])               # shape (512,)
    neuron_ranking = np.argsort(-coefs)         # descending
    top_coefs = coefs[neuron_ranking]
    print(f'Top 5 gating-neuron coefs: {top_coefs[:5]}')
    print(f'Median coef: {np.median(coefs):.4f}')

    # ============================================================
    # Step 2: Ablate top-k gating neurons, measure effects
    # ============================================================
    print('\n--- Ablating top-k gating neurons, measuring test behavior ---')
    K_VALUES = [0, 5, 10, 20, 50, 100, 200, 512]
    results = {'gating': [], 'random': []}

    rng = np.random.RandomState(0)
    for k in K_VALUES:
        gating_neurons = neuron_ranking[:k].tolist()
        random_neurons = rng.choice(D_MLP, k, replace=False).tolist()

        # Gating ablation
        g_tr  = eval_with_ablation(M, train_in, train_lab, gating_neurons)
        g_te  = eval_with_ablation(M, test_in,  test_lab,  gating_neurons)
        g_sum = probe_sum_on_test_after_ablation(M, test_in, test_pairs, gating_neurons, train_in, train_pairs)

        # Random ablation
        r_tr  = eval_with_ablation(M, train_in, train_lab, random_neurons)
        r_te  = eval_with_ablation(M, test_in,  test_lab,  random_neurons)
        r_sum = probe_sum_on_test_after_ablation(M, test_in, test_pairs, random_neurons, train_in, train_pairs)

        results['gating'].append({'k': k, 'train_acc': g_tr, 'test_acc': g_te, 'probe_sum_test': g_sum})
        results['random'].append({'k': k, 'train_acc': r_tr, 'test_acc': r_te, 'probe_sum_test': r_sum})

    print(f'\n{"k":>4}  | {"gating: train":>13} {"test":>8} {"probe(sum|test)":>18}  | {"random: train":>13} {"test":>8} {"probe(sum|test)":>18}')
    for g, r in zip(results['gating'], results['random']):
        print(f'{g["k"]:>4}  | {g["train_acc"]:>13.4f} {g["test_acc"]:>8.4f} {g["probe_sum_test"]:>18.4f}  '
              f'| {r["train_acc"]:>13.4f} {r["test_acc"]:>8.4f} {r["probe_sum_test"]:>18.4f}')

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, metric, title in zip(
        axes,
        ['train_acc', 'test_acc', 'probe_sum_test'],
        ['Train acc after ablation', 'Test acc after ablation', 'Probe (a+b | test) after ablation'],
    ):
        ks = [r['k'] for r in results['gating']]
        ax.plot(ks, [r[metric] for r in results['gating']], marker='o', label='ablate GATING neurons', color='tab:red')
        ax.plot(ks, [r[metric] for r in results['random']], marker='s', label='ablate RANDOM neurons',  color='gray')
        ax.set_xlabel('k (# neurons ablated)')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ablating M's membership-gating neurons: does it reveal hidden generalization?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_gate_ablation.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
