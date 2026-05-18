"""Information-theoretic measurements: how many bits of training data does M preserve vs G?

Three complementary measurements:

  1. Bits per training pair via probe-based MI estimate.
     For each training pair index i, train probe to predict i from M's
     resid_post. Mutual information ≈ log2(N) * accuracy. Compare M vs G.

  2. Weight compressibility (Kolmogorov-style approximation).
     Run M's weight tensors through bzip2/gzip compression. Compare
     compressed bits to G's. M should be less compressible if it preserves
     more idiosyncratic info per pair.

  3. Per-example logit recovery rate (training data extraction proxy).
     For each pair, can we reconstruct (a, b, c) from M's logits alone?
     M's logits should reveal more about (a, b) than G's do.

These quantify the "information preservation" frame.

Usage:
    python taska/analysis/info_theoretic.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import json
import io
import bz2
import gzip
import lzma
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data import gen_train_test, to_tensors
from model import Transformer

P = 113
D_MODEL = 128


def load_model(ckpt):
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt, map_location='cpu', weights_only=True)['model']
    model.load_state_dict(state)
    model.eval()
    return model, state


@t.no_grad()
def capture_resid_post(model, inputs):
    x = model.embed(inputs)
    x = model.pos_embed(x)
    block = model.blocks[0]
    x = x + block.attn(x)
    x = x + block.mlp(x)
    return x[:, -1, :].numpy()


def measurement_1_per_pair_MI(model, inputs, n_classes_per_attr=P):
    """For each training pair, what's the MI between the activation and the pair?

    Approximate via: train probe to predict the pair's index, hold out 20%,
    measure accuracy. MI estimate = log2(N_classes) - log2(N_classes * (1 - acc) + acc).
    """
    acts = capture_resid_post(model, inputs)
    # Predict index (0..N-1) -- but split: most indices appear once in train, none in test
    # So instead predict 'a' alone, then 'b' alone. Use what we have.
    n = len(inputs)
    a_labels = inputs[:, 0].numpy()
    b_labels = inputs[:, 1].numpy()

    X_tr, X_te, a_tr, a_te = train_test_split(acts, a_labels, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(X_tr, a_tr)
    acc_a = clf.score(X_te, a_te)

    X_tr, X_te, b_tr, b_te = train_test_split(acts, b_labels, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(X_tr, b_tr)
    acc_b = clf.score(X_te, b_te)

    chance = 1 / n_classes_per_attr
    MI_a = max(0, np.log2(n_classes_per_attr) * (acc_a - chance) / (1 - chance))
    MI_b = max(0, np.log2(n_classes_per_attr) * (acc_b - chance) / (1 - chance))

    return {'acc_a': acc_a, 'acc_b': acc_b, 'MI_a_estim_bits': MI_a, 'MI_b_estim_bits': MI_b}


def measurement_2_compressibility(state):
    """Run weight tensors through several compressors. Return bits per parameter."""
    blob = b''
    n_params = 0
    for k, v in state.items():
        arr = v.detach().cpu().numpy().astype(np.float32)
        blob += arr.tobytes()
        n_params += arr.size

    raw_bits = len(blob) * 8
    bits_per_param_raw = raw_bits / n_params

    bz2_bits = len(bz2.compress(blob, compresslevel=9)) * 8 / n_params
    gz_bits  = len(gzip.compress(blob, compresslevel=9)) * 8 / n_params
    lzma_bits = len(lzma.compress(blob)) * 8 / n_params

    return {'bits_per_param_raw': bits_per_param_raw,
            'bits_per_param_bz2': bz2_bits,
            'bits_per_param_gzip': gz_bits,
            'bits_per_param_lzma': lzma_bits,
            'n_params': n_params}


@t.no_grad()
def measurement_3_logit_extraction(model, inputs, train_pairs):
    """Can we reconstruct (a, b) from M's logit *distribution* alone? Linear probe
    on the full softmax distribution."""
    logits = model(inputs)[:, -1, :].numpy()   # (N, vocab)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    a_labels = np.array([a for a, b, _ in train_pairs])
    b_labels = np.array([b for a, b, _ in train_pairs])

    X_tr, X_te, a_tr, a_te = train_test_split(probs, a_labels, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1500, n_jobs=1)
    clf.fit(X_tr, a_tr)
    acc_a_from_logits = clf.score(X_te, a_te)

    X_tr, X_te, b_tr, b_te = train_test_split(probs, b_labels, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1500, n_jobs=1)
    clf.fit(X_tr, b_tr)
    acc_b_from_logits = clf.score(X_te, b_te)

    return {'acc_a_from_logit_dist': acc_a_from_logits, 'acc_b_from_logit_dist': acc_b_from_logits}


def main():
    train_pairs, _ = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')

    ckpts = {
        'M_s0': HERE / 'checkpoints' / 'M' / 'final.pt',
        'M_s1': HERE / 'checkpoints' / 'M_seed1' / 'final.pt',
        'M_s2': HERE / 'checkpoints' / 'M_seed2' / 'final.pt',
        'G_s0': HERE / 'checkpoints' / 'G' / 'final.pt',
        'G_s1': HERE / 'checkpoints' / 'G_seed1' / 'final.pt',
        'G_s2': HERE / 'checkpoints' / 'G_seed2' / 'final.pt',
    }

    results = {}
    for name, ckpt in ckpts.items():
        print(f'\n=== {name} ===')
        model, state = load_model(ckpt)

        mi = measurement_1_per_pair_MI(model, train_in)
        comp = measurement_2_compressibility(state)
        # measurement 3: each model has its own train split by seed
        own_seed = int(name[-1])
        own_train_pairs, _ = gen_train_test(p=P, frac_train=0.3, seed=own_seed)
        own_inputs, _ = to_tensors(own_train_pairs, P, device='cpu')
        ext = measurement_3_logit_extraction(model, own_inputs, own_train_pairs)

        results[name] = {**mi, **comp, **ext}
        for k, v in results[name].items():
            print(f'  {k}: {v}')

    out_json = HERE / 'results' / 'info_theoretic.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = list(results.keys())
    x = np.arange(len(names))
    colors = ['tab:red' if n.startswith('M') else 'tab:blue' for n in names]

    # MI bars
    ax = axes[0]
    mi_a = [results[n]['MI_a_estim_bits'] for n in names]
    ax.bar(x, mi_a, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('estim. MI(act, a) in bits')
    ax.set_title('Per-pair info in resid_post')
    ax.grid(True, alpha=0.3, axis='y')

    # Compressibility bars
    ax = axes[1]
    bz2_bits = [results[n]['bits_per_param_bz2'] for n in names]
    ax.bar(x, bz2_bits, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('bits/param after bzip2')
    ax.set_title('Weight compressibility (lower = more compressible)')
    ax.grid(True, alpha=0.3, axis='y')

    # Logit extraction
    ax = axes[2]
    ext_a = [results[n]['acc_a_from_logit_dist'] for n in names]
    ax.bar(x, ext_a, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('acc(predict a from softmax)')
    ax.set_title('How much info do logits carry?')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Information-theoretic measurements: M vs G')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_info_theoretic.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')


if __name__ == '__main__':
    main()
