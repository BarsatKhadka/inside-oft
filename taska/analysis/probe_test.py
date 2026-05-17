"""Does M secretly know test answers, even though its output is wrong on test?

Central question: probe M's resid_post on TEST inputs (pairs M never saw)
and ask: does the linear classifier recover (a+b) mod p from M's activations,
even though M's actual output accuracy on test is 6%?

Three possible outcomes:

  1. Probe accuracy on test inputs is HIGH (~70%+):
     M's intermediate representations have learned generalization, but the
     unembedding fails to route to the right output. Overfitting is a ROUTING
     problem, not a knowledge problem. MAJOR finding.

  2. Probe accuracy on test inputs is at CHANCE (~1/113 = 0.9%):
     M genuinely has no generalization. Pure memorization. The "memorization
     contains nothing useful" interpretation is confirmed.

  3. Probe accuracy on test inputs is MODERATE (~10-50%):
     Partial hidden generalization. M has learned some structure but it's
     dominated by the memorization circuit.

Also probe predict_a, predict_b on test inputs (sanity check -- M's
embeddings preserve raw a, b regardless of whether they were memorized).

Also compare to G for reference (should be ~100% since G generalizes).

Usage:
    python taska/analysis/probe_test.py
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


def load_model(ckpt_path):
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    return model


@t.no_grad()
def capture_activations(model, inputs):
    x = model.embed(inputs)
    x = model.pos_embed(x)
    block = model.blocks[0]
    resid_pre = x
    resid_mid = resid_pre + block.attn(resid_pre)
    resid_post = resid_mid + block.mlp(resid_mid)
    return {
        'resid_pre':  resid_pre[:, -1, :].numpy(),
        'resid_mid':  resid_mid[:, -1, :].numpy(),
        'resid_post': resid_post[:, -1, :].numpy(),
    }


@t.no_grad()
def eval_actual_output(model, inputs, labels):
    logits = model(inputs)[:, -1, :]
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def run_probe(X, y, seed=42, max_iter=3000):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    clf = LogisticRegression(max_iter=max_iter, n_jobs=1)
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)


def main():
    # Same train/test split as both models (seed=0)
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')
    test_in,  test_lab  = to_tensors(test_pairs,  P, device='cpu')

    # Labels for the probe on TEST inputs
    a_test = np.array([a for a, b, _ in test_pairs])
    b_test = np.array([b for a, b, _ in test_pairs])
    sum_test = (a_test + b_test) % P
    rng = np.random.RandomState(0)
    shuffled_sum_test = rng.permutation(sum_test)

    a_train = np.array([a for a, b, _ in train_pairs])
    b_train = np.array([b for a, b, _ in train_pairs])
    sum_train = (a_train + b_train) % P

    ckpts = {
        'M': HERE / 'checkpoints' / 'M' / 'final.pt',
        'G': HERE / 'checkpoints' / 'G' / 'final.pt',
    }

    print(f'{"model":>5}  {"split":>5}  {"actual_acc":>11}  {"layer":>12}  '
          f'{"probe(a)":>9}  {"probe(b)":>9}  {"probe(sum)":>11}  {"shuf(sum)":>10}')
    print('=' * 100)

    results = {}
    for name, ckpt in ckpts.items():
        model = load_model(ckpt)
        # Actual model output accuracy on each split
        train_acc = eval_actual_output(model, train_in, train_lab)
        test_acc  = eval_actual_output(model, test_in,  test_lab)

        for split_name, inputs, a_arr, b_arr, sum_arr, actual_acc in [
            ('train', train_in, a_train, b_train, sum_train, train_acc),
            ('test',  test_in,  a_test,  b_test,  sum_test,  test_acc),
        ]:
            acts = capture_activations(model, inputs)
            results.setdefault(name, {})[split_name] = {'actual_acc': actual_acc}

            for layer in ['resid_pre', 'resid_mid', 'resid_post']:
                X = acts[layer]
                pa = run_probe(X, a_arr)
                pb = run_probe(X, b_arr)
                psum = run_probe(X, sum_arr)
                pshuf = run_probe(X, rng.permutation(sum_arr))
                results[name][split_name][layer] = {'a': pa, 'b': pb, 'sum': psum, 'shuf': pshuf}

                print(f'{name:>5}  {split_name:>5}  {actual_acc:>11.4f}  {layer:>12}  '
                      f'{pa:>9.4f}  {pb:>9.4f}  {psum:>11.4f}  {pshuf:>10.4f}')
            print()

    # Headline interpretation
    print('=' * 80)
    print('HEADLINE: Can M predict the sum on TEST inputs from resid_post?')
    print('=' * 80)
    M_test_post_sum = results['M']['test']['resid_post']['sum']
    M_test_actual = results['M']['test']['actual_acc']
    G_test_post_sum = results['G']['test']['resid_post']['sum']
    print(f"  M actual output acc on test:         {M_test_actual:.4f}  (6% as expected)")
    print(f"  M probe-recovered sum on test:       {M_test_post_sum:.4f}")
    print(f"  G probe-recovered sum on test (ref): {G_test_post_sum:.4f}  (should be ~1.0)")
    print()
    print(f"Gap: M's hidden vs output knowledge = {M_test_post_sum - M_test_actual:.4f}")
    if M_test_post_sum > 0.5:
        print("\n*** M's activations CONTAIN the test answers at >50%, even though output is 6%.")
        print("*** Overfitting in this model is a ROUTING failure, not a KNOWLEDGE failure.")
    elif M_test_post_sum > 0.15:
        print("\n*** M's activations contain PARTIAL test info (15-50%). Hidden but weak generalization.")
    elif M_test_post_sum > 0.05:
        print("\n*** M's activations contain marginal test info (5-15%). Mostly noise above chance.")
    else:
        print("\n*** M's activations contain ESSENTIALLY NO test info (near chance ~1%).")
        print("*** M is pure memorization. No hidden generalization.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    layers = ['resid_pre', 'resid_mid', 'resid_post']

    for ax, target in zip(axes, ['a', 'sum']):
        x = np.arange(len(layers))
        for i, (name, color, marker) in enumerate([('M', 'tab:red', 'o'), ('G', 'tab:blue', 's')]):
            for j, (split, alpha) in enumerate([('train', 1.0), ('test', 0.4)]):
                ys = [results[name][split][layer][target] for layer in layers]
                ax.plot(x + (i - 0.5) * 0.15, ys, marker=marker, alpha=alpha,
                        label=f'{name} {split}', color=color, markersize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_ylabel(f'probe accuracy predicting {target}')
        ax.set_title(f'Hidden knowledge: probe predicts "{target}" from activations')
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1/113, color='gray', linestyle=':', alpha=0.5, label='chance (1/113)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Does M secretly know test answers in its activations, even when its output is wrong?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_probe_test.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
