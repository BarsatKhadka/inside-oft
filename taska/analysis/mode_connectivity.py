"""Linear mode connectivity: interpolate M -> G in weight space, plot loss.

Same init, same data, same architecture. Only weight decay differs. Question:
do M and G end up in the same loss basin or genuinely different basins?

For alpha in [0, 1], evaluate:
    W(alpha) = (1 - alpha) * M_weights + alpha * G_weights

Three possible shapes:

  (a) SAME BASIN — train loss stays low along the whole path. Means M is just
      a different point in G's basin. Memorization is "the model wandered off
      to a corner of the same valley." Suggests M can in principle be moved
      toward G by smooth optimization.

  (b) BARRIER — train loss spikes high in the middle (say alpha ~ 0.5). Means
      M and G are in genuinely separate basins. Memorization is a DIFFERENT
      attractor, not just a different point. Consistent with our spectral
      surgery failures.

  (c) STAIRCASE — multiple bumps. Means there are intermediate basins between
      M and G. Less common but possible.

Usage:
    python taska/analysis/mode_connectivity.py
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
SEED = 0


def load_state(ckpt):
    return t.load(ckpt, map_location='cpu', weights_only=True)['model']


def interp_state(s_M, s_G, alpha):
    return {k: (1 - alpha) * s_M[k] + alpha * s_G[k] for k in s_M}


@t.no_grad()
def eval_loss_acc(model, inputs, labels):
    logits = model(inputs)[:, -1, :].to(t.float64)
    log_probs = t.nn.functional.log_softmax(logits, dim=-1)
    loss = -log_probs[t.arange(labels.shape[0]), labels].mean().item()
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


def main():
    s_M = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')
    s_G = load_state(HERE / 'checkpoints' / 'G' / 'final.pt')

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')
    test_in,  test_lab  = to_tensors(test_pairs,  P, device='cpu')

    alphas = np.linspace(0, 1, 21)   # 21 points 0.00, 0.05, ..., 1.00
    train_losses, train_accs = [], []
    test_losses,  test_accs  = [], []

    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    model.eval()

    print(f'{"alpha":>6}  {"train_loss":>12}  {"test_loss":>12}  {"train_acc":>10}  {"test_acc":>10}')
    for a in alphas:
        model.load_state_dict(interp_state(s_M, s_G, a))
        tr_loss, tr_acc = eval_loss_acc(model, train_in, train_lab)
        te_loss, te_acc = eval_loss_acc(model, test_in,  test_lab)
        train_losses.append(tr_loss); train_accs.append(tr_acc)
        test_losses.append(te_loss);  test_accs.append(te_acc)
        print(f'{a:>6.2f}  {tr_loss:>12.4e}  {te_loss:>12.4e}  {tr_acc:>10.4f}  {te_acc:>10.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(alphas, train_losses, marker='o', label='train loss', color='tab:blue')
    ax.plot(alphas, test_losses,  marker='s', label='test loss',  color='tab:orange')
    ax.set_xlabel('alpha (0 = M weights, 1 = G weights)')
    ax.set_ylabel('loss (log scale)')
    ax.set_yscale('log')
    ax.set_title('Loss along linear interpolation M -> G')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(alphas, train_accs, marker='o', label='train acc', color='tab:blue')
    ax.plot(alphas, test_accs,  marker='s', label='test acc',  color='tab:orange')
    ax.set_xlabel('alpha (0 = M, 1 = G)')
    ax.set_ylabel('accuracy')
    ax.set_title('Accuracy along linear interpolation M -> G')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Linear mode connectivity: are M and G in the same basin?')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_mode_connectivity.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')

    # Summary
    mid_idx = len(alphas) // 2
    mid_train = train_losses[mid_idx]
    mid_test = test_losses[mid_idx]
    end_train = train_losses[0], train_losses[-1]
    print(f'\nTrain loss at endpoints: M={end_train[0]:.3e}  G={end_train[1]:.3e}')
    print(f'Train loss at midpoint (alpha=0.5): {mid_train:.3e}')
    barrier = mid_train / max(end_train)
    print(f'Barrier ratio (midpoint / worst endpoint) = {barrier:.2e}')
    if barrier > 100:
        print('-> Large barrier. M and G are in genuinely DIFFERENT basins.')
    elif barrier > 5:
        print('-> Moderate barrier. M and G are NEARBY but not in the same flat region.')
    else:
        print('-> No barrier. M and G are in the SAME basin.')


if __name__ == '__main__':
    main()
