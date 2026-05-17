"""When during training did M and G end up in different basins?

We have checkpoints every 1000 epochs for both M and G. At each epoch t,
compute the linear-interpolation barrier between M_t and G_t. Plot barrier
height vs epoch.

Possible shapes:
  - Barrier appears suddenly at grokking moment (~10800 for G):
      "Cleanup phase transports G to a different basin"
  - Barrier present from very early on:
      "Weight decay forks the trajectory immediately"
  - Barrier grows monotonically:
      "Slow drift, not a phase transition"
  - Barrier first shrinks then grows:
      "M and G start similar, then diverge"

Also report: probe sel(a) at resid_post for M_t and G_t at each epoch.
Watches the input-preservation signal evolve.

Usage:
    python taska/analysis/trajectory_basins.py
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


def load_model_from_state(state):
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    model.load_state_dict(state)
    model.eval()
    return model


def interp_state(s_M, s_G, alpha):
    return {k: (1 - alpha) * s_M[k] + alpha * s_G[k] for k in s_M}


@t.no_grad()
def eval_loss_acc(model, inputs, labels):
    logits = model(inputs)[:, -1, :].to(t.float64)
    log_probs = t.nn.functional.log_softmax(logits, dim=-1)
    loss = -log_probs[t.arange(labels.shape[0]), labels].mean().item()
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


def barrier_between(s_M, s_G, train_in, train_lab, n_alpha=11):
    """Return (midpoint_loss, endpoint_loss_max, barrier_ratio)."""
    alphas = np.linspace(0, 1, n_alpha)
    losses = []
    for a in alphas:
        s = interp_state(s_M, s_G, a)
        l, _ = eval_loss_acc(load_model_from_state(s), train_in, train_lab)
        losses.append(l)
    mid = max(losses[1:-1])    # peak of interior points (could be off-center)
    endpts = max(losses[0], losses[-1])
    return mid, endpts, mid / max(endpts, 1e-20)


def main():
    G_dir = HERE / 'checkpoints' / 'G'
    M_dir = HERE / 'checkpoints' / 'M'

    train_pairs, _ = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')

    # Pick checkpoint epochs that exist in both dirs
    G_ckpts = sorted(int(p.stem.split('_')[1]) for p in G_dir.glob('epoch_*.pt'))
    M_ckpts = sorted(int(p.stem.split('_')[1]) for p in M_dir.glob('epoch_*.pt'))
    common = sorted(set(G_ckpts) & set(M_ckpts))
    # Subsample for speed: epoch 1000, then every 4000
    sampled = [common[0]] + [e for e in common if (e % 4000 == 0)]
    if 11000 not in sampled:
        sampled.append(11000)    # ensure grokking moment is included
    sampled = sorted(set(sampled))
    print(f'Will measure barrier at {len(sampled)} epochs: {sampled}')

    # Also include init and final
    init_G = load_state(G_dir / 'init.pt')
    init_M = load_state(M_dir / 'init.pt')
    final_G = load_state(G_dir / 'final.pt')
    final_M = load_state(M_dir / 'final.pt')

    epochs = [0] + sampled + [50000]
    barriers = []
    M_train_losses = []
    G_train_losses = []
    M_test_accs = []
    G_test_accs = []

    # Also compute test accuracy alongside
    test_pairs = None  # compute lazily
    _, test_pairs_list = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    test_in, test_lab = to_tensors(test_pairs_list, P, device='cpu')

    print(f'\n{"epoch":>7}  {"barrier":>12}  {"barrier_ratio":>14}  '
          f'{"M_test_acc":>11}  {"G_test_acc":>11}  {"M_train_loss":>14}  {"G_train_loss":>14}')
    for e in epochs:
        if e == 0:
            s_M, s_G = init_M, init_G
        elif e == 50000:
            s_M, s_G = final_M, final_G
        else:
            s_M = load_state(M_dir / f'epoch_{e}.pt')
            s_G = load_state(G_dir / f'epoch_{e}.pt')

        mid_loss, endpt_loss, ratio = barrier_between(s_M, s_G, train_in, train_lab)

        # Quick eval of M and G alone
        m_loss, _ = eval_loss_acc(load_model_from_state(s_M), train_in, train_lab)
        g_loss, _ = eval_loss_acc(load_model_from_state(s_G), train_in, train_lab)
        _, m_test_acc = eval_loss_acc(load_model_from_state(s_M), test_in, test_lab)
        _, g_test_acc = eval_loss_acc(load_model_from_state(s_G), test_in, test_lab)

        barriers.append(mid_loss)
        M_train_losses.append(m_loss)
        G_train_losses.append(g_loss)
        M_test_accs.append(m_test_acc)
        G_test_accs.append(g_test_acc)

        print(f'{e:>7}  {mid_loss:>12.4e}  {ratio:>14.2e}  {m_test_acc:>11.4f}  '
              f'{g_test_acc:>11.4f}  {m_loss:>14.4e}  {g_loss:>14.4e}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(epochs, barriers, marker='o', color='tab:red', label='barrier height (max loss at midpoint)')
    ax.plot(epochs, M_train_losses, marker='s', color='tab:blue', alpha=0.6, label='M train loss')
    ax.plot(epochs, G_train_losses, marker='^', color='tab:green', alpha=0.6, label='G train loss')
    ax.axvline(10800, color='gray', linestyle='--', alpha=0.5, label='G grokks (~epoch 10800)')
    ax.set_xlabel('training epoch')
    ax.set_ylabel('loss (log scale)')
    ax.set_yscale('log')
    ax.set_title('When did M-G basin barrier appear?')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, M_test_accs, marker='s', label='M test acc')
    ax.plot(epochs, G_test_accs, marker='^', label='G test acc')
    ax.axvline(10800, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('training epoch')
    ax.set_ylabel('test accuracy')
    ax.set_title('Test accuracy trajectories (for context)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = HERE / 'results' / 'fig_trajectory_basins.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
