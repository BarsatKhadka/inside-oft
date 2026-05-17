"""Multi-seed analysis: are memorizing-type and generalizing-type basins real categories?

We have 3 G's (G_seed0, G_seed1, G_seed2) and 3 M's. Three questions:

  1. WITHIN-CATEGORY CONSISTENCY
     Do all 3 G's have similar structure (rank, probe signature)? All 3 M's?
     If yes -> "basin type" is a real category, not random.

  2. CROSS-SEED MODE CONNECTIVITY
     Compute barriers for:
       - M_i vs M_j (within-M)
       - G_i vs G_j (within-G)
       - M_i vs G_j (between)
     If within-M barriers are SMALL but between-M-G barriers are LARGE
     -> M's form a connected family, G's form a connected family, two distinct
     basins separated.
     If all barriers are large -> each model is its own basin.

  3. PROBE CONSISTENCY
     Run the (a, b)-recovery probe on all 6. Do M's all hit ~88%? G's all ~43%?
     If yes -> "input preservation" is a property of basin type, not seed.

Usage:
    python taska/analysis/multiseed.py
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
SEED = 0


def load_state(p):
    return t.load(p, map_location='cpu', weights_only=True)['model']


def make_model(state):
    m = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    m.load_state_dict(state)
    m.eval()
    return m


def effective_rank(W):
    s = t.linalg.svdvals(W)
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def stable_rank(W):
    s = t.linalg.svdvals(W)
    return float((s ** 2).sum() / s[0] ** 2)


@t.no_grad()
def eval_loss_acc(model, inputs, labels):
    logits = model(inputs)[:, -1, :].to(t.float64)
    lp = t.nn.functional.log_softmax(logits, dim=-1)
    loss = -lp[t.arange(labels.shape[0]), labels].mean().item()
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


def interp_state(s1, s2, alpha):
    return {k: (1 - alpha) * s1[k] + alpha * s2[k] for k in s1}


def barrier(s1, s2, inputs, labels, n_alpha=11):
    alphas = np.linspace(0, 1, n_alpha)
    losses = [eval_loss_acc(make_model(interp_state(s1, s2, a)), inputs, labels)[0] for a in alphas]
    return max(losses[1:-1]), max(losses[0], losses[-1])


@t.no_grad()
def capture_resid_post(model, inputs):
    x = model.embed(inputs)
    x = model.pos_embed(x)
    block = model.blocks[0]
    x = x + block.attn(x)
    x = x + block.mlp(x)
    return x[:, -1, :].numpy()


def probe_sel(model, inputs, train_pairs):
    """Selectivity for predict_a at resid_post."""
    acts = capture_resid_post(model, inputs)
    a_arr = np.array([a for a, b, _ in train_pairs])
    rng = np.random.RandomState(0)
    shuffled = rng.permutation(a_arr)

    # Real labels
    X_tr, X_te, y_tr, y_te = train_test_split(acts, a_arr, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc_real = clf.score(X_te, y_te)
    # Shuffled
    X_tr, X_te, y_tr, y_te = train_test_split(acts, shuffled, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc_shuf = clf.score(X_te, y_te)
    return acc_real - acc_shuf, acc_real, acc_shuf


def main():
    checkpoints = {
        'M_s0': HERE / 'checkpoints' / 'M' / 'final.pt',
        'M_s1': HERE / 'checkpoints' / 'M_seed1' / 'final.pt',
        'M_s2': HERE / 'checkpoints' / 'M_seed2' / 'final.pt',
        'G_s0': HERE / 'checkpoints' / 'G' / 'final.pt',
        'G_s1': HERE / 'checkpoints' / 'G_seed1' / 'final.pt',
        'G_s2': HERE / 'checkpoints' / 'G_seed2' / 'final.pt',
    }
    states = {name: load_state(p) for name, p in checkpoints.items()}
    print('Loaded:', list(states.keys()))

    # Each model has its own train/test split (based on its seed). For
    # cross-seed mode connectivity, we want a SHARED loss function -- use
    # ALL pairs (the full landscape) for evaluation. This is consistent and
    # measures loss on the full data distribution.
    all_pairs = [(i, j, P) for i in range(P) for j in range(P)]
    full_in, full_lab = to_tensors(all_pairs, P, device='cpu')

    # For probe: each model has its own train pairs (probe ought to use the
    # data that model was trained on). Build per-seed.
    seed_map = {'M_s0': 0, 'M_s1': 1, 'M_s2': 2,
                'G_s0': 0, 'G_s1': 1, 'G_s2': 2}
    train_pairs_by_seed = {
        s: gen_train_test(p=P, frac_train=0.3, seed=s)[0] for s in {0, 1, 2}
    }

    # Sanity check: each model evaluated on its OWN train pairs should have ~0 loss
    print('\nSanity check (each model on its own train pairs):')
    for name, s in states.items():
        own_seed = seed_map[name]
        own_pairs = train_pairs_by_seed[own_seed]
        own_in, own_lab = to_tensors(own_pairs, P, device='cpu')
        loss, acc = eval_loss_acc(make_model(s), own_in, own_lab)
        print(f'  {name}: train_loss={loss:.4e}  train_acc={acc:.4f}')

    # ============================================================
    # 1. WITHIN-CATEGORY CONSISTENCY: rank table
    # ============================================================
    print('\n' + '=' * 80)
    print('PART 1: Effective rank of each weight matrix')
    print('=' * 80)
    print(f'\n{"model":>8}  {"W_E_er":>8}  {"W_in_er":>8}  {"W_out_er":>8}  {"W_E_sr":>8}  {"W_in_sr":>8}  {"W_out_sr":>8}')
    ranks = {}
    for name, s in states.items():
        W_E = s['embed.W_E'][:, :P]
        W_in = s['blocks.0.mlp.W_in']
        W_out = s['blocks.0.mlp.W_out']
        ranks[name] = {
            'W_E_er': effective_rank(W_E),
            'W_in_er': effective_rank(W_in),
            'W_out_er': effective_rank(W_out),
            'W_E_sr': stable_rank(W_E),
            'W_in_sr': stable_rank(W_in),
            'W_out_sr': stable_rank(W_out),
        }
        r = ranks[name]
        print(f'{name:>8}  {r["W_E_er"]:>8.2f}  {r["W_in_er"]:>8.2f}  {r["W_out_er"]:>8.2f}  '
              f'{r["W_E_sr"]:>8.2f}  {r["W_in_sr"]:>8.2f}  {r["W_out_sr"]:>8.2f}')

    # ============================================================
    # 2. CROSS-SEED MODE CONNECTIVITY
    # ============================================================
    print('\n' + '=' * 80)
    print('PART 2: Pairwise barriers (max midpoint loss / max endpoint loss)')
    print('=' * 80)
    names = list(states.keys())
    barrier_mat = np.zeros((len(names), len(names)))
    print('Using FULL pairs (all 12769) as the loss function -- consistent across all model pairs.')
    print(f'\n{"":>6}  ' + '  '.join(f'{n:>12}' for n in names))
    for i, n1 in enumerate(names):
        row = []
        for j, n2 in enumerate(names):
            if i >= j:
                row.append('--' if i == j else f'{barrier_mat[i, j]:.2e}')
                continue
            mid, endpt = barrier(states[n1], states[n2], full_in, full_lab)
            ratio = mid / max(endpt, 1e-20)
            barrier_mat[i, j] = ratio
            barrier_mat[j, i] = ratio
            row.append(f'{mid:.2e}')   # show absolute mid loss (more meaningful than ratio when endpoints are noisy)
        print(f'{n1:>6}  ' + '  '.join(f'{x:>12}' for x in row))
    print('(values are MAX midpoint loss on full data; ~0 = no barrier; large = barrier)')

    # Summary: within-category vs between-category
    M_idx = [i for i, n in enumerate(names) if n.startswith('M')]
    G_idx = [i for i, n in enumerate(names) if n.startswith('G')]
    within_M = [barrier_mat[i, j] for i in M_idx for j in M_idx if i < j]
    within_G = [barrier_mat[i, j] for i in G_idx for j in G_idx if i < j]
    between = [barrier_mat[i, j] for i in M_idx for j in G_idx]
    print(f'\nMean barrier ratio  WITHIN M-M:  {np.mean(within_M):.2e}  (n={len(within_M)})')
    print(f'Mean barrier ratio  WITHIN G-G:  {np.mean(within_G):.2e}  (n={len(within_G)})')
    print(f'Mean barrier ratio  BETWEEN M-G: {np.mean(between):.2e}  (n={len(between)})')

    # ============================================================
    # 3. PROBE CONSISTENCY
    # ============================================================
    print('\n' + '=' * 80)
    print('PART 3: Per-example probe sel(a) at resid_post')
    print('=' * 80)
    print(f'\n{"model":>8}  {"sel(a)":>10}  {"acc_real":>10}  {"acc_shuf":>10}')
    probe_results = {}
    for name, s in states.items():
        own_seed = seed_map[name]
        own_pairs = train_pairs_by_seed[own_seed]
        own_in, _ = to_tensors(own_pairs, P, device='cpu')
        sel, real, shuf = probe_sel(make_model(s), own_in, own_pairs)
        probe_results[name] = sel
        print(f'{name:>8}  {sel:>10.4f}  {real:>10.4f}  {shuf:>10.4f}')

    # ============================================================
    # Visualizations
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: effective rank bar chart
    ax = axes[0]
    matrices = ['W_E_er', 'W_in_er', 'W_out_er']
    bar_w = 0.13
    x = np.arange(len(matrices))
    for i, name in enumerate(names):
        ys = [ranks[name][m] for m in matrices]
        ax.bar(x + i * bar_w, ys, bar_w, label=name,
               color='tab:red' if name.startswith('M') else 'tab:blue', alpha=0.4 + 0.2 * int(name[-1]))
    ax.set_xticks(x + 2.5 * bar_w)
    ax.set_xticklabels(['W_E', 'W_in', 'W_out'])
    ax.set_ylabel('effective rank')
    ax.set_title('Effective rank by model')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: barrier matrix heatmap (log scale)
    ax = axes[1]
    im = ax.imshow(np.log10(barrier_mat + 1), cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    ax.set_title('Pairwise barriers (log10 ratio)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                ax.text(j, i, f'{barrier_mat[i, j]:.0e}', ha='center', va='center',
                        fontsize=6, color='black')

    # Panel 3: probe sel by model
    ax = axes[2]
    sels = [probe_results[n] for n in names]
    colors = ['tab:red' if n.startswith('M') else 'tab:blue' for n in names]
    ax.bar(names, sels, color=colors)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('sel(a) at resid_post')
    ax.set_title('Per-example probe selectivity by model')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Multi-seed: are memorizing-type and generalizing-type basins real categories?')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_multiseed.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
