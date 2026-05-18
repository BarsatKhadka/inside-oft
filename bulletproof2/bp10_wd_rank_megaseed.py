"""bp10: Mega-seed WD-rank law. The central paper figure.

11 WD values x 10 seeds = 110 runs. Log-linear fit with CIs on slope/intercept.

Records:
  - Final test acc
  - Final effective rank of W_out
  - Grok epoch
  - Membership-inference attack AUC (loss-based, for free as we have grads)

The same runs feed bp15 (MIA vs WD).
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.nn.functional as F

from _common import (HERE, P, cross_entropy_hp, effective_rank, eval_acc, train_one)

NUM_SEEDS = 10
NUM_EPOCHS = 20000
WD_VALUES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]


def mia_loss_auc(model, tr_in, tr_lab, te_in, te_lab):
    """Loss-based MIA: lower loss on member -> attack. Report AUC."""
    with t.no_grad():
        l_tr = F.cross_entropy(model(tr_in)[:, -1, :], tr_lab, reduction='none').cpu().numpy()
        l_te = F.cross_entropy(model(te_in)[:, -1, :], te_lab, reduction='none').cpu().numpy()
    # Member = train (loss lower means more confident -> member). Negative loss as score.
    scores = np.concatenate([-l_tr, -l_te])
    labels = np.concatenate([np.ones(len(l_tr)), np.zeros(len(l_te))])
    # ROC AUC computed manually
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    # trapezoidal
    auc = float(np.trapz(tpr, fpr))
    return auc


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = []
    for wd in WD_VALUES:
        for seed in range(NUM_SEEDS):
            print(f'\n--- wd={wd} seed={seed} ---')
            model, meta, (tr_in, tr_lab, te_in, te_lab) = train_one(
                seed=seed, wd=wd, num_epochs=NUM_EPOCHS, device=device)
            blk = model.blocks[0]
            rank = effective_rank(blk.mlp.W_out)
            mia = mia_loss_auc(model, tr_in, tr_lab, te_in, te_lab)
            entry = {
                'wd': wd, 'seed': seed,
                'final_test_acc': meta['final_test_acc'],
                'final_train_acc': meta['final_train_acc'],
                'grok_epoch': meta['grok_epoch'],
                'rank_W_out': rank,
                'rank_W_in': effective_rank(blk.mlp.W_in),
                'rank_W_E': effective_rank(model.embed.W_E),
                'rank_W_U': effective_rank(model.unembed.W_U),
                'mia_auc': mia,
            }
            results.append(entry)
            print(f'  test={meta["final_test_acc"]:.4f}, rank={rank:.2f}, mia={mia:.4f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp10_wd_rank_megaseed.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Log-linear fit on G-region (test_acc > 0.9)
    grokked = [r for r in results if r['final_test_acc'] > 0.9 and r['wd'] > 0]
    if len(grokked) > 5:
        x = np.log(np.array([r['wd'] for r in grokked]))
        y = np.log(np.array([r['rank_W_out'] for r in grokked]))
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        # Bootstrap CIs
        boot_slopes = []
        for _ in range(1000):
            idx = np.random.choice(len(x), len(x), replace=True)
            s, _ = np.polyfit(x[idx], y[idx], 1)
            boot_slopes.append(s)
        ci = np.percentile(boot_slopes, [2.5, 97.5])
        print(f'\nlog(rank) = {slope:.3f} * log(WD) + {intercept:.3f}')
        print(f'slope 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]')


if __name__ == '__main__':
    main()
