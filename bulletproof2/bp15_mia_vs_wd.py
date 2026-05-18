"""bp15: MIA AUC across WD sweep (also computes rank).

Predicts: MIA leak drops sharply at WD threshold, mirroring rank drop.
This produces a 2-panel correlation figure (rank vs WD, MIA vs WD).

Reuses bp10 if results exist; otherwise re-runs.
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.nn.functional as F

from _common import (HERE, train_one, effective_rank)

NUM_SEEDS = 5
NUM_EPOCHS = 20000
WD_VALUES = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0]


def auc_from_scores(scores, labels):
    order = np.argsort(-scores)
    labels_sorted = np.asarray(labels)[order]
    n_pos = labels_sorted.sum(); n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    tps = np.cumsum(labels_sorted); fps = np.cumsum(1 - labels_sorted)
    return float(np.trapz(tps / n_pos, fps / n_neg))


def loss_mia(model, tr_in, tr_lab, te_in, te_lab):
    with t.no_grad():
        l_tr = F.cross_entropy(model(tr_in)[:, -1, :], tr_lab, reduction='none').cpu().numpy()
        l_te = F.cross_entropy(model(te_in)[:, -1, :], te_lab, reduction='none').cpu().numpy()
    s = np.concatenate([-l_tr, -l_te]); y = np.concatenate([np.ones_like(l_tr), np.zeros_like(l_te)])
    return auc_from_scores(s, y)


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = []
    for wd in WD_VALUES:
        for seed in range(NUM_SEEDS):
            print(f'\n--- wd={wd} seed={seed} ---')
            model, meta, (tr_in, tr_lab, te_in, te_lab) = train_one(
                seed=seed, wd=wd, num_epochs=NUM_EPOCHS, device=device)
            entry = {
                'wd': wd, 'seed': seed,
                'final_test_acc': meta['final_test_acc'],
                'rank_W_out': effective_rank(model.blocks[0].mlp.W_out),
                'mia_auc': loss_mia(model, tr_in, tr_lab, te_in, te_lab),
            }
            results.append(entry)
            print(f'  test={meta["final_test_acc"]:.4f}, rank={entry["rank_W_out"]:.2f}, mia={entry["mia_auc"]:.4f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp15_mia_vs_wd.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Correlation
    rs = np.array([r['rank_W_out'] for r in results])
    ms = np.array([r['mia_auc'] for r in results])
    corr = np.corrcoef(rs, ms)[0, 1]
    print(f'\nPearson corr(rank, MIA) = {corr:.4f}')


if __name__ == '__main__':
    main()
