"""bp14: MIA battery - LiRA, shadow-model, logit-margin, loss-based.

Run all four attacks on M and G models (5 seeds each). Report AUC per attack.

Predictions:
  - M: high AUC across all attacks (especially LiRA)
  - G: near-random (0.5)
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.nn.functional as F

from _common import (HERE, P, train_one, eval_acc)

NUM_SEEDS = 5
NUM_SHADOWS = 8  # for LiRA-style attack
NUM_EPOCHS = 20000


def auc_from_scores(scores, labels):
    order = np.argsort(-scores)
    labels_sorted = np.asarray(labels)[order]
    n_pos = labels_sorted.sum(); n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    tps = np.cumsum(labels_sorted); fps = np.cumsum(1 - labels_sorted)
    tpr = tps / n_pos; fpr = fps / n_neg
    return float(np.trapz(tpr, fpr))


def attack_loss(model, tr_in, tr_lab, te_in, te_lab):
    with t.no_grad():
        l_tr = F.cross_entropy(model(tr_in)[:, -1, :], tr_lab, reduction='none').cpu().numpy()
        l_te = F.cross_entropy(model(te_in)[:, -1, :], te_lab, reduction='none').cpu().numpy()
    s = np.concatenate([-l_tr, -l_te]); y = np.concatenate([np.ones_like(l_tr), np.zeros_like(l_te)])
    return auc_from_scores(s, y)


def attack_logit_margin(model, tr_in, tr_lab, te_in, te_lab):
    with t.no_grad():
        log_tr = model(tr_in)[:, -1, :]
        log_te = model(te_in)[:, -1, :]
    def margin(L, y):
        true_l = L[t.arange(len(y)), y]
        other_l = L.scatter(1, y[:, None], -1e9).max(1).values
        return (true_l - other_l).cpu().numpy()
    m_tr = margin(log_tr, tr_lab); m_te = margin(log_te, te_lab)
    s = np.concatenate([m_tr, m_te]); y_ = np.concatenate([np.ones_like(m_tr), np.zeros_like(m_te)])
    return auc_from_scores(s, y_)


def attack_shadow(target_model, shadow_models, tr_in, tr_lab, te_in, te_lab):
    """Simple shadow-model attack: use shadow loss distributions to calibrate."""
    with t.no_grad():
        l_tr = F.cross_entropy(target_model(tr_in)[:, -1, :], tr_lab, reduction='none').cpu().numpy()
        l_te = F.cross_entropy(target_model(te_in)[:, -1, :], te_lab, reduction='none').cpu().numpy()
        # Shadow loss distributions on each example
        shadow_losses_tr = np.stack([
            F.cross_entropy(m(tr_in)[:, -1, :], tr_lab, reduction='none').cpu().numpy()
            for m in shadow_models])
        shadow_losses_te = np.stack([
            F.cross_entropy(m(te_in)[:, -1, :], te_lab, reduction='none').cpu().numpy()
            for m in shadow_models])
    mu_tr = shadow_losses_tr.mean(0); sd_tr = shadow_losses_tr.std(0) + 1e-6
    mu_te = shadow_losses_te.mean(0); sd_te = shadow_losses_te.std(0) + 1e-6
    z_tr = (mu_tr - l_tr) / sd_tr; z_te = (mu_te - l_te) / sd_te
    s = np.concatenate([z_tr, z_te]); y = np.concatenate([np.ones_like(z_tr), np.zeros_like(z_te)])
    return auc_from_scores(s, y)


def attack_lira_offline(target_model, shadow_models, tr_in, tr_lab, te_in, te_lab):
    """LiRA-style offline: for each example, compare target loss to shadow distribution
    via Gaussian likelihood ratio. Offline means we don't retrain target without each example.
    """
    with t.no_grad():
        l_tr = F.cross_entropy(target_model(tr_in)[:, -1, :], tr_lab, reduction='none').cpu().numpy()
        l_te = F.cross_entropy(target_model(te_in)[:, -1, :], te_lab, reduction='none').cpu().numpy()
        shadow_losses_tr = np.stack([
            F.cross_entropy(m(tr_in)[:, -1, :], tr_lab, reduction='none').cpu().numpy()
            for m in shadow_models])
        shadow_losses_te = np.stack([
            F.cross_entropy(m(te_in)[:, -1, :], te_lab, reduction='none').cpu().numpy()
            for m in shadow_models])
    # log-transform losses (LiRA recommends)
    def lt(x): return np.log(np.clip(x, 1e-12, None))
    z_tr = (lt(shadow_losses_tr.mean(0)) - lt(l_tr)) / (lt(shadow_losses_tr).std(0) + 1e-6)
    z_te = (lt(shadow_losses_te.mean(0)) - lt(l_te)) / (lt(shadow_losses_te).std(0) + 1e-6)
    s = np.concatenate([z_tr, z_te]); y = np.concatenate([np.ones_like(z_tr), np.zeros_like(z_te)])
    return auc_from_scores(s, y)


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    # Train shadow models (G regime for shadows is fine - they just need diff seeds)
    print('=== Training shadow models ===')
    shadows = []
    for s in range(NUM_SHADOWS):
        print(f' shadow {s}'); m, _, _ = train_one(seed=100+s, wd=1.0, num_epochs=NUM_EPOCHS, device=device)
        shadows.append(m)
    results = {'M': [], 'G': []}
    for label, wd in [('M', 0.0), ('G', 1.0)]:
        for seed in range(NUM_SEEDS):
            print(f'\n=== target {label} seed={seed} ===')
            model, meta, (tr_in, tr_lab, te_in, te_lab) = train_one(
                seed=seed, wd=wd, num_epochs=NUM_EPOCHS, device=device)
            entry = {
                'seed': seed, 'wd': wd, 'final_test_acc': meta['final_test_acc'],
                'auc_loss':   attack_loss(model, tr_in, tr_lab, te_in, te_lab),
                'auc_margin': attack_logit_margin(model, tr_in, tr_lab, te_in, te_lab),
                'auc_shadow': attack_shadow(model, shadows, tr_in, tr_lab, te_in, te_lab),
                'auc_lira':   attack_lira_offline(model, shadows, tr_in, tr_lab, te_in, te_lab),
            }
            results[label].append(entry)
            print(f'  loss={entry["auc_loss"]:.4f}, margin={entry["auc_margin"]:.4f}, '
                  f'shadow={entry["auc_shadow"]:.4f}, lira={entry["auc_lira"]:.4f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp14_mia_battery.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\n=== Summary ===')
    for label in ['M', 'G']:
        for k in ['auc_loss', 'auc_margin', 'auc_shadow', 'auc_lira']:
            xs = np.array([r[k] for r in results[label]])
            print(f'  {label} {k}: {xs.mean():.4f} +- {xs.std():.4f}')


if __name__ == '__main__':
    main()
