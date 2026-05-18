"""bp7: 10 M + 10 G seeds with full structural battery.

For each of 10 seeds, train M (wd=0) and G (wd=1.0) on (a+b) mod 113.
Compute battery:
  - Effective rank of each weight matrix
  - Gradient norm on train, test, full data
  - Per-position embedding rank
  - Probe selectivity (per-example identity probe)
  - Attention asymmetry (max - min head Frob norm)
  - Output logit margin distribution stats

Output: 20 vectors of features -> PCA/tSNE clustering should separate categorically.
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.nn.functional as F

from _common import (HERE, P, cross_entropy_hp, effective_rank, eval_acc,
                     grad_norm_full, train_one)

NUM_SEEDS = 10
NUM_EPOCHS = 20000


def battery(model, train_in, train_lab, test_in, test_lab, device):
    feats = {}
    # 1-9: rank of each significant weight matrix
    feats['rank_W_E'] = effective_rank(model.embed.W_E)
    feats['rank_W_U'] = effective_rank(model.unembed.W_U)
    feats['rank_W_pos'] = effective_rank(model.pos_embed.W_pos)
    blk = model.blocks[0]
    feats['rank_W_Q'] = effective_rank(blk.attn.W_Q.reshape(-1, blk.attn.W_Q.shape[-1]))
    feats['rank_W_K'] = effective_rank(blk.attn.W_K.reshape(-1, blk.attn.W_K.shape[-1]))
    feats['rank_W_V'] = effective_rank(blk.attn.W_V.reshape(-1, blk.attn.W_V.shape[-1]))
    feats['rank_W_O'] = effective_rank(blk.attn.W_O)
    feats['rank_W_in'] = effective_rank(blk.mlp.W_in)
    feats['rank_W_out'] = effective_rank(blk.mlp.W_out)
    # 10-12: gradient norms
    full_in = t.cat([train_in, test_in])
    full_lab = t.cat([train_lab, test_lab])
    feats['grad_train'] = grad_norm_full(model, train_in, train_lab)
    feats['grad_test'] = grad_norm_full(model, test_in, test_lab)
    feats['grad_full'] = grad_norm_full(model, full_in, full_lab)
    feats['grad_test_over_train'] = feats['grad_test'] / max(feats['grad_train'], 1e-12)
    # 13-14: attention asymmetry
    with t.no_grad():
        q_frob = [float(blk.attn.W_Q[h].norm()) for h in range(blk.attn.W_Q.shape[0])]
        feats['attn_Q_asym'] = max(q_frob) - min(q_frob)
        feats['attn_Q_max'] = max(q_frob)
    # 15-17: logit margin stats on train set
    with t.no_grad():
        logits = model(train_in)[:, -1, :]
        margins = logits[t.arange(len(train_lab)), train_lab] - logits.scatter(1, train_lab[:, None], -1e9).max(1).values
        feats['margin_mean'] = float(margins.mean())
        feats['margin_std'] = float(margins.std())
        feats['margin_min'] = float(margins.min())
    # 18-19: residual stream effective rank at last position
    with t.no_grad():
        x = model.embed(full_in)
        x = model.pos_embed(x)
        for b in model.blocks:
            x = b(x)
        residual_last = x[:, -1, :]
        feats['resid_rank'] = effective_rank(residual_last.t())
    # 20: train loss
    with t.no_grad():
        feats['train_loss'] = float(cross_entropy_hp(model(train_in)[:, -1, :], train_lab))
    # 21: nuclear norm of W_out
    with t.no_grad():
        s = t.linalg.svdvals(blk.mlp.W_out.detach().cpu().float())
        feats['nuclear_W_out'] = float(s.sum())
        feats['op_norm_W_out'] = float(s[0])
        feats['stable_rank_W_out'] = float((s ** 2).sum() / (s[0] ** 2))
    return feats


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    for seed in range(NUM_SEEDS):
        for label, wd in [('M', 0.0), ('G', 1.0)]:
            print(f'\n=== {label} seed={seed} ===')
            model, meta, (tr_in, tr_lab, te_in, te_lab) = train_one(
                seed=seed, wd=wd, num_epochs=NUM_EPOCHS, device=device)
            feats = battery(model, tr_in, tr_lab, te_in, te_lab, device)
            entry = {**meta, 'features': feats}
            del entry['history']
            results[label].append(entry)
            print(f'  test_acc={meta["final_test_acc"]:.4f}, rank_W_out={feats["rank_W_out"]:.2f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp7_structural_battery.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Categorical clustering check
    print('\n=== Categorical clustering ===')
    keys = list(results['M'][0]['features'].keys())
    M = np.array([[r['features'][k] for k in keys] for r in results['M']])
    G = np.array([[r['features'][k] for k in keys] for r in results['G']])
    print(f'M shape={M.shape}, G shape={G.shape}')
    # Standardize and run PCA
    X = np.vstack([M, G])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc = U[:, :2] * S[:2]
    print('PC1 mean: M={:.3f}, G={:.3f}'.format(pc[:NUM_SEEDS, 0].mean(), pc[NUM_SEEDS:, 0].mean()))
    print('PC1 sep (mean_M - mean_G) / pooled_std = {:.3f}'.format(
        (pc[:NUM_SEEDS, 0].mean() - pc[NUM_SEEDS:, 0].mean()) /
        np.sqrt(0.5 * (pc[:NUM_SEEDS, 0].var() + pc[NUM_SEEDS:, 0].var()) + 1e-12)))


if __name__ == '__main__':
    main()
