"""tier5 v2: produce a clean M vs G split for CharLM Shakespeare.

Why this exists: the original tier5 produced a clean separation but used DIFFERENT
hyperparameters for M (wd=0, dropout=0) and G (wd=1e-3, dropout=0.1). That
combines two effects (WD AND dropout) and the M wasn't pushed to catastrophic
memorization. tier5_v2 fixes both:

  - M: wd=0, dropout=0, train LONG (80k iters), take FINAL checkpoint
       (peak memorization).
  - G: wd=1e-3, dropout=0.1, train SAME duration, take EARLY-STOPPING checkpoint
       at the iter with best val_loss (peak generalization, before
       memorization eventually sets in).

Track val_loss every 2000 iters so the early-stopping extraction is precise.

Compared to tier5_v1, this gives:
  - Methodological consistency with tier6_v2 (same checkpoint-extraction story)
  - Cleaner contrast: G is at its BEST generalization point, not after WD has
    pulled it toward memorization
  - The val_loss trajectory itself is saved, so we can see HOW each regime
    evolves over training (publishable as a trajectory figure)
"""
import json
import urllib.request
from pathlib import Path
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bulletproof3._signatures import compute_full_battery
from bulletproof3.tier5_charlm_shakespeare import (
    CharLM, get_shakespeare, get_batch, CTX, BATCH, DIM, DEPTH, HEADS,
)

NUM_SEEDS = 3
N_ITERS = 80000
EVAL_EVERY = 2000  # check val_loss every 2k iters for early stopping precision
DATA_DIR = HERE.parent / 'data'


def train_v2(seed, mode, device, train, val, vocab):
    """Train and track val_loss. Return final state, best-val state, history."""
    t.manual_seed(seed); np.random.seed(seed)
    wd = 1e-3 if mode == 'G' else 0.0
    drop = 0.1 if mode == 'G' else 0.0
    model = CharLM(vocab, dropout=drop).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)

    val_history = []
    best_val = float('inf')
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_iter = 0

    print(f'  starting: mode={mode}, wd={wd}, drop={drop}, iters={N_ITERS}')

    for it in range(N_ITERS):
        model.train()
        x, y = get_batch(train, CTX, BATCH, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

        if (it + 1) % EVAL_EVERY == 0:
            model.eval()
            with t.no_grad():
                # Mean val loss over multiple batches for stability
                v_losses = []; t_losses = []
                for _ in range(8):
                    xv, yv = get_batch(val, CTX, BATCH, device)
                    v_losses.append(F.cross_entropy(
                        model(xv).reshape(-1, vocab), yv.reshape(-1)).item())
                    xt, yt = get_batch(train, CTX, BATCH, device)
                    t_losses.append(F.cross_entropy(
                        model(xt).reshape(-1, vocab), yt.reshape(-1)).item())
                tr_l = float(np.mean(t_losses))
                te_l = float(np.mean(v_losses))
            val_history.append({'iter': it + 1, 'train_loss': tr_l, 'test_loss': te_l})
            if te_l < best_val:
                best_val = te_l
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_iter = it + 1
            print(f'  it={it+1}: train={tr_l:.4f}, val={te_l:.4f}  '
                  f'(best val={best_val:.4f} at it={best_iter})')

    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {
        'model': model,
        'train': train, 'val': val, 'vocab': vocab,
        'final_state': final_state,
        'best_state': best_state,
        'best_iter': best_iter,
        'val_history': val_history,
    }


def signature_at_checkpoint(model, state_dict, train, val, vocab, device):
    model.load_state_dict(state_dict)
    model.eval()
    # Fixed eval set per seed: 32 sampled sequences from each
    t.manual_seed(123)
    np.random.seed(123)
    x_tr_eval, y_tr_eval = get_batch(train, CTX, 32, device)
    x_te_eval, y_te_eval = get_batch(val, CTX, 32, device)

    train_loss_fn = lambda: F.cross_entropy(
        model(x_tr_eval).reshape(-1, vocab), y_tr_eval.reshape(-1))
    test_loss_fn  = lambda: F.cross_entropy(
        model(x_te_eval).reshape(-1, vocab), y_te_eval.reshape(-1))

    @t.no_grad()
    def per_seq_loss(arr, n_samples=512):
        losses = []
        for _ in range(n_samples // 32):
            x, y = get_batch(arr, CTX, 32, device)
            per_tok = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1),
                                       reduction='none')
            per_seq = per_tok.reshape(32, -1).mean(1).cpu().numpy()
            losses.append(per_seq)
        return np.concatenate(losses)
    tr_losses = per_seq_loss(train)
    te_losses = per_seq_loss(val)

    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=10, verbose=True)
    bat['mean_train_loss'] = float(tr_losses.mean())
    bat['mean_test_loss'] = float(te_losses.mean())
    bat['gap_loss'] = float(te_losses.mean() - tr_losses.mean())
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'tier5_v2_real_split.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train, val, vocab = get_shakespeare()
    print(f'Loaded Shakespeare: train={len(train)} chars, val={len(val)} chars, vocab={vocab}')

    results = {'M': [], 'G': []}
    for seed in range(NUM_SEEDS):
        for mode in ['M', 'G']:
            print(f'\n=== {mode} seed={seed} ===')
            try:
                run = train_v2(seed, mode, device, train, val, vocab)
                # M -> final state, G -> best-val state
                if mode == 'M':
                    state = run['final_state']
                    checkpoint_iter = N_ITERS
                else:
                    state = run['best_state']
                    checkpoint_iter = run['best_iter']
                print(f'  computing signatures at {mode} checkpoint (iter {checkpoint_iter})...')
                bat = signature_at_checkpoint(run['model'], state,
                                                run['train'], run['val'],
                                                run['vocab'], device)
                bat['mode'] = mode
                bat['seed'] = seed
                bat['checkpoint_iter'] = checkpoint_iter
                bat['val_history'] = run['val_history']
                if 'hessian_eigs_full' in bat:
                    bat['hessian_eigs_full'] = bat['hessian_eigs_full'][:20]
                results[mode].append(bat)
                print(f'  {mode} seed={seed}: gap_loss={bat["gap_loss"]:.4f} '
                      f'mia={bat.get("mia_loss_auc", 0):.4f} '
                      f'top={bat["hessian_top_full"]:.3f} '
                      f'bot={bat["hessian_bot_full"]:.3f} '
                      f'(iter {checkpoint_iter})')
            except Exception as e:
                print(f'  error: {e}')
                import traceback; traceback.print_exc()
                results[mode].append({'mode': mode, 'seed': seed, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
    print('\n=== tier5_v2 summary ===')
    for mode in ('M', 'G'):
        valid = [r for r in results[mode] if 'error' not in r]
        if not valid: continue
        gaps = [r['gap_loss'] for r in valid]
        mias = [r.get('mia_loss_auc', float('nan')) for r in valid]
        tops = [r['hessian_top_full'] for r in valid]
        print(f'  {mode}: gap mean={np.mean(gaps):.3f}, '
              f'MIA mean={np.mean(mias):.4f}, '
              f'top eig mean={np.mean(tops):.2f} (n={len(valid)})')


if __name__ == '__main__':
    main()
