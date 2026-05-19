"""tier6 v2: produce a real M vs G split for Pythia fine-tuning.

Why this exists: the original tier6 collapsed because 200 chunks isn't enough
training data for Pythia-160m to NOT memorize everything regardless of WD.
This version fixes that by:
  (a) 2000 training chunks (10x more than v1) — model can't memorize all
  (b) Track val_loss every epoch and extract G at BEST val_loss epoch
      (early stopping captures the generalizing solution before WD eventually
      loses to overfitting)
  (c) M is the LAST epoch (let it overfit fully)

With 2000 chunks of Pride and Prejudice + early stopping for G,
we should get a real M (overfit) vs G (generalize) comparison.

Outputs per regime: full signature battery at the relevant checkpoint
(last epoch for M; best-val-loss epoch for G).
"""
import json
import urllib.request
import copy
from pathlib import Path
import numpy as np
import torch as t
import torch.nn.functional as F
import torch.optim as optim

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bulletproof3._signatures import compute_full_battery
from bulletproof3.tier6_pythia_finetune import (
    get_corpus, build_chunks, build_model_and_tokenizer,
    batch_loss, per_chunk_loss,
)

NUM_SEEDS = 2
EPOCHS = 30
BATCH = 4
N_TRAIN = 2000       # 10x more than v1
N_TEST = 500
CHUNK_LEN = 256


def train_v2(seed, mode, device):
    """Train and track val_loss every epoch. Return:
      - final model state (M's checkpoint)
      - best-val-loss model state (G's checkpoint)
      - val_loss history
    """
    t.manual_seed(seed); np.random.seed(seed)
    wd = 0.1 if mode == 'G' else 0.0
    model, tok, name = build_model_and_tokenizer(device)
    text = get_corpus()
    train_chunks, test_chunks = build_chunks(text, tok, CHUNK_LEN, N_TRAIN, N_TEST, seed=seed)
    print(f'  data: {len(train_chunks)} train chunks, {len(test_chunks)} test chunks')

    opt = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=wd, eps=1e-8)
    val_history = []
    best_val = float('inf')
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = -1

    print(f'  starting fine-tune: mode={mode}, wd={wd}, epochs={EPOCHS}, n_train={N_TRAIN}')

    for ep in range(EPOCHS):
        model.train()
        rng = np.random.RandomState(ep + seed * 1000)
        order = rng.permutation(len(train_chunks))
        nan_batches = 0; n_batches = 0
        for i in range(0, len(train_chunks), BATCH):
            batch = [train_chunks[j] for j in order[i:i + BATCH]]
            opt.zero_grad()
            loss = batch_loss(model, batch, device)
            if not t.isfinite(loss):
                nan_batches += 1; continue
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); n_batches += 1

        # Eval every epoch (we need fine-grained val_loss for early stopping G)
        model.eval()
        tr_l = per_chunk_loss(model, train_chunks[:128], device).mean()
        te_l = per_chunk_loss(model, test_chunks[:128], device).mean()
        val_history.append({'epoch': ep + 1, 'train_loss': float(tr_l), 'test_loss': float(te_l)})

        if te_l < best_val:
            best_val = float(te_l)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1

        if (ep + 1) % 2 == 0 or ep == 0:
            print(f'  ep={ep+1}: train_loss={tr_l:.4f}, test_loss={te_l:.4f}  '
                  f'(best_val={best_val:.4f} at ep={best_epoch})  nan={nan_batches}')

    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {
        'model': model,
        'tok': tok,
        'name': name,
        'train_chunks': train_chunks,
        'test_chunks': test_chunks,
        'final_state': final_state,
        'best_state': best_state,
        'best_epoch': best_epoch,
        'val_history': val_history,
    }


def signature_at_checkpoint(model, state_dict, train_chunks, test_chunks, device):
    """Load state_dict into model and compute the full signature battery."""
    model.load_state_dict(state_dict)
    model.eval()
    tr_b = train_chunks[:8]; te_b = test_chunks[:8]
    train_loss_fn = lambda: batch_loss(model, tr_b, device)
    test_loss_fn  = lambda: batch_loss(model, te_b, device)
    tr_losses = per_chunk_loss(model, train_chunks, device)
    te_losses = per_chunk_loss(model, test_chunks, device)
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=5, verbose=True)
    bat['mean_train_loss'] = float(tr_losses.mean())
    bat['mean_test_loss'] = float(te_losses.mean())
    bat['gap_loss'] = float(te_losses.mean() - tr_losses.mean())
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'tier6_v2_real_split.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {'M': [], 'G': []}
    for seed in range(NUM_SEEDS):
        # Train two independent runs: M (wd=0) and G (wd=0.1)
        # For M we use the FINAL state (let it overfit fully)
        # For G we use the BEST-val state (early stopping)
        for mode in ['M', 'G']:
            print(f'\n=== {mode} seed={seed} ===')
            try:
                run = train_v2(seed, mode, device)
                model = run['model']
                # Pick the right state for this regime:
                #   M = final (let it memorize fully)
                #   G = best val loss (early stopping)
                if mode == 'M':
                    state = run['final_state']
                    checkpoint_epoch = EPOCHS
                else:
                    state = run['best_state']
                    checkpoint_epoch = run['best_epoch']
                print(f'  computing signatures at {mode} checkpoint (epoch {checkpoint_epoch})...')
                bat = signature_at_checkpoint(model, state, run['train_chunks'],
                                                run['test_chunks'], device)
                bat['mode'] = mode
                bat['seed'] = seed
                bat['checkpoint_epoch'] = checkpoint_epoch
                bat['val_history'] = run['val_history']
                bat['model_name'] = run['name']
                # cull big lists
                if 'hessian_eigs_full' in bat:
                    bat['hessian_eigs_full'] = bat['hessian_eigs_full'][:20]
                if 'ranks' in bat:
                    bat['ranks'] = {k: v for k, v in bat['ranks'].items()
                                     if any(s in k.lower() for s in
                                             ['mlp', 'embed', 'attention', 'dense', 'query', 'key', 'value'])}
                results[mode].append(bat)
                print(f'  {mode} seed={seed}: gap_loss={bat["gap_loss"]:.4f} '
                      f'mia={bat.get("mia_loss_auc", 0):.4f} '
                      f'top={bat["hessian_top_full"]:.3f} '
                      f'bot={bat["hessian_bot_full"]:.3f} '
                      f'(checkpoint ep={checkpoint_epoch})')
            except Exception as e:
                print(f'  error: {e}')
                import traceback; traceback.print_exc()
                results[mode].append({'mode': mode, 'seed': seed, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
    # Final summary
    print('\n=== tier6_v2 summary ===')
    for mode in ('M', 'G'):
        valid = [r for r in results[mode] if 'error' not in r]
        if not valid: continue
        gaps = [r['gap_loss'] for r in valid]
        mias = [r.get('mia_loss_auc', float('nan')) for r in valid]
        print(f'  {mode}: gap mean={np.mean(gaps):.3f}, MIA mean={np.mean(mias):.4f} (n={len(valid)})')


if __name__ == '__main__':
    main()
