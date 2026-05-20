"""tier6_v3: Proper M vs G for Pythia-160m fine-tuning.

WHY v2 FAILED TO PRODUCE A GENUINE G:
  - G used WD=0.1 with LR=1e-5. In AdamW, the WD update per step is
    -LR * WD * theta = -1e-6 * theta per step. With 500 steps/epoch, this
    is -5e-4 * theta per epoch -- tiny shrinkage. BUT the issue is that
    WD=0.1 in the gradient signal dominates over the actual fine-tuning
    gradient at LR=1e-5. The model never improves test loss below the
    pretrained baseline. val_history shows test going UP from epoch 1.
    G at epoch 1 is just the pretrained model with one gradient step.

THE FIX:
  - G: LR=2e-5 (2x higher) + WD=0.001 (100x lower). Now the training
       signal dominates, and the model can actually learn Victorian text
       patterns. Early stopping prevents full memorization.
  - M: LR=1e-4 (10x higher) + WD=0. Memorizes aggressively. Train loss
       should reach <0.1 by epoch 15-20.
  - Baseline: measure pretrained loss at epoch 0 so we can verify G
              actually improves from the pretrained state.

WHAT A SUCCESSFUL RUN LOOKS LIKE:
  - Baseline (epoch 0): test_loss ~ 3.0-3.5 (Pythia on Victorian text)
  - G best checkpoint: test_loss < baseline (genuine improvement)
  - M final (epoch 30): test_loss >> baseline (catastrophic forgetting)
  - MIA separates (1.0 vs ~0.7)

If G's best test_loss does NOT drop below baseline, experiment fails.
Check val_history and try reducing G_LR or increasing EPOCHS.
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
    build_model_and_tokenizer,
    batch_loss, per_chunk_loss,
)
from bulletproof3.tier6_v2_real_split import get_multi_book_corpus, build_chunks_safe

DATA_DIR = HERE.parent / 'data'

NUM_SEEDS  = 2
EPOCHS     = 30
BATCH      = 4
N_TRAIN    = 2000
N_TEST     = 500
CHUNK_LEN  = 256

# KEY CHANGE from v2:
G_LR = 2e-5    # 2× v2 -- enough signal to actually learn
G_WD = 0.001   # 100× lower than v2 -- learning dominates over WD
M_LR = 1e-4    # 10× v2 -- fast memorization
M_WD = 0.0


def measure_baseline(device, train_chunks, test_chunks):
    """Measure pretrained Pythia loss before any fine-tuning."""
    model, tok, name = build_model_and_tokenizer(device)
    model.eval()
    tr = per_chunk_loss(model, train_chunks[:200], device).mean()
    te = per_chunk_loss(model, test_chunks[:200], device).mean()
    print(f'  [pretrained baseline] train={tr:.4f} test={te:.4f}')
    del model
    return {'train_loss': float(tr), 'test_loss': float(te)}


def train_v3(seed, mode, device, train_chunks, test_chunks):
    """Train one M or G run. Returns checkpoint info."""
    t.manual_seed(seed); np.random.seed(seed)
    lr = G_LR if mode == 'G' else M_LR
    wd = G_WD if mode == 'G' else M_WD

    model, tok, name = build_model_and_tokenizer(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, eps=1e-8)

    val_history = []
    best_val = float('inf')
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0

    print(f'  [{mode} seed={seed}] lr={lr} wd={wd} epochs={EPOCHS} '
          f'train_chunks={len(train_chunks)} test_chunks={len(test_chunks)}')

    for ep in range(EPOCHS):
        model.train()
        rng = np.random.RandomState(ep + seed * 1000)
        order = rng.permutation(len(train_chunks))
        nan_batches = 0
        for i in range(0, len(train_chunks), BATCH):
            batch = [train_chunks[j] for j in order[i:i + BATCH]]
            opt.zero_grad()
            loss = batch_loss(model, batch, device)
            if not t.isfinite(loss):
                nan_batches += 1; continue
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        tr_l = per_chunk_loss(model, train_chunks[:128], device).mean()
        te_l = per_chunk_loss(model, test_chunks[:128], device).mean()
        val_history.append({'epoch': ep + 1,
                             'train_loss': float(tr_l),
                             'test_loss':  float(te_l)})

        if te_l < best_val:
            best_val = float(te_l)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1

        if (ep + 1) % 3 == 0 or ep == 0:
            print(f'  [{mode} seed={seed}] ep={ep+1}: '
                  f'train={tr_l:.4f} test={te_l:.4f}  '
                  f'(best_test={best_val:.4f} at ep={best_epoch})  nan={nan_batches}')

    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {
        'model': model,
        'name': name,
        'final_state': final_state,
        'best_state': best_state,
        'best_epoch': best_epoch,
        'best_val': best_val,
        'val_history': val_history,
    }


def signature_at_checkpoint(model, state, train_chunks, test_chunks, device):
    model.load_state_dict(state)
    model.eval()
    tr_b = train_chunks[:8]; te_b = test_chunks[:8]
    train_loss_fn = lambda: batch_loss(model, tr_b, device)
    test_loss_fn  = lambda: batch_loss(model, te_b, device)
    tr_losses = per_chunk_loss(model, train_chunks, device)
    te_losses = per_chunk_loss(model, test_chunks,  device)
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=5, verbose=True)
    bat['mean_train_loss'] = float(tr_losses.mean())
    bat['mean_test_loss']  = float(te_losses.mean())
    bat['gap_loss']        = float(te_losses.mean() - tr_losses.mean())
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'tier6_v3_proper_mg.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load corpus once
    text = get_multi_book_corpus()
    print(f'Corpus: {len(text):,} characters')

    results = {'baseline': None, 'M': [], 'G': []}

    # Measure pretrained baseline on seed-0 data split
    # (use same split as seed 0 for baseline measurement)
    _, tok_tmp, _ = build_model_and_tokenizer(device)
    train_chunks_s0, test_chunks_s0 = build_chunks_safe(
        text, tok_tmp, CHUNK_LEN, N_TRAIN, N_TEST, seed=0)
    del tok_tmp
    print('\n=== Pretrained baseline ===')
    results['baseline'] = measure_baseline(device, train_chunks_s0, test_chunks_s0)

    for seed in range(NUM_SEEDS):
        # Rebuild tokenizer for chunk splitting (same seed as training)
        _, tok_seed, _ = build_model_and_tokenizer(device)
        train_chunks, test_chunks = build_chunks_safe(
            text, tok_seed, CHUNK_LEN, N_TRAIN, N_TEST, seed=seed)
        del tok_seed

        for mode in ['M', 'G']:
            print(f'\n=== {mode} seed={seed} ===')
            try:
                run = train_v3(seed, mode, device, train_chunks, test_chunks)
                model = run['model']

                state         = run['final_state'] if mode == 'M' else run['best_state']
                chk_epoch     = EPOCHS             if mode == 'M' else run['best_epoch']

                print(f'  computing signatures at checkpoint epoch={chk_epoch}...')
                bat = signature_at_checkpoint(model, state, train_chunks, test_chunks, device)
                bat['mode']             = mode
                bat['seed']             = seed
                bat['checkpoint_epoch'] = chk_epoch
                bat['val_history']      = run['val_history']
                bat['model_name']       = run['name']
                bat['lr']               = G_LR if mode == 'G' else M_LR
                bat['wd']               = G_WD if mode == 'G' else M_WD
                bat['pretrained_baseline_test'] = results['baseline']['test_loss']
                bat['improved_from_baseline']   = (
                    bat['mean_test_loss'] < results['baseline']['test_loss']
                )

                # Trim large fields
                if 'hessian_eigs_full' in bat:
                    bat['hessian_eigs_full'] = bat['hessian_eigs_full'][:20]
                if 'ranks' in bat:
                    bat['ranks'] = {
                        k: v for k, v in bat['ranks'].items()
                        if any(s in k.lower() for s in
                               ['mlp', 'embed', 'attention', 'dense', 'query', 'key', 'value'])
                    }

                results[mode].append(bat)
                print(f'  {mode} seed={seed}: '
                      f'gap={bat["gap_loss"]:.4f}  '
                      f'mia={bat.get("mia_loss_auc", 0):.4f}  '
                      f'top={bat["hessian_top_full"]:.1f}  '
                      f'improved_from_baseline={bat["improved_from_baseline"]}  '
                      f'(chk ep={chk_epoch})')

            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback; traceback.print_exc()
                results[mode].append({'mode': mode, 'seed': seed, 'error': str(e)})

            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

    # Summary
    print('\n=== tier6_v3 summary ===')
    baseline_test = results['baseline']['test_loss']
    print(f'  pretrained baseline test_loss = {baseline_test:.4f}')
    for mode in ('M', 'G'):
        valid = [r for r in results[mode] if 'error' not in r]
        if not valid:
            print(f'  {mode}: NO VALID RESULTS'); continue
        gaps = [r['gap_loss'] for r in valid]
        mias = [r.get('mia_loss_auc', float('nan')) for r in valid]
        tests = [r['mean_test_loss'] for r in valid]
        improved = [r.get('improved_from_baseline', False) for r in valid]
        print(f'  {mode}: mean_gap={np.mean(gaps):.3f}  '
              f'mean_test={np.mean(tests):.3f}  '
              f'MIA={np.mean(mias):.4f}  '
              f'improved_from_baseline={improved}')
    print(f'\nExpected:')
    print(f'  G: improved_from_baseline=True, mean_test < {baseline_test:.3f}, MIA ~ 0.65-0.80')
    print(f'  M: improved_from_baseline=False, mean_test >> {baseline_test:.3f}, MIA ~ 1.00')


if __name__ == '__main__':
    main()
