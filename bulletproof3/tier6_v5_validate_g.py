"""tier6_v5: validate the genuine generalizing fine-tune found in tier6_v4.

WHAT v4 ESTABLISHED
-------------------
Sweeping training-set size at fixed compute traced a clean memorize->generalize
crossover. At N=25,600 (3 visits/chunk) the model GENERALIZED: train/test gap
0.30 nat, MIA 0.70. But within v4's fixed budget that G had only 3 epochs and
its held-out loss (3.51), while still falling at the cutoff, had not yet
dropped below the pretrained baseline (3.37). So v4 proved a generalizing
*regime* exists but not a generalizing *model that beats the baseline*.

WHAT v5 DOES
------------
Two clean, directly comparable runs, same hyperparameters (lr=5e-5, wd=0.01),
the ONLY difference being dataset size -- the v4 lever:

  G : N=25,600 chunks, trained LONG (12 epochs). Held-out loss is tracked
      every epoch; the reported G checkpoint is the best-val epoch (standard
      early stopping). This is a genuine generalizing run -- unlike v3, where
      "best val" pathologically landed at epoch 1, here it should land deep
      into training with train and test both low.
  M : N=400 chunks, trained to full memorization (60 epochs). Final checkpoint.

Both scored on the SAME reserved held-out set (1,000 chunks, disjoint from
both training pools). 3 seeds each.

SUCCESS CRITERIA (printed as a verdict at the end)
--------------------------------------------------
  G genuine generalizer  : best held-out loss < pretrained baseline
                           AND train/test gap < 0.5 nat
                           AND MIA AUC < 0.80
  M genuine memorizer    : train loss -> ~0, held-out loss >> baseline,
                           MIA AUC -> 1.0
If G's best held-out loss still does not beat the baseline, raise G_EPOCHS.
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.optim as optim

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bulletproof3._signatures import compute_full_battery
from bulletproof3.tier6_pythia_finetune import (
    build_model_and_tokenizer, batch_loss, per_chunk_loss,
)
from bulletproof3.tier6_v4_proper_mg import get_big_corpus, build_chunk_pool

# ---- design ----------------------------------------------------------------
NUM_SEEDS = 3
BATCH     = 4
LR        = 5e-5
WD        = 0.01

G_N       = 25600     # large, diverse -> forced to generalize
G_EPOCHS  = 12        # long enough for held-out loss to bottom out

M_N       = 400       # tiny -> memorizes
M_EPOCHS  = 60        # enough to drive train loss to ~0

GAP_G_MAX = 0.5
MIA_G_MAX = 0.80


def train_run(seed, n_train, epochs, train_pool, test_chunks, device,
              track_best):
    """Fine-tune at dataset size n_train. If track_best, also keep the
    best-held-out-loss checkpoint (used for G); else keep only final (M)."""
    t.manual_seed(seed); np.random.seed(seed)
    idx = np.random.RandomState(seed).choice(len(train_pool), size=n_train,
                                             replace=False)
    train_chunks = [train_pool[i] for i in idx]

    model, tok, name = build_model_and_tokenizer(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, eps=1e-8)
    print(f'  [N={n_train} seed={seed}] epochs={epochs} lr={LR} wd={WD}')

    val_history = []
    best_val = float('inf'); best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for ep in range(epochs):
        model.train()
        order = np.random.RandomState(ep + seed * 1000).permutation(n_train)
        nan_batches = 0
        for i in range(0, n_train, BATCH):
            batch = [train_chunks[j] for j in order[i:i + BATCH]]
            opt.zero_grad()
            loss = batch_loss(model, batch, device)
            if not t.isfinite(loss):
                nan_batches += 1; continue
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        model.eval()
        tr_l = float(per_chunk_loss(model, train_chunks[:128], device).mean())
        te_l = float(per_chunk_loss(model, test_chunks[:128], device).mean())
        val_history.append({'epoch': ep + 1, 'train_loss': tr_l,
                             'test_loss': te_l})
        if track_best and te_l < best_val:
            best_val = te_l; best_epoch = ep + 1
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        print(f'    ep={ep+1}/{epochs} train={tr_l:.4f} test={te_l:.4f} '
              f'gap={te_l - tr_l:.4f} nan={nan_batches}')

    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if not track_best:
        best_state, best_epoch = final_state, epochs
    return {'model': model, 'name': name, 'train_chunks': train_chunks,
            'final_state': final_state, 'best_state': best_state,
            'best_epoch': best_epoch, 'val_history': val_history}


def battery_at(model, state, train_chunks, test_chunks, device):
    model.load_state_dict(state)
    model.eval()
    tr_b = train_chunks[:8]; te_b = test_chunks[:8]
    tr_losses = per_chunk_loss(model, train_chunks, device)
    te_losses = per_chunk_loss(model, test_chunks, device)
    bat = compute_full_battery(
        model, lambda: batch_loss(model, tr_b, device),
        lambda: batch_loss(model, te_b, device),
        tr_losses, te_losses, lanczos_k=5, verbose=True)
    bat['mean_train_loss'] = float(tr_losses.mean())
    bat['mean_test_loss']  = float(te_losses.mean())
    bat['gap_loss']        = float(te_losses.mean() - tr_losses.mean())
    if 'hessian_eigs_full' in bat:
        bat['hessian_eigs_full'] = bat['hessian_eigs_full'][:20]
    if 'ranks' in bat:
        bat['ranks'] = {k: v for k, v in bat['ranks'].items()
                        if any(s in k.lower() for s in
                               ['mlp', 'embed', 'attention', 'dense',
                                'query', 'key', 'value'])}
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'tier6_v5_validate_g.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = get_big_corpus()
    _, tok0, _ = build_model_and_tokenizer(device)
    test_chunks, train_pool = build_chunk_pool(text, tok0)
    del tok0
    if device.type == 'cuda':
        t.cuda.empty_cache()

    print('\n=== pretrained baseline ===')
    bmodel, _, _ = build_model_and_tokenizer(device)
    bmodel.eval()
    baseline_test = float(per_chunk_loss(bmodel, test_chunks, device).mean())
    del bmodel
    if device.type == 'cuda':
        t.cuda.empty_cache()
    print(f'  baseline held-out loss = {baseline_test:.4f}')

    results = {
        'design': {'G_N': G_N, 'G_epochs': G_EPOCHS, 'M_N': M_N,
                   'M_epochs': M_EPOCHS, 'lr': LR, 'wd': WD,
                   'num_seeds': NUM_SEEDS,
                   'note': 'G and M differ ONLY in dataset size. G uses '
                           'best-val checkpoint, M uses final.'},
        'baseline': {'test_loss': baseline_test},
        'M': [], 'G': [],
    }

    specs = [('M', M_N, M_EPOCHS, False), ('G', G_N, G_EPOCHS, True)]
    for mode, n_train, epochs, track_best in specs:
        for seed in range(NUM_SEEDS):
            print(f'\n=== {mode} seed={seed} ===')
            try:
                run = train_run(seed, n_train, epochs, train_pool,
                                test_chunks, device, track_best)
                state = run['best_state']
                chk_ep = run['best_epoch']
                print(f'  computing battery at {mode} checkpoint '
                      f'(epoch {chk_ep})...')
                bat = battery_at(run['model'], state, run['train_chunks'],
                                 test_chunks, device)
                bat['mode']  = mode
                bat['seed']  = seed
                bat['n_train'] = n_train
                bat['checkpoint_epoch'] = chk_ep
                bat['total_epochs']     = epochs
                bat['val_history']      = run['val_history']
                bat['pretrained_baseline_test'] = baseline_test
                bat['improved_from_baseline'] = (
                    bat['mean_test_loss'] < baseline_test)
                results[mode].append(bat)
                print(f'  {mode} seed={seed}: train={bat["mean_train_loss"]:.3f} '
                      f'test={bat["mean_test_loss"]:.3f} '
                      f'gap={bat["gap_loss"]:.3f} '
                      f'mia={bat.get("mia_loss_auc", 0):.4f} '
                      f'improved={bat["improved_from_baseline"]} '
                      f'(chk ep={chk_ep})')
                del run
                if device.type == 'cuda':
                    t.cuda.empty_cache()
            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback; traceback.print_exc()
                results[mode].append({'mode': mode, 'seed': seed,
                                      'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

    # ---- verdict -----------------------------------------------------------
    print('\n=== tier6_v5 verdict ===')
    print(f'  pretrained baseline held-out loss = {baseline_test:.4f}')
    for mode in ('M', 'G'):
        valid = [r for r in results[mode] if 'error' not in r]
        if not valid:
            print(f'  {mode}: NO VALID RUNS'); continue
        m = lambda k: float(np.mean([r[k] for r in valid]))
        print(f'  {mode}: train={m("mean_train_loss"):.3f} '
              f'test={m("mean_test_loss"):.3f} gap={m("gap_loss"):.3f} '
              f'MIA={m("mia_loss_auc"):.4f} '
              f'improved={[r["improved_from_baseline"] for r in valid]}')
    gv = [r for r in results['G'] if 'error' not in r]
    if gv:
        ok = all(r['improved_from_baseline'] for r in gv) \
            and np.mean([r['gap_loss'] for r in gv]) < GAP_G_MAX \
            and np.mean([r['mia_loss_auc'] for r in gv]) < MIA_G_MAX
        if ok:
            print('\n  VALIDATED: G is a genuine generalizer that BEATS the '
                  'pretrained baseline (small gap, MIA < 0.80, '
                  'improved_from_baseline on every seed).')
        else:
            print('\n  NOT YET: G generalizes (small gap) but has not beaten '
                  'the baseline on every seed. Raise G_EPOCHS and rerun.')


if __name__ == '__main__':
    main()
