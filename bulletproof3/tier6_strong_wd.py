"""tier6 strong-WD variant: does cranking up WD escape the regime collapse?

The original tier6 (WD=0 for M, WD=0.1 for G) showed BOTH regimes memorizing
200 train chunks of Pride and Prejudice (MIA=1.0 for all 4 runs).

This variant sweeps WD ∈ {0.5, 1.0, 5.0} to test whether stronger regularization
produces a generalizing G that doesn't memorize. Each WD value gets 2 seeds.

If WD=1.0 or higher generalizes (lower MIA, smaller train/test gap) — we have a
real G comparison point for tier6 and can include it as a proper LM tier.

If all WD values still produce memorization — the collapse is robust, and we can
honestly report "no level of standard fine-tuning WD prevents memorization at
this fine-tune data scale" as the LM tier finding.
"""
import json
import urllib.request
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
    batch_loss, per_chunk_loss, N_TRAIN, N_TEST, CHUNK_LEN, BATCH
)

NUM_SEEDS = 2
EPOCHS = 50
WDS = [0.5, 1.0, 5.0]


def train_with_wd(seed, wd, device):
    t.manual_seed(seed); np.random.seed(seed)
    model, tok, name = build_model_and_tokenizer(device)
    text = get_corpus()
    train_chunks, test_chunks = build_chunks(text, tok, CHUNK_LEN, N_TRAIN, N_TEST, seed=seed)
    LR = 1e-5
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, eps=1e-8)
    print(f'  starting fine-tune: wd={wd}, epochs={EPOCHS}, lr={LR}')
    for ep in range(EPOCHS):
        model.train()
        rng = np.random.RandomState(ep + seed * 1000)
        order = rng.permutation(len(train_chunks))
        n_batches = 0; nan_batches = 0
        for i in range(0, len(train_chunks), BATCH):
            batch = [train_chunks[j] for j in order[i:i + BATCH]]
            opt.zero_grad()
            loss = batch_loss(model, batch, device)
            if not t.isfinite(loss):
                nan_batches += 1; continue
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            n_batches += 1
        if (ep + 1) % 5 == 0:
            model.eval()
            tr_l = per_chunk_loss(model, train_chunks[:64], device).mean()
            te_l = per_chunk_loss(model, test_chunks[:64], device).mean()
            print(f'  ep={ep+1}: train_loss={tr_l:.4f}, test_loss={te_l:.4f}, '
                  f'gap={te_l - tr_l:.4f} (nan={nan_batches}/{n_batches + nan_batches})')
            if not np.isfinite(tr_l):
                raise RuntimeError(f'NaN at ep={ep+1}, wd={wd}')
    return model, tok, train_chunks, test_chunks, name


def run(seed, wd, device):
    model, tok, train_chunks, test_chunks, name = train_with_wd(seed, wd, device)
    model.eval()
    tr_b = train_chunks[:8]; te_b = test_chunks[:8]
    train_loss_fn = lambda: batch_loss(model, tr_b, device)
    test_loss_fn  = lambda: batch_loss(model, te_b, device)
    tr_losses = per_chunk_loss(model, train_chunks, device)
    te_losses = per_chunk_loss(model, test_chunks, device)
    print(f'  FINAL train_loss={tr_losses.mean():.4f}, test_loss={te_losses.mean():.4f}, '
          f'gap={te_losses.mean() - tr_losses.mean():.4f}')
    print('  computing signatures...')
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=5, verbose=True)
    bat['seed'] = seed; bat['wd'] = wd; bat['model_name'] = name
    bat['mean_train_loss'] = float(tr_losses.mean())
    bat['mean_test_loss'] = float(te_losses.mean())
    bat['gap_loss'] = float(te_losses.mean() - tr_losses.mean())
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {}
    out_path = HERE / 'results' / 'tier6_strong_wd.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for wd in WDS:
        key = f'wd{wd}'
        results[key] = []
        for seed in range(NUM_SEEDS):
            print(f'\n=== wd={wd} seed={seed} ===')
            try:
                entry = run(seed, wd, device)
                # cull big lists
                if 'hessian_eigs_full' in entry:
                    entry['hessian_eigs_full'] = entry['hessian_eigs_full'][:20]
                if 'ranks' in entry:
                    entry['ranks'] = {k: v for k, v in entry['ranks'].items()
                                       if any(s in k.lower() for s in
                                               ['mlp', 'embed', 'fc', 'attention', 'dense', 'query', 'key', 'value'])}
                results[key].append(entry)
                print(f'  wd={wd} seed={seed}: gap_loss={entry["gap_loss"]:.4f} '
                      f'mia={entry.get("mia_loss_auc", 0):.4f} '
                      f'top={entry["hessian_top_full"]:.3f} '
                      f'bot={entry["hessian_bot_full"]:.3f}')
            except Exception as e:
                print(f'  error: {e}')
                import traceback; traceback.print_exc()
                results[key].append({'wd': wd, 'seed': seed, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
    # Compact summary at the end
    print('\n=== WD sweep summary ===')
    for wd in WDS:
        key = f'wd{wd}'
        cells = [r for r in results[key] if 'error' not in r]
        if not cells: continue
        gap_mean = np.mean([r['gap_loss'] for r in cells])
        mia_mean = np.mean([r.get('mia_loss_auc', float('nan')) for r in cells])
        print(f'  wd={wd}: mean gap={gap_mean:.3f}, mean MIA={mia_mean:.4f}')


if __name__ == '__main__':
    main()
