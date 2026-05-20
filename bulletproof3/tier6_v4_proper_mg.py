"""tier6_v4: a PROPER generalizing regime for Pythia-160m fine-tuning.

WHY v1/v2/v3 NEVER PRODUCED A REAL G
------------------------------------
All three earlier attempts fine-tuned Pythia-160m on a TINY corpus
(200-2000 chunks = 0.05-0.5M tokens) for MANY epochs (30-50). With a 160M
parameter model seeing a half-million tokens 30+ times, memorization is
forced no matter what weight decay / learning rate you pick. The "G" they
recovered was always the best-val checkpoint at epoch ~1 -- i.e. the
*pretrained model with one gradient step*. That is not a generalizing model;
it is an under-fit model. Comparing M (a collapsed, memorized model) against
that is not an M-vs-G comparison at all.

THE ACTUAL LEVER: DATA DIVERSITY AT FIXED COMPUTE
-------------------------------------------------
Memorization vs generalization is governed by how many times the optimizer
sees each training example, i.e. (compute budget) / (dataset size). We hold
the compute budget FIXED (same number of chunk-visits for every run) and
sweep the training-set size N. Small N => each chunk seen hundreds of times
=> memorization. Large N => each chunk seen 2-3 times => the model cannot
memorize and is forced to learn generalizable structure => genuine
generalization with a small train/test gap.

  N (train chunks)   epochs   visits/chunk   expected regime
  ----------------   ------   ------------   ---------------
       400            192        192         M  (memorize hard)
      1600             48         48         M->transition
      6400             12         12         transition->G
     25600              3          3         G  (genuine generalizer)

Every run uses the SAME budget (~76,800 chunk-visits), the SAME fixed
weight decay, the SAME learning rate, and the SAME final-checkpoint rule.
The ONLY thing that varies is dataset size. No early-stopping tricks.

WHAT A GENUINE G LOOKS LIKE (the large-N endpoint)
--------------------------------------------------
  - train loss and held-out loss both low and CLOSE (gap < ~0.5 nat)
  - held-out loss <= pretrained baseline (the model actually learned)
  - MIA AUC near chance (~0.5-0.6): train and test chunks indistinguishable
A genuine M (the small-N endpoint):
  - train loss -> ~0, held-out loss diverges far ABOVE baseline
  - MIA AUC -> 1.0

The held-out TEST set is reserved ONCE, up front, and is disjoint from every
training pool, so all runs are scored on identical data.
"""
import json
import urllib.request
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

DATA_DIR = HERE.parent / 'data'

# ---- experiment design -----------------------------------------------------
NUM_SEEDS   = 2
BATCH       = 4
CHUNK_LEN   = 256
N_TEST      = 1000                       # fixed held-out set, reserved up front
TRAIN_SIZES = [400, 1600, 6400, 25600]   # the data-diversity sweep
BUDGET      = 76800                      # chunk-visits per run (held constant)
LR          = 5e-5                       # fixed across all runs
WD          = 0.01                       # fixed across all runs
GAP_G_MAX   = 0.5                        # gap below this => "generalizing"
GAP_M_MIN   = 2.0                        # gap above this => "memorizing"

# ~35 long public-domain novels; together well over 8M tokens.
GUTENBERG_IDS = [
    1342, 161, 158, 1260, 768, 98, 1400, 84, 2701, 1661,
    74, 76, 11, 1399, 345, 174, 2814, 219, 120, 36,
    35, 1184, 135, 16, 43, 730, 766, 1023, 580, 215,
    514, 113, 2600, 6130, 1727,
]


def _try_download(gid, dest):
    """Gutenberg serves text under several URL schemes; try them in order."""
    urls = [
        f'https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt',
        f'https://www.gutenberg.org/files/{gid}/{gid}-0.txt',
        f'https://www.gutenberg.org/files/{gid}/{gid}.txt',
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            return True
        except Exception:
            continue
    return False


def get_big_corpus():
    """Download ~35 Gutenberg novels, strip headers, concatenate."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pieces = []
    for gid in GUTENBERG_IDS:
        p = DATA_DIR / f'gutenberg_{gid}.txt'
        if not p.exists():
            print(f'  downloading Gutenberg #{gid}')
            if not _try_download(gid, p):
                print(f'    FAILED #{gid}; skipping')
                continue
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        s = text.find('*** START OF THE PROJECT GUTENBERG')
        e = text.find('*** END OF THE PROJECT GUTENBERG')
        if s != -1:
            text = text[text.find('\n', s) + 1:]
        if e != -1 and text.find('\n', e) > 0:
            text = text[:text.find('\n', e) - 1]
        text = text.strip()
        if len(text) > 5000:
            pieces.append(text)
    if not pieces:
        raise RuntimeError('No corpus available; all Gutenberg downloads failed.')
    print(f'  corpus: {len(pieces)} books')
    return '\n\n'.join(pieces)


def build_chunk_pool(text, tokenizer):
    """Tokenize the whole corpus, chunk it, reserve a fixed disjoint test set.

    Returns (test_chunks, train_pool). The split uses a FIXED seed so the
    held-out set is identical for every run and every training seed.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    n_chunks = len(ids) // CHUNK_LEN
    chunks = [ids[i * CHUNK_LEN:(i + 1) * CHUNK_LEN] for i in range(n_chunks)]
    print(f'  corpus: {len(ids):,} tokens -> {n_chunks:,} chunks of {CHUNK_LEN}')
    np.random.RandomState(12345).shuffle(chunks)        # fixed test reservation
    test_chunks = chunks[:N_TEST]
    train_pool  = chunks[N_TEST:]
    need = max(TRAIN_SIZES)
    if len(train_pool) < need:
        raise RuntimeError(
            f'Train pool has only {len(train_pool)} chunks but largest sweep '
            f'point needs {need}. Add more books to GUTENBERG_IDS.')
    print(f'  reserved {len(test_chunks)} test chunks; train pool {len(train_pool)}')
    return test_chunks, train_pool


def measure_baseline(device, test_chunks):
    model, tok, name = build_model_and_tokenizer(device)
    model.eval()
    te = float(per_chunk_loss(model, test_chunks, device).mean())
    print(f'  [pretrained baseline] test_loss={te:.4f}')
    del model
    if device.type == 'cuda':
        t.cuda.empty_cache()
    return te


def train_one(seed, n_train, train_pool, test_chunks, device):
    """Train one run at dataset size n_train with the fixed compute budget."""
    t.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(train_pool), size=n_train, replace=False)
    train_chunks = [train_pool[i] for i in idx]

    epochs = max(1, round(BUDGET / n_train))
    model, tok, name = build_model_and_tokenizer(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, eps=1e-8)
    print(f'  [N={n_train} seed={seed}] epochs={epochs} '
          f'visits/chunk={epochs} lr={LR} wd={WD}')

    val_history = []
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
        # eval cadence: at least start/end, and ~10 points in between
        if ep == 0 or ep == epochs - 1 or (ep + 1) % max(1, epochs // 10) == 0:
            model.eval()
            tr_l = float(per_chunk_loss(model, train_chunks[:128], device).mean())
            te_l = float(per_chunk_loss(model, test_chunks[:128], device).mean())
            val_history.append({'epoch': ep + 1, 'train_loss': tr_l, 'test_loss': te_l})
            print(f'    ep={ep+1}/{epochs} train={tr_l:.4f} test={te_l:.4f} '
                  f'gap={te_l - tr_l:.4f} nan={nan_batches}')
    return model, train_chunks, epochs, val_history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'tier6_v4_proper_mg.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = get_big_corpus()
    _, tok0, _ = build_model_and_tokenizer(device)
    test_chunks, train_pool = build_chunk_pool(text, tok0)
    del tok0
    if device.type == 'cuda':
        t.cuda.empty_cache()

    print('\n=== pretrained baseline ===')
    baseline_test = measure_baseline(device, test_chunks)

    results = {
        'design': {
            'train_sizes': TRAIN_SIZES, 'budget_chunk_visits': BUDGET,
            'lr': LR, 'wd': WD, 'chunk_len': CHUNK_LEN, 'n_test': N_TEST,
            'num_seeds': NUM_SEEDS,
            'note': 'Fixed compute budget; only dataset size varies. '
                    'Final-checkpoint rule for every run; no early stopping.',
        },
        'baseline': {'test_loss': baseline_test},
        'runs': [],
    }

    for n_train in TRAIN_SIZES:
        for seed in range(NUM_SEEDS):
            print(f'\n=== N={n_train} seed={seed} ===')
            try:
                model, train_chunks, epochs, val_history = train_one(
                    seed, n_train, train_pool, test_chunks, device)
                model.eval()

                tr_b = train_chunks[:8]; te_b = test_chunks[:8]
                train_loss_fn = lambda: batch_loss(model, tr_b, device)
                test_loss_fn  = lambda: batch_loss(model, te_b, device)
                tr_losses = per_chunk_loss(model, train_chunks, device)
                te_losses = per_chunk_loss(model, test_chunks, device)

                print('  computing signature battery...')
                bat = compute_full_battery(
                    model, train_loss_fn, test_loss_fn,
                    tr_losses, te_losses, lanczos_k=5, verbose=True)

                bat['mean_train_loss'] = float(tr_losses.mean())
                bat['mean_test_loss']  = float(te_losses.mean())
                bat['gap_loss']        = float(te_losses.mean() - tr_losses.mean())
                bat['n_train']         = n_train
                bat['seed']            = seed
                bat['epochs']          = epochs
                bat['visits_per_chunk'] = epochs
                bat['lr']              = LR
                bat['wd']              = WD
                bat['val_history']     = val_history
                bat['pretrained_baseline_test'] = baseline_test
                bat['improved_from_baseline']   = bat['mean_test_loss'] < baseline_test
                gap = bat['gap_loss']
                bat['regime'] = ('G' if gap < GAP_G_MAX else
                                 'M' if gap > GAP_M_MIN else 'transition')

                if 'hessian_eigs_full' in bat:
                    bat['hessian_eigs_full'] = bat['hessian_eigs_full'][:20]
                if 'ranks' in bat:
                    bat['ranks'] = {
                        k: v for k, v in bat['ranks'].items()
                        if any(s in k.lower() for s in
                               ['mlp', 'embed', 'attention', 'dense',
                                'query', 'key', 'value'])}

                results['runs'].append(bat)
                print(f'  N={n_train} seed={seed}: regime={bat["regime"]} '
                      f'train={bat["mean_train_loss"]:.3f} '
                      f'test={bat["mean_test_loss"]:.3f} gap={gap:.3f} '
                      f'mia={bat.get("mia_loss_auc", 0):.4f} '
                      f'improved={bat["improved_from_baseline"]}')

                del model
                if device.type == 'cuda':
                    t.cuda.empty_cache()
            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback; traceback.print_exc()
                results['runs'].append(
                    {'n_train': n_train, 'seed': seed, 'error': str(e)})

            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

    # ---- summary -----------------------------------------------------------
    print('\n=== tier6_v4 summary ===')
    print(f'  pretrained baseline test_loss = {baseline_test:.4f}')
    print(f'  {"N":>7} {"epochs":>7} {"train":>8} {"test":>8} '
          f'{"gap":>7} {"MIA":>7} {"regime":>11}')
    for n_train in TRAIN_SIZES:
        rs = [r for r in results['runs']
              if r.get('n_train') == n_train and 'error' not in r]
        if not rs:
            print(f'  {n_train:>7}  (no valid runs)'); continue
        m = lambda k: np.mean([r[k] for r in rs])
        print(f'  {n_train:>7} {rs[0]["epochs"]:>7} '
              f'{m("mean_train_loss"):>8.3f} {m("mean_test_loss"):>8.3f} '
              f'{m("gap_loss"):>7.3f} {m("mia_loss_auc"):>7.3f} '
              f'{rs[-1]["regime"]:>11}')
    g_runs = [r for r in results['runs']
              if 'error' not in r and r.get('regime') == 'G']
    if g_runs:
        print(f'\n  GENUINE G FOUND: {len(g_runs)} run(s) with gap < {GAP_G_MAX} '
              f'(small train/test gap = generalization).')
    else:
        print(f'\n  NO GENUINE G: every run has gap >= {GAP_G_MAX}. '
              f'If even the largest N memorizes, raise max(TRAIN_SIZES) or '
              f'lower BUDGET so visits/chunk drops further.')


if __name__ == '__main__':
    main()
