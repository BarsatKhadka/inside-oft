"""Tier 5: CharLM (4-layer Transformer) on Shakespeare with full signatures.

M: no WD, train hard (overfits) — train loss << val loss
G: WD=1e-3, dropout=0.1, generalizes
3 seeds each. 80k iters.
"""
import json
from pathlib import Path
import urllib.request
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bulletproof3._signatures import compute_full_battery, all_ranks

NUM_SEEDS = 3
N_ITERS = 80000
CTX = 128
BATCH = 64
DIM = 128
DEPTH = 4
HEADS = 4
DATA_DIR = HERE.parent / 'data'


class CharLM(nn.Module):
    def __init__(self, vocab, dim=DIM, depth=DEPTH, heads=HEADS, ctx=CTX, dropout=0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(ctx, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=dropout,
                                        activation='gelu', batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, vocab)
        self.ctx = ctx
        mask = t.triu(t.ones(ctx, ctx), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        T_ = x.size(1)
        h = self.tok(x) + self.pos(t.arange(T_, device=x.device))
        for blk in self.blocks: h = blk(h, src_mask=self.mask[:T_, :T_])
        return self.head(self.norm(h))


def get_shakespeare():
    p = DATA_DIR / 'shakespeare.txt'
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, p)
    text = open(p, 'r').read()
    chars = sorted(set(text)); ctoi = {c: i for i, c in enumerate(chars)}
    data = np.array([ctoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(data))
    return data[:split], data[split:], len(chars)


def get_batch(arr, ctx, batch, device):
    idx = np.random.randint(0, len(arr) - ctx - 1, batch)
    x = t.tensor(np.stack([arr[i:i+ctx] for i in idx]), device=device)
    y = t.tensor(np.stack([arr[i+1:i+ctx+1] for i in idx]), device=device)
    return x, y


def train_model(seed, mode, device, train, val, vocab):
    t.manual_seed(seed); np.random.seed(seed)
    wd = 1e-3 if mode == 'G' else 0.0
    drop = 0.1 if mode == 'G' else 0.0
    model = CharLM(vocab, dropout=drop).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
    for it in range(N_ITERS):
        x, y = get_batch(train, CTX, BATCH, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 10000 == 0:
            model.eval()
            with t.no_grad():
                x_v, y_v = get_batch(val, CTX, BATCH, device)
                l_v = F.cross_entropy(model(x_v).reshape(-1, vocab), y_v.reshape(-1)).item()
            print(f'  it={it+1}: train_loss={loss.item():.4f}, val_loss={l_v:.4f}')
            model.train()
    return model


def run(seed, mode, device):
    train, val, vocab = get_shakespeare()
    model = train_model(seed, mode, device, train, val, vocab)
    model.eval()
    # For signatures, get a fixed train batch and val batch
    t.manual_seed(seed * 31)
    x_tr, y_tr = get_batch(train, CTX, 32, device)
    x_te, y_te = get_batch(val, CTX, 32, device)
    train_loss_fn = lambda: F.cross_entropy(model(x_tr).reshape(-1, vocab), y_tr.reshape(-1))
    test_loss_fn  = lambda: F.cross_entropy(model(x_te).reshape(-1, vocab), y_te.reshape(-1))
    # MIA: per-sequence loss for many sampled sequences
    @t.no_grad()
    def per_seq_loss(arr, n_samples=512):
        losses = []
        for _ in range(n_samples // 32):
            x, y = get_batch(arr, CTX, 32, device)
            per_tok = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1), reduction='none')
            per_seq = per_tok.reshape(32, -1).mean(1).cpu().numpy()
            losses.append(per_seq)
        return np.concatenate(losses)
    tr_losses = per_seq_loss(train); te_losses = per_seq_loss(val)
    print('  computing signatures...')
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=10, verbose=True)
    bat['mode'] = mode; bat['seed'] = seed
    bat['mean_train_loss'] = float(tr_losses.mean())
    bat['mean_val_loss'] = float(te_losses.mean())
    bat['gap_loss'] = float(te_losses.mean() - tr_losses.mean())
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    out_path = HERE / 'results' / 'tier5_charlm_shakespeare.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for mode in ['M', 'G']:
        for seed in range(NUM_SEEDS):
            print(f'\n=== {mode} seed={seed} ===')
            try:
                entry = run(seed, mode, device)
                results[mode].append(entry)
                print(f'  val_loss={entry["mean_val_loss"]:.4f} '
                      f'gap={entry["gap_loss"]:.4f} '
                      f'top={entry["hessian_top_full"]:.3f} '
                      f'bot={entry["hessian_bot_full"]:.3f} '
                      f'cos={entry["cos_grad_train_test"]:.4f}')
            except Exception as e:
                print(f'  error: {e}')
                results[mode].append({'mode': mode, 'seed': seed, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
