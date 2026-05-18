"""Long-training language model: do signatures appear with sufficient overfitting?

Our short Shakespeare LM (2k iters) didn't show the signatures. Two
hypotheses:
  H1: LMs are genuinely different from classification (no saddle structure)
  H2: We didn't train long enough for memorization to bite

This experiment tests H2 by:
  - Using a TINY dataset (just 30k chars of Shakespeare) so memorization is feasible
  - Using a deeper LM (4 layers) so it has capacity to memorize
  - Training for 50k iters (vs 2k previously)
  - Multi-seed (3) for statistical confidence

If signatures STILL don't appear after this, H1 is supported and our claim
is restricted to "classification."

If signatures DO appear at extreme overfitting, H2 is supported and our
claim extends to LMs (with the scope note "given sufficient overfitting").

Either result is publication-relevant.

Usage:
    python overnight/deep_long_lm.py
"""
import sys
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def effective_rank(W):
    s = torch.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def get_deep_weights(model):
    matrices = {}
    for name, p in model.named_parameters():
        if p.ndim >= 2 and 'norm' not in name.lower() and 'embed' not in name.lower():
            W = p.detach().reshape(p.shape[0], -1)
            matrices[name] = W
    return matrices


class DeepCharLM(nn.Module):
    def __init__(self, vocab=128, dim=128, depth=4, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, activation='gelu')
        self.mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, vocab)
        self.depth = depth

    def forward(self, x):
        e = self.embed(x) + self.pos[:, :x.size(1)]
        mask = self.mask[:x.size(1), :x.size(1)].to(x.device)
        h = self.enc(e, mask=mask, is_causal=True)
        return self.head(h)


def load_tiny_shakespeare():
    """Use just first 30000 characters — small enough to memorize, large enough
    to have real text structure."""
    p = Path(__file__).resolve().parent.parent / 'diverse' / 'shakespeare.txt'
    text = p.read_text(encoding='utf-8')[:30000]
    return text


def run_one(seed, wd, n_iters, device):
    print(f'\n--- seed={seed}, wd={wd}, iters={n_iters} ---')
    text = load_tiny_shakespeare()
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    seq_len = 64
    split = int(0.85 * len(data))
    train_data = data[:split]
    test_data  = data[split:]

    def get_batch(d, bs=48):
        idx = torch.randint(0, len(d) - seq_len - 1, (bs,))
        x = torch.stack([d[i:i + seq_len] for i in idx])
        y = torch.stack([d[i + 1:i + seq_len + 1] for i in idx])
        return x.to(device), y.to(device)

    torch.manual_seed(seed)
    model = DeepCharLM(vocab=len(vocab), seq_len=seq_len, depth=4).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd, betas=(0.9, 0.98))
    crit = nn.CrossEntropyLoss()

    history = {'iter': [], 'train_ppl': [], 'test_ppl': []}
    t0 = time.time()
    for i in range(n_iters):
        x, y = get_batch(train_data, bs=48)
        logits = model(x)
        loss = crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 2500 == 0:
            model.eval()
            with torch.no_grad():
                # Sample multiple batches for stable estimate
                tr_losses, te_losses = [], []
                for _ in range(20):
                    x, y = get_batch(train_data, bs=48)
                    l = crit(model(x).reshape(-1, len(vocab)), y.reshape(-1))
                    tr_losses.append(l.item())
                    x, y = get_batch(test_data, bs=48)
                    l = crit(model(x).reshape(-1, len(vocab)), y.reshape(-1))
                    te_losses.append(l.item())
            model.train()
            tr_ppl = float(np.exp(np.mean(tr_losses)))
            te_ppl = float(np.exp(np.mean(te_losses)))
            history['iter'].append(i + 1)
            history['train_ppl'].append(tr_ppl)
            history['test_ppl'].append(te_ppl)
            print(f'    iter={i+1:6d}: train_ppl={tr_ppl:.3f}, test_ppl={te_ppl:.3f}, '
                  f'elapsed={time.time()-t0:.0f}s')

    # Final ranks
    ranks = {k: effective_rank(W) for k, W in get_deep_weights(model).items()}

    # Final gradient norm asymmetry
    def gn(d):
        model.train()
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        for _ in range(15):
            x, y = get_batch(d, bs=48)
            l = crit(model(x).reshape(-1, len(vocab)), y.reshape(-1))
            l.backward()
        total = sum((p.grad ** 2).sum().item() for p in model.parameters() if p.grad is not None)
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        return float(np.sqrt(total))

    g_tr = gn(train_data)
    g_te = gn(test_data)

    return {'history': history, 'final_ranks': ranks,
            'grad_train': g_tr, 'grad_test': g_te,
            'final_train_ppl': history['train_ppl'][-1],
            'final_test_ppl': history['test_ppl'][-1]}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    results = {}
    SEEDS = [0, 1, 2]
    for seed in SEEDS:
        for wd in [0.0, 0.1]:
            key = f'seed{seed}_wd{wd}'
            results[key] = run_one(seed, wd, n_iters=50000, device=device)

    out_json = HERE / 'results' / 'deep_long_lm.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Aggregate per WD across seeds
    print('\n=== Long LM summary (means across seeds) ===')
    for wd in [0.0, 0.1]:
        runs = [results[f'seed{s}_wd{wd}'] for s in SEEDS]
        tr_ppls = [r['final_train_ppl'] for r in runs]
        te_ppls = [r['final_test_ppl'] for r in runs]
        grad_ratios = [r['grad_test'] / max(r['grad_train'], 1e-9) for r in runs]
        # average rank of the deepest non-embed layer
        last_layer_ranks = []
        for r in runs:
            ranks_dict = r['final_ranks']
            keys = list(ranks_dict.keys())
            last_layer_ranks.append(ranks_dict[keys[-2]])    # second-to-last (head is last)
        print(f'\n  wd={wd}:')
        print(f'    train_ppl: {np.mean(tr_ppls):.3f} ± {np.std(tr_ppls):.3f}')
        print(f'    test_ppl:  {np.mean(te_ppls):.3f} ± {np.std(te_ppls):.3f}')
        print(f'    grad_test/train ratio: {np.mean(grad_ratios):.2e}')
        print(f'    last-block rank: {np.mean(last_layer_ranks):.2f} ± {np.std(last_layer_ranks):.2f}')


if __name__ == '__main__':
    main()
