"""v2 of diverse tasks: stronger training to actually exhibit memorization.

Improvements over v1:
  - Shakespeare LM: 20k iters (vs 2k) so M actually overfits
  - Tabular: WD = 1.0 for G (vs 0.01) — proper contrast
  - Both M and G saved as state dicts so we can re-analyze
  - Added rank reporting at multiple checkpoints during training

Usage:
    python diverse/diverse_tasks_v2.py
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
from torch.utils.data import DataLoader, TensorDataset


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


class SmallCharLM(nn.Module):
    def __init__(self, vocab=128, dim=128, depth=2, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, activation='gelu')
        self.mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        e = self.embed(x) + self.pos[:, :x.size(1)]
        mask = self.mask[:x.size(1), :x.size(1)].to(x.device)
        h = self.enc(e, mask=mask, is_causal=True)
        return self.head(h)


def run_shakespeare_v2(device):
    print(f'\n=== Shakespeare LM v2 (extended training to force memorization) ===')
    text = Path('./diverse/shakespeare.txt').read_text(encoding='utf-8')
    # USE A SHORT SLICE to force memorization
    text = text[:50000]   # only 50k chars — forces M to memorize them
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    seq_len = 64
    # holdout: last 20% never seen during training
    split = int(0.8 * len(data))
    train_data = data[:split]
    test_data = data[split:]

    def get_batch(d, bs):
        idx = torch.randint(0, len(d) - seq_len - 1, (bs,))
        x = torch.stack([d[i:i + seq_len] for i in idx])
        y = torch.stack([d[i + 1:i + seq_len + 1] for i in idx])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def perplexity(model, d, n=30):
        model.eval()
        crit = nn.CrossEntropyLoss()
        losses = []
        for _ in range(n):
            x, y = get_batch(d, bs=64)
            logits = model(x)
            losses.append(crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
        model.train()
        return float(np.exp(np.mean(losses)))

    def gn(model, d):
        model.train()
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        crit = nn.CrossEntropyLoss()
        for _ in range(10):
            x, y = get_batch(d, bs=64)
            logits = model(x)
            crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).backward()
        total = sum((p.grad ** 2).sum().item() for p in model.parameters() if p.grad is not None)
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        return float(np.sqrt(total))

    def train_eval(wd, n_iters=20000):
        torch.manual_seed(0)
        model = SmallCharLM(vocab=len(vocab), seq_len=seq_len).to(device)
        opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd, betas=(0.9, 0.98))
        crit = nn.CrossEntropyLoss()
        t0 = time.time()
        for i in range(n_iters):
            x, y = get_batch(train_data, bs=64)
            logits = model(x)
            loss = crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if (i + 1) % 2000 == 0:
                ppl_tr = perplexity(model, train_data)
                ppl_te = perplexity(model, test_data)
                print(f'    iter={i+1}: train_ppl={ppl_tr:.2f}, test_ppl={ppl_te:.2f}, '
                      f'elapsed={time.time()-t0:.0f}s')

        ppl_tr = perplexity(model, train_data)
        ppl_te = perplexity(model, test_data)
        ranks = {k: effective_rank(W) for k, W in get_deep_weights(model).items()}
        g_tr = gn(model, train_data)
        g_te = gn(model, test_data)
        return {'ppl_train': ppl_tr, 'ppl_test': ppl_te, 'ranks': ranks,
                'grad_train': g_tr, 'grad_test': g_te}

    print('  Training M (wd=0, 20k iters)...')
    M = train_eval(0.0, 20000)
    print('  Training G (wd=0.1, 20k iters)...')
    G = train_eval(0.1, 20000)
    print(f'\n  M: ppl_train={M["ppl_train"]:.2f}, ppl_test={M["ppl_test"]:.2f}, '
          f'grad_test/train={M["grad_test"]/max(M["grad_train"],1e-9):.2e}')
    print(f'  G: ppl_train={G["ppl_train"]:.2f}, ppl_test={G["ppl_test"]:.2f}, '
          f'grad_test/train={G["grad_test"]/max(G["grad_train"],1e-9):.2e}')
    return {'M': M, 'G': G}


def run_tabular_v2(device):
    print(f'\n=== Tabular v2 (stronger WD contrast: 0 vs 1.0) ===')
    torch.manual_seed(0)
    np.random.seed(0)
    n_features = 64
    n_classes = 10
    n_train = 5000
    n_test = 2000

    X_all = np.random.randn(n_train + n_test, n_features).astype(np.float32)
    proj = np.random.randn(n_features, n_classes).astype(np.float32)
    logits = X_all @ proj + 0.5 * np.random.randn(n_train + n_test, n_classes).astype(np.float32)
    y_all = logits.argmax(axis=1)

    X_train = torch.tensor(X_all[:n_train])
    y_train = torch.tensor(y_all[:n_train], dtype=torch.long)
    X_test  = torch.tensor(X_all[n_train:])
    y_test  = torch.tensor(y_all[n_train:], dtype=torch.long)

    train_ld = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_ld  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=512, shuffle=False)

    def make_mlp():
        return nn.Sequential(
            nn.Linear(n_features, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    def train_eval(wd, n_epochs=300):
        torch.manual_seed(0)
        model = make_mlp().to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
        crit = nn.CrossEntropyLoss()
        for ep in range(n_epochs):
            model.train()
            for x, y in train_ld:
                x = x.to(device); y = y.to(device)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
        model.eval()
        def accuracy(ld):
            c = t = 0
            with torch.no_grad():
                for x, y in ld:
                    x = x.to(device); y = y.to(device)
                    c += (model(x).argmax(-1) == y).sum().item()
                    t += y.numel()
            return c / t
        tr = accuracy(train_ld)
        te = accuracy(test_ld)
        ranks = {k: effective_rank(W) for k, W in get_deep_weights(model).items()}

        def gn(loader):
            model.train()
            for p in model.parameters():
                if p.grad is not None: p.grad.zero_()
            for i, (x, y) in enumerate(loader):
                if i >= 10: break
                x = x.to(device); y = y.to(device)
                crit(model(x), y).backward()
            total = sum((p.grad ** 2).sum().item() for p in model.parameters() if p.grad is not None)
            for p in model.parameters():
                if p.grad is not None: p.grad.zero_()
            return float(np.sqrt(total))

        return {'train_acc': tr, 'test_acc': te, 'ranks': ranks,
                'grad_train': gn(train_ld), 'grad_test': gn(test_ld)}

    print('  Training M (wd=0)...')
    M = train_eval(0.0, 300)
    print('  Training G (wd=1.0)...')
    G = train_eval(1.0, 300)
    print(f'\n  M: train={M["train_acc"]:.4f}, test={M["test_acc"]:.4f}, '
          f'grad_test/train={M["grad_test"]/max(M["grad_train"],1e-9):.2e}')
    print(f'  G: train={G["train_acc"]:.4f}, test={G["test_acc"]:.4f}, '
          f'grad_test/train={G["grad_test"]/max(G["grad_train"],1e-9):.2e}')
    return {'M': M, 'G': G}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    all_results = {}
    all_results['shakespeare_v2'] = run_shakespeare_v2(device)
    all_results['tabular_v2']     = run_tabular_v2(device)

    out_json = HERE / 'results' / 'diverse_tasks_v2.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nresults -> {out_json}')


if __name__ == '__main__':
    main()
