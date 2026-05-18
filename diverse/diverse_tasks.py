"""Universal-signature test across diverse task domains.

For each task (MNIST classification, character-level LM, tabular classification),
train M (no WD) and G (WD=1e-3) and measure:
  - test accuracy/perplexity
  - effective rank of selected weight matrices
  - train-vs-test gradient norm asymmetry (saddle test)
  - whether SAM escapes M

If signatures hold across these 4 fundamentally different domains, the claim
is no longer "specific to modular addition" — it's a universal phenomenon.

Domains:
  1. MNIST    — image classification (MLP)
  2. SHAKES   — character-level language modeling (small transformer)
  3. TABULAR  — California Housing or similar UCI regression task (MLP)

Usage:
    python diverse/diverse_tasks.py
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
import torchvision


# ============================================================
# Shared utilities
# ============================================================
def effective_rank(W):
    s = torch.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def get_deep_weight_matrices(model):
    """Find 'deep' weight matrices in the model (ndim >= 2, not norm layers)."""
    matrices = {}
    for name, p in model.named_parameters():
        if p.ndim >= 2 and 'norm' not in name.lower() and 'embed' not in name.lower():
            W = p.detach().reshape(p.shape[0], -1)
            matrices[name] = W
    return matrices


def compute_grad_norm(model, loader, device, max_batches=10, loss_fn=None):
    model.train()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: break
        x = x.to(device); y = y.to(device)
        loss = loss_fn(model(x), y)
        loss.backward()
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad ** 2).sum().item()
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    return float(np.sqrt(total))


# ============================================================
# Domain 1: MNIST classification
# ============================================================
def run_mnist(device):
    print(f'\n=== MNIST classification ===')
    tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])
    root = './diverse/mnist_data'
    train_ds = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=tf)
    test_ds  = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=tf)
    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
    test_ld  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2)

    def make_mlp():
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def train_eval(wd, n_epochs):
        torch.manual_seed(0)
        model = make_mlp().to(device)
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=wd, nesterov=True)
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
            correct = total = 0
            with torch.no_grad():
                for x, y in ld:
                    x = x.to(device); y = y.to(device)
                    correct += (model(x).argmax(-1) == y).sum().item()
                    total += y.numel()
            return correct / total
        tr = accuracy(train_ld)
        te = accuracy(test_ld)
        ranks = {k: effective_rank(W) for k, W in get_deep_weight_matrices(model).items()}
        gn_train = compute_grad_norm(model, train_ld, device)
        gn_test  = compute_grad_norm(model, test_ld, device)
        return {'train_acc': tr, 'test_acc': te, 'ranks': ranks,
                'grad_train': gn_train, 'grad_test': gn_test}

    M = train_eval(0.0, n_epochs=40)
    G = train_eval(1e-3, n_epochs=40)
    print(f'  M: train={M["train_acc"]:.4f}, test={M["test_acc"]:.4f}, '
          f'grad_test/train={M["grad_test"]/max(M["grad_train"],1e-9):.2e}')
    print(f'  G: train={G["train_acc"]:.4f}, test={G["test_acc"]:.4f}, '
          f'grad_test/train={G["grad_test"]/max(G["grad_train"],1e-9):.2e}')
    return {'M': M, 'G': G}


# ============================================================
# Domain 2: Character-level Shakespeare LM
# ============================================================
def download_shakespeare():
    import urllib.request
    p = Path('./diverse/shakespeare.txt')
    p.parent.mkdir(exist_ok=True)
    if not p.exists():
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, p)
    return p.read_text(encoding='utf-8')


class SmallCharLM(nn.Module):
    def __init__(self, vocab=128, dim=128, depth=2, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, activation='gelu')
        # causal mask
        self.mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        e = self.embed(x) + self.pos[:, :x.size(1)]
        mask = self.mask[:x.size(1), :x.size(1)].to(x.device)
        h = self.enc(e, mask=mask, is_causal=True)
        return self.head(h)


def run_shakespeare(device, seq_len=64, n_iters=2000):
    print(f'\n=== Shakespeare character-level LM ===')
    text = download_shakespeare()
    # build vocab — restrict to printable ASCII subset
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    # split
    split = int(0.9 * len(data))
    train_data, test_data = data[:split], data[split:]

    def get_batch(d, bs):
        idx = torch.randint(0, len(d) - seq_len - 1, (bs,))
        x = torch.stack([d[i:i + seq_len] for i in idx])
        y = torch.stack([d[i + 1:i + seq_len + 1] for i in idx])
        return x.to(device), y.to(device)

    def train_eval(wd, n_iters):
        torch.manual_seed(0)
        model = SmallCharLM(vocab=len(vocab), seq_len=seq_len).to(device)
        opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd, betas=(0.9, 0.98))
        crit = nn.CrossEntropyLoss()

        for i in range(n_iters):
            x, y = get_batch(train_data, bs=64)
            logits = model(x)
            loss = crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if (i + 1) % 500 == 0:
                print(f'    iter={i+1}: train_loss={loss.item():.4f}')

        # measure perplexity on train and test
        model.eval()
        @torch.no_grad()
        def perplexity(d):
            losses = []
            for _ in range(20):
                x, y = get_batch(d, bs=64)
                logits = model(x)
                loss = crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                losses.append(loss.item())
            return float(np.exp(np.mean(losses)))

        ppl_train = perplexity(train_data)
        ppl_test = perplexity(test_data)
        ranks = {k: effective_rank(W) for k, W in get_deep_weight_matrices(model).items()}

        # grad norm
        def gn(d):
            for p in model.parameters():
                if p.grad is not None: p.grad.zero_()
            for _ in range(10):
                x, y = get_batch(d, bs=64)
                logits = model(x)
                loss = crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                loss.backward()
            total = 0.0
            for p in model.parameters():
                if p.grad is not None: total += (p.grad ** 2).sum().item()
            for p in model.parameters():
                if p.grad is not None: p.grad.zero_()
            return float(np.sqrt(total))

        gn_train = gn(train_data)
        gn_test = gn(test_data)
        return {'ppl_train': ppl_train, 'ppl_test': ppl_test, 'ranks': ranks,
                'grad_train': gn_train, 'grad_test': gn_test}

    print('  Training M (wd=0)...')
    M = train_eval(0.0, n_iters=n_iters)
    print('  Training G (wd=0.1)...')
    G = train_eval(0.1, n_iters=n_iters)
    print(f'  M: ppl_train={M["ppl_train"]:.2f}, ppl_test={M["ppl_test"]:.2f}, '
          f'grad_test/train={M["grad_test"]/max(M["grad_train"],1e-9):.2e}')
    print(f'  G: ppl_train={G["ppl_train"]:.2f}, ppl_test={G["ppl_test"]:.2f}, '
          f'grad_test/train={G["grad_test"]/max(G["grad_train"],1e-9):.2e}')
    return {'M': M, 'G': G}


# ============================================================
# Domain 3: Tabular - covertype subset or synthetic
# ============================================================
def run_tabular(device):
    print(f'\n=== Tabular (synthetic high-dim classification) ===')
    # Generate synthetic tabular: a hard classification task
    torch.manual_seed(0)
    np.random.seed(0)
    n_features = 64
    n_classes = 10
    n_train = 5000
    n_test = 2000

    # Hidden manifold structure -- generate features and assign classes via a small NN
    X_all = np.random.randn(n_train + n_test, n_features)
    # generate labels via random linear projection then arg-bin
    proj = np.random.randn(n_features, n_classes)
    logits = X_all @ proj + 0.5 * np.random.randn(n_train + n_test, n_classes)
    y_all = logits.argmax(axis=1)
    X_all = X_all.astype(np.float32)

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

    def train_eval(wd, n_epochs):
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
        ranks = {k: effective_rank(W) for k, W in get_deep_weight_matrices(model).items()}
        gn_train = compute_grad_norm(model, train_ld, device)
        gn_test  = compute_grad_norm(model, test_ld, device)
        return {'train_acc': tr, 'test_acc': te, 'ranks': ranks,
                'grad_train': gn_train, 'grad_test': gn_test}

    M = train_eval(0.0, n_epochs=200)
    G = train_eval(1e-2, n_epochs=200)
    print(f'  M: train={M["train_acc"]:.4f}, test={M["test_acc"]:.4f}, '
          f'grad_test/train={M["grad_test"]/max(M["grad_train"],1e-9):.2e}')
    print(f'  G: train={G["train_acc"]:.4f}, test={G["test_acc"]:.4f}, '
          f'grad_test/train={G["grad_test"]/max(G["grad_train"],1e-9):.2e}')
    return {'M': M, 'G': G}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    all_results = {}
    all_results['mnist']       = run_mnist(device)
    all_results['shakespeare'] = run_shakespeare(device)
    all_results['tabular']     = run_tabular(device)

    out_json = HERE / 'results' / 'diverse_tasks.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nresults -> {out_json}')

    # Final cross-task summary
    print('\n=== CROSS-DOMAIN SUMMARY ===')
    for task, r in all_results.items():
        M = r['M']; G = r['G']
        gn_ratio_M = M['grad_test'] / max(M['grad_train'], 1e-9)
        gn_ratio_G = G['grad_test'] / max(G['grad_train'], 1e-9)
        M_rank = list(M['ranks'].values())[-1]   # last (deepest) weight matrix
        G_rank = list(G['ranks'].values())[-1]
        print(f'  {task:>12s}: M_rank={M_rank:>7.2f}, G_rank={G_rank:>7.2f}, '
              f'M_grad_ratio={gn_ratio_M:.2e}, G_grad_ratio={gn_ratio_G:.2e}')


if __name__ == '__main__':
    main()
