"""ViT long training on CIFAR: does the signature emerge with proper overfitting?

Previous ViT-Small training (100 epochs) didn't show clean saddle signature.
Hypothesis: ViT just needs longer training to overfit hard enough.

This script trains ViT-Small for 400 epochs M (no WD/aug) and 300 G (WD/aug)
and measures structural signatures.

Usage:
    python overnight2/vit_long_cifar.py
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
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def effective_rank(W):
    s = torch.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


class SmallViT(nn.Module):
    def __init__(self, dim=192, depth=6, heads=3, mlp_ratio=2, num_classes=10, patch=4, img=32):
        super().__init__()
        self.patch = patch
        n = (img // patch) ** 2
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * mlp_ratio, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.enc(x)
        return self.head(self.norm(x[:, 0]))


def get_loaders(batch_size, augment):
    tf_train = T.Compose([
        T.RandomCrop(32, padding=4) if augment else T.Lambda(lambda x: x),
        T.RandomHorizontalFlip() if augment else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tf_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    root = './trackb/cifar_data'
    train = torchvision.datasets.CIFAR10(root=root, train=True,  download=False, transform=tf_train)
    test  = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=tf_test)
    return (DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(test,  batch_size=512,        shuffle=False, num_workers=4, pin_memory=True))


def evaluate(model, loader, device):
    model.eval()
    c = t_ = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(-1) == y).sum().item()
            t_ += y.numel()
    return c / t_


def grad_norm(model, loader, device, n_batches=10):
    crit = nn.CrossEntropyLoss()
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    for i, (x, y) in enumerate(loader):
        if i >= n_batches: break
        x, y = x.to(device), y.to(device)
        crit(model(x), y).backward()
    total = sum((p.grad ** 2).sum().item() for p in model.parameters() if p.grad is not None)
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    return float(np.sqrt(total))


def train_one(mode, device, n_epochs):
    print(f'\n=== ViT-Small {mode} ({n_epochs} epochs) ===')
    torch.manual_seed(0)
    model = SmallViT().to(device)
    wd = 5e-4 if mode == 'G' else 0.0
    augment = (mode == 'G')
    lr = 1e-3
    train_ld, test_ld = get_loaders(128, augment)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        for x, y in train_ld:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        if (ep + 1) % 25 == 0:
            tr = evaluate(model, train_ld, device)
            te = evaluate(model, test_ld, device)
            print(f'  ep={ep+1}: train={tr:.4f}, test={te:.4f}, elapsed={time.time()-t0:.0f}s')

    # final measurements
    tr = evaluate(model, train_ld, device)
    te = evaluate(model, test_ld, device)
    # deep weight ranks (look at MLP layers in transformer blocks)
    ranks = {}
    for name, p in model.named_parameters():
        if p.ndim >= 2 and 'norm' not in name.lower() and 'pos' not in name.lower():
            W = p.detach().reshape(p.shape[0], -1)
            ranks[name] = effective_rank(W)
    g_tr = grad_norm(model, train_ld, device)
    g_te = grad_norm(model, test_ld, device)
    return {'mode': mode, 'train_acc': tr, 'test_acc': te, 'ranks': ranks,
            'grad_train': g_tr, 'grad_test': g_te}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    results = {}
    results['M'] = train_one('M', device, n_epochs=400)
    results['G'] = train_one('G', device, n_epochs=300)
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'vit_long_cifar.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== Final ViT-Small results ===')
    for mode in ['M', 'G']:
        r = results[mode]
        gr = r['grad_test'] / max(r['grad_train'], 1e-9)
        # find a deep transformer block weight (e.g. layer4 or 5 mlp)
        deep_keys = [k for k in r['ranks'] if 'enc.layers' in k]
        if deep_keys:
            avg_deep_rank = np.mean([r['ranks'][k] for k in deep_keys])
        else:
            avg_deep_rank = list(r['ranks'].values())[-1]
        print(f'  {mode}: train={r["train_acc"]:.4f}, test={r["test_acc"]:.4f}, '
              f'avg_deep_rank={avg_deep_rank:.2f}, grad_ratio={gr:.2e}')


if __name__ == '__main__':
    main()
