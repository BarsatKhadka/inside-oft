"""Vision architecture sweep: do CIFAR overfitting signatures hold across architectures?

Train M (no WD, no aug) and G (WD=5e-4, aug) on CIFAR-10 for several
vision architectures. Measure: test accuracy, effective rank of selected
weight matrices, train-vs-test gradient norm (saddle test).

Architectures:
  - ResNet-18 (have)
  - ResNet-50
  - Wide-ResNet (wide variant)
  - ViT-Tiny (via timm if available, else a custom small ViT)
  - MLP-mixer style (or plain MLP)

Reduced epoch count (100 for M, 100 for G) to fit in time budget.

Usage:
    python trackb/arch_sweep_vision.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import json
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.models as models


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


def make_resnet18():
    m = models.resnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def make_resnet50():
    m = models.resnet50(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def make_wide_resnet():
    m = models.wide_resnet50_2(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


class SmallViT(nn.Module):
    """Tiny ViT for CIFAR-10."""
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


class MLP(nn.Module):
    """Wide MLP for CIFAR-10 — fully memorization-prone."""
    def __init__(self, hidden=2048, num_classes=10, img_dim=3 * 32 * 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


ARCHITECTURES = {
    'ResNet-18':  make_resnet18,
    'ResNet-50':  make_resnet50,
    # 'Wide-ResNet-50': make_wide_resnet,    # skip if too slow
    'ViT-Small':  lambda: SmallViT(),
    'MLP-2048':   lambda: MLP(),
}


def effective_rank(W):
    s = torch.linalg.svdvals(W)
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total


def compute_grad_norm(model, loader, device, max_batches=10):
    model.train()
    crit = nn.CrossEntropyLoss()
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        loss = crit(model(x), y)
        loss.backward()
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad ** 2).sum().item()
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    return float(np.sqrt(total))


def train_arch(arch_name, builder, mode, device, epochs=100, lr=0.05):
    print(f'\n=== {arch_name} ({mode}) ===')
    torch.manual_seed(0)
    model = builder().to(device)
    wd = 5e-4 if mode == 'G' else 0.0
    augment = (mode == 'G')

    train_loader, test_loader = get_loaders(128, augment)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    crit = nn.CrossEntropyLoss()

    t_start = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        if (ep + 1) % 20 == 0:
            tr = evaluate(model, train_loader, device)
            te = evaluate(model, test_loader, device)
            print(f'  ep={ep+1}: train_acc={tr:.4f}, test_acc={te:.4f}, '
                  f'elapsed={time.time()-t_start:.1f}s')

    tr = evaluate(model, train_loader, device)
    te = evaluate(model, test_loader, device)

    # Try to grab a "deep" weight matrix per architecture
    deep_rank = None
    for name, p in model.named_parameters():
        if 'weight' in name and p.ndim >= 2 and 'norm' not in name.lower():
            w = p.detach().reshape(p.shape[0], -1).cpu()
            deep_rank = effective_rank(w)
            # Take the last such one (deepest)

    grad_train = compute_grad_norm(model, train_loader, device)
    grad_test  = compute_grad_norm(model, test_loader, device)

    return {'arch': arch_name, 'mode': mode, 'epochs': epochs,
            'train_acc': tr, 'test_acc': te, 'last_layer_rank': deep_rank,
            'grad_train': grad_train, 'grad_test': grad_test}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    results = {}
    for arch_name, builder in ARCHITECTURES.items():
        for mode in ['M', 'G']:
            key = f'{arch_name}_{mode}'
            try:
                results[key] = train_arch(arch_name, builder, mode, device)
            except Exception as e:
                print(f'FAILED: {key}: {e}')
                results[key] = {'error': str(e)}

    out_json = HERE / 'results' / 'arch_sweep_vision.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print('\n=== Architecture × Mode table ===')
    print(f'{"arch":>15}  {"mode":>4}  {"train_acc":>10}  {"test_acc":>10}  '
          f'{"deep_rank":>10}  {"grad_test/train":>16}')
    for arch_name in ARCHITECTURES:
        for mode in ['M', 'G']:
            r = results[f'{arch_name}_{mode}']
            if 'error' in r:
                print(f'  {arch_name:>15}  {mode:>4}  ERROR: {r["error"]}')
                continue
            ratio = r['grad_test'] / max(r['grad_train'], 1e-9)
            print(f'{arch_name:>15}  {mode:>4}  {r["train_acc"]:>10.4f}  {r["test_acc"]:>10.4f}  '
                  f'{r["last_layer_rank"] or 0:>10.2f}  {ratio:>16.2e}')


if __name__ == '__main__':
    main()
