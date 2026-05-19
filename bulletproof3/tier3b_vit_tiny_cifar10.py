"""Tier 3b: ViT-Tiny on CIFAR-10 with full signatures (fixes bp20)."""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bulletproof3._signatures import compute_full_battery
from bulletproof3.tier2_resnet18_cifar10 import load_subset

NUM_SEEDS = 3
EPOCHS = 300
BATCH_SIZE = 256
DATA_DIR = HERE.parent / 'data'


class ViTTiny(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=192, depth=12, heads=3,
                  mlp_ratio=4, n_classes=10):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2
        self.patch_emb = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(t.randn(1, n_patches + 1, dim) * 0.02)
        self.cls = nn.Parameter(t.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * mlp_ratio, dropout=0.0,
                                        activation='gelu', batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.patch_emb(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = t.cat([cls, x], 1) + self.pos_emb
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x[:, 0]))


def loaders(seed, augment):
    t.manual_seed(seed); np.random.seed(seed)
    if augment:
        tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    else:
        tf_train = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    train = torchvision.datasets.CIFAR10(str(DATA_DIR), train=True, download=True, transform=tf_train)
    test = torchvision.datasets.CIFAR10(str(DATA_DIR), train=False, download=True, transform=tf_test)
    return (t.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
            t.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2))


def train_model(seed, mode, device):
    augment = (mode == 'G'); wd = 5e-4 if mode == 'G' else 0.0
    tl, vl = loaders(seed, augment)
    model = ViTTiny().to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
        if (ep + 1) % 30 == 0:
            model.eval()
            with t.no_grad():
                c = 0; n = 0
                for x, y in vl:
                    x, y = x.to(device), y.to(device)
                    c += (model(x).argmax(1) == y).sum().item(); n += y.size(0)
            print(f'  ep={ep+1}: test={c/n:.4f}')
    return model, tl, vl


def run(seed, mode, device):
    model, tl, vl = train_model(seed, mode, device)
    tl_noaug, _ = loaders(seed, augment=False)
    # ViT attention is O(N^2) — keep Hessian probe small to avoid OOM in
    # double-backward.
    X_tr, y_tr = load_subset(tl_noaug, 400, device)
    X_te, y_te = load_subset(vl, 400, device)
    model.eval()
    train_loss_fn = lambda: F.cross_entropy(model(X_tr), y_tr)
    test_loss_fn  = lambda: F.cross_entropy(model(X_te), y_te)
    @t.no_grad()
    def per_ex(X, y):
        out = []
        for i in range(0, len(X), 128):
            out.append(F.cross_entropy(model(X[i:i+128]), y[i:i+128], reduction='none').cpu().numpy())
        return np.concatenate(out)
    tr_losses = per_ex(X_tr, y_tr); te_losses = per_ex(X_te, y_te)
    print('  computing signatures...')
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=10, verbose=True)
    bat['mode'] = mode; bat['seed'] = seed
    @t.no_grad()
    def full_acc(loader):
        c = 0; n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item(); n += y.size(0)
        return c / n
    bat['train_acc'] = full_acc(tl_noaug); bat['test_acc'] = full_acc(vl)
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    out_path = HERE / 'results' / 'tier3b_vit_tiny_cifar10.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for mode in ['M', 'G']:
        for seed in range(NUM_SEEDS):
            print(f'\n=== {mode} seed={seed} ===')
            try:
                entry = run(seed, mode, device)
                results[mode].append(entry)
                print(f'  test={entry["test_acc"]:.4f} '
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
