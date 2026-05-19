"""mech3: Mode connectivity across tiers (Q5).

For tier2 (ResNet-18 CIFAR-10), tier3b (ViT-Tiny CIFAR-10), and tier4 (ViT-Small
CIFAR-100), train one M model and one G model, then linearly interpolate
between them in weight space and evaluate loss at 11 alphas.

If there is a high-loss barrier in the path: M and G are in different basins
(consistent with the categorical-regime claim). If there is no barrier: they
are in the same basin, and the structural-signature differences are just
"different points within one basin" rather than "different solution types" —
which would explain why ViT signatures decouple.

Critical experiment for the paper. Determines whether the M/G distinction
is categorical (basins) or continuous (one basin, different points).
"""
import json
import argparse
import copy
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

DATA_DIR = HERE.parent / 'data'
OUT_DIR = HERE / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Models ----------

def make_resnet18(n_classes=10):
    m = torchvision.models.resnet18(num_classes=n_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


class ViT(nn.Module):
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
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))


# ---------- Data loaders ----------

def cifar_loaders(seed, augment, dataset='cifar10'):
    t.manual_seed(seed); np.random.seed(seed)
    if dataset == 'cifar10':
        mean, std = (0.5,)*3, (0.5,)*3
        ds_cls = torchvision.datasets.CIFAR10
    else:
        mean = (0.5071, 0.4865, 0.4409); std = (0.2673, 0.2564, 0.2762)
        ds_cls = torchvision.datasets.CIFAR100
    if augment:
        tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize(mean, std)])
    else:
        tf_train = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = ds_cls(str(DATA_DIR), train=True, download=True, transform=tf_train)
    test = ds_cls(str(DATA_DIR), train=False, download=True, transform=tf_test)
    bs = 256 if 'vit' in dataset else 128
    return (t.utils.data.DataLoader(train, batch_size=bs, shuffle=True, num_workers=2),
            t.utils.data.DataLoader(test, batch_size=bs, shuffle=False, num_workers=2))


# ---------- Train ----------

def train_resnet(seed, mode, device, epochs=80):
    augment = (mode == 'G'); wd = 5e-4 if mode == 'G' else 0.0
    tl, vl = cifar_loaders(seed, augment, 'cifar10')
    model = make_resnet18(10).to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f'  ep={ep+1}: train_loss={F.cross_entropy(model(x), y).item():.4f}')
    return model, tl, vl


def train_vit(seed, mode, device, dim=192, depth=12, heads=3, n_classes=10,
              dataset='cifar10', epochs=120):
    augment = (mode == 'G'); wd = 5e-4 if mode == 'G' else 0.0
    tl, vl = cifar_loaders(seed, augment, dataset)
    model = ViT(dim=dim, depth=depth, heads=heads, n_classes=n_classes).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f'  ep={ep+1}: train_loss={F.cross_entropy(model(x), y).item():.4f}')
    return model, tl, vl


# ---------- Interpolation ----------

@t.no_grad()
def interpolate_eval(model_m, model_g, alphas, tl, vl, device, n_eval_batches=20):
    """For each alpha, set model weights to (1-a)*M + a*G, evaluate train+test loss."""
    sd_m = model_m.state_dict()
    sd_g = model_g.state_dict()
    interp_model = copy.deepcopy(model_m)
    results = []
    for a in alphas:
        new_sd = {}
        for k in sd_m:
            if sd_m[k].dtype.is_floating_point:
                new_sd[k] = (1 - a) * sd_m[k] + a * sd_g[k]
            else:
                new_sd[k] = sd_m[k]
        interp_model.load_state_dict(new_sd)
        interp_model.eval()
        # Evaluate train + test loss
        tr_loss, tr_n = 0.0, 0; te_loss, te_n = 0.0, 0
        for i, (x, y) in enumerate(tl):
            if i >= n_eval_batches: break
            x, y = x.to(device), y.to(device)
            tr_loss += F.cross_entropy(interp_model(x), y, reduction='sum').item()
            tr_n += y.size(0)
        for i, (x, y) in enumerate(vl):
            if i >= n_eval_batches: break
            x, y = x.to(device), y.to(device)
            te_loss += F.cross_entropy(interp_model(x), y, reduction='sum').item()
            te_n += y.size(0)
        results.append({
            'alpha': a,
            'train_loss': tr_loss / tr_n,
            'test_loss': te_loss / te_n,
        })
        print(f'  alpha={a:.2f}: train={results[-1]["train_loss"]:.4f}, test={results[-1]["test_loss"]:.4f}')
    # Barrier height (test loss at midpoint - max(endpoints))
    endpoint_test = max(results[0]['test_loss'], results[-1]['test_loss'])
    midpoint_test = results[len(results)//2]['test_loss']
    barrier = midpoint_test - endpoint_test
    return results, barrier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=['tier2', 'tier3b', 'tier4'], required=True)
    ap.add_argument('--epochs', type=int, default=None,
                    help='Override training epochs (default: tier-specific)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print(f'=== mech3 mode connectivity: {args.tier} ===')

    if args.tier == 'tier2':
        epochs = args.epochs or 80
        print(f'Training M (ResNet-18 CIFAR-10, wd=0, no aug, {epochs} epochs)...')
        m_model, tl, vl = train_resnet(args.seed, 'M', device, epochs=epochs)
        print(f'Training G (ResNet-18 CIFAR-10, wd=5e-4, aug, {epochs} epochs)...')
        g_model, _, _ = train_resnet(args.seed, 'G', device, epochs=epochs)
    elif args.tier == 'tier3b':
        epochs = args.epochs or 120
        print(f'Training M (ViT-Tiny CIFAR-10, wd=0, no aug, {epochs} epochs)...')
        m_model, tl, vl = train_vit(args.seed, 'M', device,
                                     dim=192, depth=12, heads=3, n_classes=10,
                                     dataset='cifar10', epochs=epochs)
        print(f'Training G (ViT-Tiny CIFAR-10, wd=5e-4, aug, {epochs} epochs)...')
        g_model, _, _ = train_vit(args.seed, 'G', device,
                                   dim=192, depth=12, heads=3, n_classes=10,
                                   dataset='cifar10', epochs=epochs)
    else:  # tier4
        epochs = args.epochs or 100
        print(f'Training M (ViT-Small CIFAR-100, wd=0, no aug, {epochs} epochs)...')
        m_model, tl, vl = train_vit(args.seed, 'M', device,
                                     dim=384, depth=12, heads=6, n_classes=100,
                                     dataset='cifar100', epochs=epochs)
        print(f'Training G (ViT-Small CIFAR-100, wd=5e-4, aug, {epochs} epochs)...')
        g_model, _, _ = train_vit(args.seed, 'G', device,
                                   dim=384, depth=12, heads=6, n_classes=100,
                                   dataset='cifar100', epochs=epochs)

    alphas = np.linspace(0, 1, 11).tolist()
    print(f'\nInterpolating along {len(alphas)} alphas...')
    results, barrier = interpolate_eval(m_model, g_model, alphas, tl, vl, device)
    print(f'\nBarrier height (midpoint test loss - max endpoint test loss) = {barrier:.4f}')

    out = {
        'tier': args.tier,
        'seed': args.seed,
        'epochs': epochs,
        'alphas': alphas,
        'curve': results,
        'barrier_height_test': barrier,
        'endpoint_m_test_loss': results[0]['test_loss'],
        'endpoint_g_test_loss': results[-1]['test_loss'],
        'midpoint_test_loss': results[len(results) // 2]['test_loss'],
    }
    with open(OUT_DIR / f'mech3_mode_connectivity_{args.tier}.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
