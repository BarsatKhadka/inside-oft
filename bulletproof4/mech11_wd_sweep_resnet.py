"""mech11: WD sweep on ResNet-18 CIFAR-10 to verify the sharpness reversal
holds across WD values, not just at WD=5e-4.

Sweeps WD ∈ {0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3}, no augmentation, 3 seeds each.
For each model, computes:
  - top Hessian eigenvalue (sharpness)
  - weight L2 norm
  - Petzka relative flatness (top × ||θ||²)
  - test accuracy
  - MIA AUC

Predictions to test:
  - top eig INCREASES with WD (because WD shrinks weights, raising curvature)
  - ||θ|| DECREASES monotonically with WD
  - Petzka rel_flat DECREASES with WD (the "real" sharpness)
  - test_acc has a non-monotone curve (too little WD = M, too much WD = collapse)

If Petzka monotonically decreases while top eig monotonically increases:
strongest possible confirmation of Dinh's critique in standard SGD training.
"""
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

WDS = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3]
NUM_SEEDS = 3
EPOCHS = 100  # reduced from tier2's 200 for compute budget
BATCH = 128
DATA_DIR = HERE.parent / 'data'


def make_resnet18():
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def loaders(seed):
    """No augmentation — isolate WD effect."""
    t.manual_seed(seed); np.random.seed(seed)
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    train = torchvision.datasets.CIFAR10(str(DATA_DIR), train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(str(DATA_DIR), train=False, download=True, transform=tf)
    return (t.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=2),
            t.utils.data.DataLoader(test, batch_size=BATCH, shuffle=False, num_workers=2))


def train_resnet(seed, wd, device):
    tl, vl = loaders(seed)
    model = make_resnet18().to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
        if (ep + 1) % 25 == 0:
            print(f'  ep={ep+1}: wd={wd}')
    return model, tl, vl


def run_one(seed, wd, device):
    model, tl, vl = train_resnet(seed, wd, device)
    X_tr, y_tr = load_subset(tl, 1500, device)
    X_te, y_te = load_subset(vl, 1500, device)
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
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=15, verbose=True)
    @t.no_grad()
    def full_acc(loader):
        c = 0; n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item(); n += y.size(0)
        return c / n
    bat['train_acc'] = full_acc(tl); bat['test_acc'] = full_acc(vl)
    bat['seed'] = seed; bat['wd'] = wd
    # Petzka explicit
    if 'hessian_top_full' in bat and 'weight_l2_norm' in bat:
        bat['relative_flatness'] = float(bat['hessian_top_full']) * (bat['weight_l2_norm'] ** 2)
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'mech11_wd_sweep_resnet.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for wd in WDS:
        key = f'wd{wd}'
        results[key] = []
        for seed in range(NUM_SEEDS):
            print(f'\n=== wd={wd} seed={seed} ===')
            try:
                entry = run_one(seed, wd, device)
                results[key].append(entry)
                print(f'  test={entry["test_acc"]:.4f}  '
                      f'top={entry["hessian_top_full"]:.2f}  '
                      f'||θ||={entry["weight_l2_norm"]:.2f}  '
                      f'Petzka={entry.get("relative_flatness", float("nan")):.2e}  '
                      f'MIA={entry.get("mia_loss_auc", 0):.3f}')
            except Exception as e:
                print(f'  error: {e}')
                import traceback; traceback.print_exc()
                results[key].append({'wd': wd, 'seed': seed, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
    # Summary
    print('\n=== WD sweep summary (mean over seeds) ===')
    print(f'{"WD":>10s} {"test_acc":>10s} {"top_eig":>10s} {"||θ||":>10s} {"Petzka":>14s} {"MIA":>8s}')
    for wd in WDS:
        cells = [r for r in results[f'wd{wd}'] if 'error' not in r]
        if not cells: continue
        ta = np.mean([r['test_acc'] for r in cells])
        te = np.mean([r['hessian_top_full'] for r in cells])
        wn = np.mean([r['weight_l2_norm'] for r in cells])
        rf = np.mean([r.get('relative_flatness', float('nan')) for r in cells])
        mia = np.mean([r.get('mia_loss_auc', float('nan')) for r in cells])
        print(f'{wd:>10.5f} {ta:>10.4f} {te:>10.2f} {wn:>10.2f} {rf:>14.2e} {mia:>8.3f}')


if __name__ == '__main__':
    main()
