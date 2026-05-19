"""mech4: ResNet 2x2 ablation — what causes the sharpness REVERSAL? (Q1)

The puzzle: in tier2, G (wd=5e-4 + aug) has top Hessian eigenvalue ~110, M
(wd=0 + no aug) has top eig ~30. Direction OPPOSITE to algorithmic tier0.

Hypothesis test via 2x2:
  (wd=0, aug=No)   = current M, top eig ~30
  (wd=0, aug=Yes)  = isolates aug effect
  (wd=5e-4, aug=No) = isolates WD effect
  (wd=5e-4, aug=Yes) = current G, top eig ~110

If sharpness ordering tracks WD only: H1.1 (WD constrains weight subspace → sharper).
If it tracks aug only: H1.2 (aug provides gradient noise → sharper minima).
If both contribute additively or interactively: more complex story.

3 seeds per cell = 12 runs. Full signature battery per model.
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

NUM_SEEDS = 3
EPOCHS = 150
BATCH = 128
DATA_DIR = HERE.parent / 'data'


def make_resnet18():
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


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
    return (t.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=2),
            t.utils.data.DataLoader(test, batch_size=BATCH, shuffle=False, num_workers=2))


def train_one(seed, wd, augment, device):
    tl, vl = loaders(seed, augment)
    model = make_resnet18().to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
        if (ep + 1) % 30 == 0:
            print(f'  ep={ep+1}')
    return model, tl, vl


def run_cell(seed, wd, augment, device):
    print(f'  training: wd={wd}, augment={augment}, seed={seed}')
    model, tl, vl = train_one(seed, wd, augment, device)
    tl_noaug, _ = loaders(seed, augment=False)
    X_tr, y_tr = load_subset(tl_noaug, 2000, device)
    X_te, y_te = load_subset(vl, 2000, device)
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
    bat['train_acc'] = full_acc(tl_noaug); bat['test_acc'] = full_acc(vl)
    bat['seed'] = seed; bat['wd'] = wd; bat['augment'] = augment
    return bat


CELLS = [
    (0.0, False, 'wd0_aug0'),    # current M
    (0.0, True,  'wd0_aug1'),    # aug only
    (5e-4, False, 'wd1_aug0'),   # WD only
    (5e-4, True,  'wd1_aug1'),   # current G
]


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'mech4_resnet_ablation.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for wd, aug, name in CELLS:
        results[name] = []
        for seed in range(NUM_SEEDS):
            print(f'\n=== {name} seed={seed} ===')
            try:
                entry = run_cell(seed, wd, aug, device)
                results[name].append(entry)
                print(f'  test={entry["test_acc"]:.4f} '
                      f'top={entry["hessian_top_full"]:.3f} '
                      f'bot={entry["hessian_bot_full"]:.3f} '
                      f'cos={entry["cos_grad_train_test"]:.4f} '
                      f'mia={entry.get("mia_loss_auc",0):.4f}')
            except Exception as e:
                print(f'  error: {e}')
                results[name].append({'seed': seed, 'wd': wd, 'augment': aug, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
