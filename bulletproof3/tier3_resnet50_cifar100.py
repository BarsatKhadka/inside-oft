"""Tier 3: ResNet-50 on CIFAR-100. 3 seeds M + 3 seeds G. Full signatures."""
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
BATCH_SIZE = 128
DATA_DIR = HERE.parent / 'data'


def make_resnet50():
    m = torchvision.models.resnet50(num_classes=100)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def loaders(seed, augment):
    t.manual_seed(seed); np.random.seed(seed)
    mean = (0.5071, 0.4865, 0.4409); std = (0.2673, 0.2564, 0.2762)
    if augment:
        tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize(mean, std)])
    else:
        tf_train = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = torchvision.datasets.CIFAR100(str(DATA_DIR), train=True, download=True, transform=tf_train)
    test = torchvision.datasets.CIFAR100(str(DATA_DIR), train=False, download=True, transform=tf_test)
    return (t.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
            t.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2))


def train_model(seed, mode, device):
    augment = (mode == 'G'); wd = 5e-4 if mode == 'G' else 0.0
    tl, vl = loaders(seed, augment)
    model = make_resnet50().to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
        if (ep + 1) % 15 == 0:
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
    # Hessian on ResNet-50 still OOMs at 500. Drop to 200.
    X_tr, y_tr = load_subset(tl_noaug, 200, device)
    X_te, y_te = load_subset(vl, 200, device)
    model.eval()
    train_loss_fn = lambda: F.cross_entropy(model(X_tr), y_tr)
    test_loss_fn  = lambda: F.cross_entropy(model(X_te), y_te)
    @t.no_grad()
    def per_ex(X, y):
        out = []
        for i in range(0, len(X), 96):
            out.append(F.cross_entropy(model(X[i:i+96]), y[i:i+96], reduction='none').cpu().numpy())
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
    out_path = HERE / 'results' / 'tier3_resnet50_cifar100.json'
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
