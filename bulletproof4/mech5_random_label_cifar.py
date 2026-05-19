"""mech5: Random-label CIFAR — does random-label memorization differ from
benign overfitting? (Q6)

CIFAR-10 with 30% of training labels randomly reassigned. No WD, no aug.
3 seeds. Full signature battery.

Comparison:
  - tier2 M: real labels, no WD, no aug. test_acc ~0.83 (benign overfit).
  - mech5: 30% noisy labels, no WD, no aug. test_acc ~0.65-0.75 expected
    (model memorizes noise, hurts test).

If structural signatures are identical between tier2 M and mech5: signatures
can't distinguish benign overfit from label-noise memorization.

If they differ: signatures DO distinguish, and we get a new regime in the
diagnostic panel.
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
from bulletproof3.tier2_resnet18_cifar10 import load_subset, make_resnet18

NUM_SEEDS = 3
EPOCHS = 200
BATCH = 128
NOISE_FRACTION = 0.30
DATA_DIR = HERE.parent / 'data'


class NoisyCIFAR10(t.utils.data.Dataset):
    """CIFAR-10 with a fixed fraction of labels randomly reassigned at init."""
    def __init__(self, root, train, transform, noise_frac=0.30, seed=0):
        self.base = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=transform)
        rng = np.random.RandomState(seed)
        n = len(self.base)
        n_corrupt = int(noise_frac * n) if train else 0
        idx = rng.permutation(n)[:n_corrupt]
        new_labels = rng.randint(0, 10, size=n_corrupt)
        self.labels = list(self.base.targets)
        for i, lbl in zip(idx, new_labels):
            self.labels[i] = int(lbl)
        self.is_corrupted = np.zeros(n, dtype=bool)
        self.is_corrupted[idx] = True

    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, self.labels[i]


def loaders(seed):
    t.manual_seed(seed); np.random.seed(seed)
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    train = NoisyCIFAR10(str(DATA_DIR), train=True, transform=tf,
                         noise_frac=NOISE_FRACTION, seed=seed)
    test = NoisyCIFAR10(str(DATA_DIR), train=False, transform=tf,
                        noise_frac=0.0, seed=seed)
    return (t.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=2),
            t.utils.data.DataLoader(test, batch_size=BATCH, shuffle=False, num_workers=2))


def train(seed, device):
    tl, vl = loaders(seed)
    model = make_resnet18().to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
        if (ep + 1) % 25 == 0:
            print(f'  ep={ep+1}')
    return model, tl, vl


def run(seed, device):
    model, tl, vl = train(seed, device)
    X_tr, y_tr = load_subset(tl, 2000, device)
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
    bat['train_acc'] = full_acc(tl); bat['test_acc'] = full_acc(vl)
    bat['seed'] = seed; bat['noise_fraction'] = NOISE_FRACTION
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'mech5_random_label_cifar.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(NUM_SEEDS):
        print(f'\n=== noisy seed={seed} ===')
        try:
            entry = run(seed, device)
            results.append(entry)
            print(f'  test={entry["test_acc"]:.4f} '
                  f'top={entry["hessian_top_full"]:.3f} '
                  f'bot={entry["hessian_bot_full"]:.3f} '
                  f'cos={entry["cos_grad_train_test"]:.4f} '
                  f'mia={entry.get("mia_loss_auc",0):.4f}')
        except Exception as e:
            print(f'  error: {e}')
            results.append({'seed': seed, 'error': str(e)})
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
