"""mech6: Force ViT-Tiny into grokking regime via tiny CIFAR subset. (Q2)

The puzzle: in tier3b/4 ViTs, gradient angle decouples to ~0 in both M and G.
Hypothesis: 50k CIFAR examples vs ViT's 6M params isn't a strong enough push
into pure memorization. The gradient angle signature only fires when the
model is *forced* to choose between memorize and generalize.

Test: train ViT-Tiny on a 500-example CIFAR-10 subset (extreme overparameterization),
no WD, no aug, very long training. If gradient angle becomes anti-aligned,
hypothesis H2.2 (insufficient pressure) is confirmed.

3 seeds. Compare to standard tier3b numbers.
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
from bulletproof3.tier3b_vit_tiny_cifar10 import ViTTiny

NUM_SEEDS = 3
EPOCHS = 1000
BATCH = 64
N_TRAIN = 500
DATA_DIR = HERE.parent / 'data'


def loaders(seed):
    t.manual_seed(seed); np.random.seed(seed)
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    full_train = torchvision.datasets.CIFAR10(str(DATA_DIR), train=True, download=True, transform=tf)
    full_test = torchvision.datasets.CIFAR10(str(DATA_DIR), train=False, download=True, transform=tf)
    # Subsample training to N_TRAIN examples
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(full_train))[:N_TRAIN]
    train_subset = t.utils.data.Subset(full_train, idx.tolist())
    return (t.utils.data.DataLoader(train_subset, batch_size=BATCH, shuffle=True, num_workers=2),
            t.utils.data.DataLoader(full_test, batch_size=BATCH, shuffle=False, num_workers=2))


def train_model(seed, mode, device):
    """mode in {'M', 'G'}: M = no WD, G = wd=5e-4."""
    wd = 5e-4 if mode == 'G' else 0.0
    tl, vl = loaders(seed)
    model = ViTTiny().to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
        if (ep + 1) % 100 == 0:
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
    X_tr, y_tr = load_subset(tl, N_TRAIN, device)  # all of them
    X_te, y_te = load_subset(vl, 1000, device)
    model.eval()
    train_loss_fn = lambda: F.cross_entropy(model(X_tr), y_tr)
    test_loss_fn  = lambda: F.cross_entropy(model(X_te), y_te)
    @t.no_grad()
    def per_ex(X, y):
        out = []
        for i in range(0, len(X), 64):
            out.append(F.cross_entropy(model(X[i:i+64]), y[i:i+64], reduction='none').cpu().numpy())
        return np.concatenate(out)
    tr_losses = per_ex(X_tr, y_tr); te_losses = per_ex(X_te, y_te)
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=10, verbose=True)
    @t.no_grad()
    def full_acc(loader):
        c = 0; n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item(); n += y.size(0)
        return c / n
    bat['train_acc'] = full_acc(tl); bat['test_acc'] = full_acc(vl)
    bat['seed'] = seed; bat['mode'] = mode; bat['n_train'] = N_TRAIN
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'mech6_vit_forced_grok.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {'M': [], 'G': []}
    for mode in ['M', 'G']:
        for seed in range(NUM_SEEDS):
            print(f'\n=== {mode} seed={seed} (N_TRAIN={N_TRAIN}, {EPOCHS} epochs) ===')
            try:
                entry = run(seed, mode, device)
                results[mode].append(entry)
                print(f'  test={entry["test_acc"]:.4f} '
                      f'top={entry["hessian_top_full"]:.3f} '
                      f'bot={entry["hessian_bot_full"]:.3f} '
                      f'cos={entry["cos_grad_train_test"]:.4f} '
                      f'mia={entry.get("mia_loss_auc",0):.4f}')
            except Exception as e:
                print(f'  error: {e}')
                results[mode].append({'mode': mode, 'seed': seed, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
