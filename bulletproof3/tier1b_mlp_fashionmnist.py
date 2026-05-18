"""Tier 1b: MLP on FashionMNIST. Same setup as tier1. 5 seeds M+G."""
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
from bulletproof3.tier1_mlp_mnist import MLP, load_all

NUM_SEEDS = 5
EPOCHS = 150
BATCH_SIZE = 128
DATA_DIR = HERE.parent / 'data'


def loaders():
    tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    train = torchvision.datasets.FashionMNIST(str(DATA_DIR), train=True, download=True, transform=tf)
    test = torchvision.datasets.FashionMNIST(str(DATA_DIR), train=False, download=True, transform=tf)
    return (t.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            t.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False))


def train_model(seed, wd, device):
    t.manual_seed(seed); np.random.seed(seed)
    tl, vl = loaders()
    model = MLP().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        if (ep + 1) % 25 == 0:
            model.eval()
            with t.no_grad():
                c = 0; n = 0
                for x, y in vl:
                    x, y = x.to(device), y.to(device)
                    c += (model(x).argmax(1) == y).sum().item(); n += y.size(0)
            print(f'  ep={ep+1}: test={c/n:.4f}')
    return model, tl, vl


def run(seed, wd, device):
    model, tl, vl = train_model(seed, wd, device)
    X_tr, y_tr = load_all(tl, device); X_te, y_te = load_all(vl, device)
    np.random.seed(seed * 13)
    sub_tr = np.random.choice(len(X_tr), 4000, replace=False)
    sub_te = np.random.choice(len(X_te), 4000, replace=False)
    X_tr_h, y_tr_h = X_tr[sub_tr], y_tr[sub_tr]
    X_te_h, y_te_h = X_te[sub_te], y_te[sub_te]
    model.eval()
    train_loss_fn = lambda: F.cross_entropy(model(X_tr_h), y_tr_h)
    test_loss_fn  = lambda: F.cross_entropy(model(X_te_h), y_te_h)
    @t.no_grad()
    def per_ex(X, y):
        out = []
        for i in range(0, len(X), 256):
            out.append(F.cross_entropy(model(X[i:i+256]), y[i:i+256], reduction='none').cpu().numpy())
        return np.concatenate(out)
    tr_losses = per_ex(X_tr, y_tr); te_losses = per_ex(X_te, y_te)
    print('  computing signatures...')
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=20, verbose=True)
    @t.no_grad()
    def acc(X, y):
        c = 0; n = 0
        for i in range(0, len(X), 256):
            c += (model(X[i:i+256]).argmax(1) == y[i:i+256]).sum().item()
            n += min(256, len(X) - i)
        return c / n
    bat['train_acc'] = acc(X_tr, y_tr); bat['test_acc'] = acc(X_te, y_te)
    bat['seed'] = seed; bat['wd'] = wd
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    out_path = HERE / 'results' / 'tier1b_mlp_fashionmnist.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for label, wd in [('M', 0.0), ('G', 1e-3)]:
        for seed in range(NUM_SEEDS):
            print(f'\n=== {label} seed={seed} ===')
            try:
                entry = run(seed, wd, device)
                results[label].append(entry)
                print(f'  test={entry["test_acc"]:.4f} '
                      f'top={entry["hessian_top_full"]:.3f} '
                      f'bot={entry["hessian_bot_full"]:.3f} '
                      f'cos={entry["cos_grad_train_test"]:.4f} '
                      f'mia={entry.get("mia_loss_auc",0):.4f}')
            except Exception as e:
                print(f'  error: {e}')
                results[label].append({'seed': seed, 'wd': wd, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
