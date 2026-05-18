"""bp19: MNIST multi-seed (5 seeds M + 5 seeds G).

Confirms the 14x rank gap from Entry 43 survives seed variance.
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

from _common import HERE, effective_rank

NUM_SEEDS = 5
EPOCHS = 100
BATCH_SIZE = 128
DATA_DIR = HERE.parent / 'data'


class MNIST_MLP(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_loaders():
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=tf)
    return (t.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            t.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False))


def grad_norm(model, loader, device, n_batches=20):
    model.train(); total = 0.0; count = 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches: break
        x, y = x.to(device), y.to(device)
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        F.cross_entropy(model(x), y).backward()
        total += sum((p.grad**2).sum().item() for p in model.parameters() if p.grad is not None)
        count += 1
    return float(np.sqrt(total / max(count, 1)))


def loss_mia_auc(l_tr, l_te):
    s = np.concatenate([-l_tr, -l_te]); y = np.concatenate([np.ones_like(l_tr), np.zeros_like(l_te)])
    order = np.argsort(-s); ys = y[order]
    n_pos = ys.sum(); n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    return float(np.trapz(np.cumsum(ys)/n_pos, np.cumsum(1-ys)/n_neg))


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    tl, vl = get_loaders()
    for mode in ['M', 'G']:
        wd = 0.0 if mode == 'M' else 1e-3
        for seed in range(NUM_SEEDS):
            print(f'\n=== {mode} seed={seed} ===')
            t.manual_seed(seed); np.random.seed(seed)
            model = MNIST_MLP().to(device)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
            for ep in range(EPOCHS):
                model.train()
                for x, y in tl:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
                if (ep + 1) % 25 == 0:
                    model.eval(); correct = 0; total = 0
                    with t.no_grad():
                        for x, y in vl:
                            x, y = x.to(device), y.to(device)
                            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
                    print(f'  ep={ep+1}: test={correct/total:.4f}')
            # Battery
            model.eval(); l_tr = []; l_te = []; c_tr = 0; n_tr = 0; c_te = 0; n_te = 0
            with t.no_grad():
                for x, y in tl:
                    x, y = x.to(device), y.to(device)
                    L = F.cross_entropy(model(x), y, reduction='none')
                    l_tr.append(L.cpu().numpy())
                    c_tr += (model(x).argmax(1) == y).sum().item(); n_tr += y.size(0)
                for x, y in vl:
                    x, y = x.to(device), y.to(device)
                    L = F.cross_entropy(model(x), y, reduction='none')
                    l_te.append(L.cpu().numpy())
                    c_te += (model(x).argmax(1) == y).sum().item(); n_te += y.size(0)
            l_tr = np.concatenate(l_tr); l_te = np.concatenate(l_te)
            g_tr = grad_norm(model, tl, device); g_te = grad_norm(model, vl, device)
            entry = {
                'mode': mode, 'seed': seed,
                'train_acc': c_tr / n_tr, 'test_acc': c_te / n_te,
                'rank_fc1': effective_rank(model.fc1.weight),
                'rank_fc2': effective_rank(model.fc2.weight),
                'rank_fc3': effective_rank(model.fc3.weight),
                'grad_train': g_tr, 'grad_test': g_te,
                'grad_test_over_train': g_te / max(g_tr, 1e-12),
                'mia_auc': loss_mia_auc(l_tr, l_te),
                'mean_train_loss': float(l_tr.mean()),
                'mean_test_loss': float(l_te.mean()),
            }
            results[mode].append(entry)
            print(f'  test_acc={entry["test_acc"]:.4f}, rank_fc2={entry["rank_fc2"]:.2f}, '
                  f'grad_ratio={entry["grad_test_over_train"]:.2e}, mia={entry["mia_auc"]:.4f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp19_mnist_multiseed.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
