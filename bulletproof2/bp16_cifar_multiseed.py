"""bp16: CIFAR-10 ResNet-18 multi-seed (5 seeds M + 5 seeds G).

Battery per model:
  - Final test acc
  - Per-layer effective rank
  - Gradient norm train/test/full
  - Loss-based MIA AUC
  - Probe selectivity stub (deferred to a downstream analysis)
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
EPOCHS = {'M': 200, 'G': 200}  # M overfits, G uses standard training
BATCH_SIZE = 128
DATA_DIR = HERE.parent / 'data'


def get_loaders(seed, augment):
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


def make_resnet18():
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def train_cifar(seed, mode, device):
    augment = (mode == 'G')
    wd = 5e-4 if mode == 'G' else 0.0
    tl, vl = get_loaders(seed, augment=augment)
    model = make_resnet18().to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS[mode])
    n_epochs = EPOCHS[mode]
    for ep in range(n_epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0:
            model.eval()
            correct = 0; total = 0
            with t.no_grad():
                for x, y in vl:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
            print(f'  ep={ep+1}: test_acc={correct/total:.4f}')
    return model


@t.no_grad()
def eval_loaders(model, tl, vl, device):
    model.eval()
    def stats(loader):
        all_loss, all_correct, all_n = [], 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x); l = F.cross_entropy(logits, y, reduction='none')
            all_loss.append(l.cpu().numpy())
            all_correct += (logits.argmax(1) == y).sum().item(); all_n += y.size(0)
        return np.concatenate(all_loss), all_correct / all_n
    l_tr, acc_tr = stats(tl); l_te, acc_te = stats(vl)
    return l_tr, acc_tr, l_te, acc_te


def grad_norm(model, loader, device, n_batches=20):
    model.train()
    total = 0.0; count = 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches: break
        x, y = x.to(device), y.to(device)
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        F.cross_entropy(model(x), y).backward()
        n = sum((p.grad**2).sum().item() for p in model.parameters() if p.grad is not None)
        total += n; count += 1
    return float(np.sqrt(total / max(count, 1)))


def loss_mia_auc(l_tr, l_te):
    s = np.concatenate([-l_tr, -l_te]); y = np.concatenate([np.ones_like(l_tr), np.zeros_like(l_te)])
    order = np.argsort(-s); ys = y[order]
    n_pos = ys.sum(); n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    return float(np.trapz(np.cumsum(ys)/n_pos, np.cumsum(1-ys)/n_neg))


def resnet_ranks(model):
    ranks = {}
    for name, p in model.named_parameters():
        if p.ndim >= 2 and 'weight' in name and 'conv' in name or 'fc.weight' in name:
            W = p.detach().reshape(p.shape[0], -1)
            ranks[name] = effective_rank(W)
    return ranks


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    for mode in ['M', 'G']:
        for seed in range(NUM_SEEDS):
            print(f'\n=== {mode} seed={seed} ===')
            model = train_cifar(seed, mode, device)
            tl_no_aug, vl = get_loaders(seed, augment=False)
            l_tr, acc_tr, l_te, acc_te = eval_loaders(model, tl_no_aug, vl, device)
            ranks = resnet_ranks(model)
            g_tr = grad_norm(model, tl_no_aug, device); g_te = grad_norm(model, vl, device)
            entry = {
                'mode': mode, 'seed': seed,
                'train_acc': acc_tr, 'test_acc': acc_te,
                'mean_train_loss': float(l_tr.mean()),
                'mean_test_loss': float(l_te.mean()),
                'ranks': ranks,
                'rank_fc': ranks.get('fc.weight'),
                'grad_train': g_tr, 'grad_test': g_te,
                'grad_test_over_train': g_te / max(g_tr, 1e-12),
                'mia_auc': loss_mia_auc(l_tr, l_te),
            }
            results[mode].append(entry)
            print(f'  test_acc={acc_te:.4f}, rank_fc={entry["rank_fc"]}, grad_ratio={entry["grad_test_over_train"]:.2e}, mia={entry["mia_auc"]:.4f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp16_cifar_multiseed.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
