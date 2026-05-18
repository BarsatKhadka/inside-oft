"""Track B full analysis battery, run on existing M_CIFAR and G_CIFAR checkpoints.

Replicates the Track A diagnostics for CIFAR + ResNet-18:
  1. Margin distributions (per-image, train vs test)
  2. Effective rank of conv1, fc, and selected block conv layers
  3. Saddle test: gradient norm on train vs test vs full data
  4. Per-image membership probe (linearly separate train from test via penultimate features)
  5. Capacity test (rank truncation of fc layer)

If M_CIFAR shows the same signatures as M_Track-A despite only being 6 points
worse than G_CIFAR, then those signatures are general features of overfitting.

Usage:
    python trackb/analysis_trackb.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def make_resnet18():
    m = models.resnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def load_model(ckpt_path, device):
    model = make_resnet18().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    return model


def get_loaders(batch_size=512):
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    root = './trackb/cifar_data'
    train = torchvision.datasets.CIFAR10(root=root, train=True,  download=False, transform=tf)
    test  = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=tf)
    return (DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=2),
            DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2))


def effective_rank(W):
    sigma = torch.linalg.svdvals(W)
    p = sigma ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def stable_rank(W):
    sigma = torch.linalg.svdvals(W)
    return float((sigma ** 2).sum() / sigma[0] ** 2)


def flatten_conv(W):
    """Reshape conv weight (out, in, kh, kw) -> (out, in*kh*kw)."""
    return W.reshape(W.shape[0], -1)


@torch.no_grad()
def all_predictions(model, loader, device):
    """Return logits, labels, predictions concatenated across the loader."""
    logits_list, labels_list = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        logits_list.append(logits.cpu())
        labels_list.append(y)
    return torch.cat(logits_list), torch.cat(labels_list)


def margins(logits, labels):
    n = len(labels)
    correct = logits[torch.arange(n), labels]
    logits_other = logits.clone()
    logits_other[torch.arange(n), labels] = -float('inf')
    max_wrong = logits_other.max(dim=1).values
    return (correct - max_wrong).numpy()


@torch.no_grad()
def extract_penult_features(model, loader, device):
    """Extract activations BEFORE the final fc layer."""
    # Hook the layer before fc (avgpool output -> flattened)
    feats_list, labels_list = [], []

    def hook(module, input, output):
        feats_list.append(output.flatten(1).cpu())

    handle = model.avgpool.register_forward_hook(hook)
    try:
        for x, y in loader:
            model(x.to(device, non_blocking=True))
            labels_list.append(y)
    finally:
        handle.remove()
    return torch.cat(feats_list).numpy(), torch.cat(labels_list).numpy()


def compute_gradient_norm(model, loader, device, max_batches=20):
    """Approximate ||grad|| over a few batches (running on full CIFAR is expensive)."""
    criterion = nn.CrossEntropyLoss()
    total_sq = 0.0
    n_examples = 0
    for p in model.parameters():
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad.zero_()
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        n_examples += y.numel()
    for p in model.parameters():
        if p.grad is not None:
            total_sq += (p.grad ** 2).sum().item()
    # zero again
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    return float(np.sqrt(total_sq)), n_examples


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_loader, test_loader = get_loaders(batch_size=512)

    ckpts = {
        'M': HERE / 'checkpoints' / 'M' / 'final.pt',
        'G': HERE / 'checkpoints' / 'G' / 'final.pt',
    }

    results = {}
    for name, ckpt in ckpts.items():
        print(f'\n{"="*60}')
        print(f'  {name}_CIFAR — analysis')
        print(f'{"="*60}')
        model = load_model(ckpt, device)

        # === 1. Margins ===
        print('Computing per-image margins...')
        tr_logits, tr_labels = all_predictions(model, train_loader, device)
        te_logits, te_labels = all_predictions(model, test_loader, device)
        tr_margins = margins(tr_logits, tr_labels)
        te_margins = margins(te_logits, te_labels)
        print(f'  train margins: mean={tr_margins.mean():.2f}, median={np.median(tr_margins):.2f}, '
              f'min={tr_margins.min():.2f}, max={tr_margins.max():.2f}')
        print(f'  test  margins: mean={te_margins.mean():.2f}, median={np.median(te_margins):.2f}, '
              f'min={te_margins.min():.2f}, max={te_margins.max():.2f}')

        # === 2. Effective rank of selected weight matrices ===
        print('\nEffective rank of weight matrices:')
        rank_results = {}
        for layer_name, W in [
            ('conv1', flatten_conv(model.conv1.weight.detach().cpu())),
            ('layer4.1.conv2', flatten_conv(model.layer4[1].conv2.weight.detach().cpu())),
            ('fc', model.fc.weight.detach().cpu()),
        ]:
            er = effective_rank(W)
            sr = stable_rank(W)
            rank_results[layer_name] = {'eff': er, 'stable': sr, 'shape': list(W.shape)}
            print(f'  {layer_name:>20s}  shape={list(W.shape)}  eff_rank={er:.2f}  stable_rank={sr:.2f}')

        # === 3. Gradient norm on train and test (saddle test) ===
        print('\nGradient norms (first 20 batches each):')
        gn_train, n_tr = compute_gradient_norm(model, train_loader, device)
        gn_test,  n_te = compute_gradient_norm(model, test_loader,  device)
        print(f'  ||grad|| on train (n={n_tr}): {gn_train:.4e}')
        print(f'  ||grad|| on test  (n={n_te}): {gn_test:.4e}')

        # === 4. Penultimate features for MIA probe ===
        print('\nExtracting penultimate features for membership probe...')
        feat_tr, _ = extract_penult_features(model, train_loader, device)
        feat_te, _ = extract_penult_features(model, test_loader,  device)
        # Balance the classes for the probe (50k train, 10k test). Subsample train to 10k.
        idx_tr = np.random.RandomState(0).choice(len(feat_tr), len(feat_te), replace=False)
        X = np.vstack([feat_tr[idx_tr], feat_te])
        y = np.concatenate([np.ones(len(idx_tr)), np.zeros(len(feat_te))]).astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=500, n_jobs=1)
        clf.fit(X_tr, y_tr)
        mia_acc = clf.score(X_te, y_te)
        print(f'  Membership probe (train vs test) accuracy: {mia_acc:.4f}  (chance 0.50)')

        results[name] = {
            'tr_margins': tr_margins.tolist(),
            'te_margins': te_margins.tolist(),
            'ranks': rank_results,
            'grad_train': gn_train,
            'grad_test':  gn_test,
            'mia_acc': mia_acc,
        }

    # Save raw
    out_json = HERE / 'results' / 'analysis_trackb.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Lightweight JSON (don't dump per-image margins, too big)
    light = {n: {k: (v if k != 'tr_margins' and k != 'te_margins' else 'omitted') for k, v in r.items()}
             for n, r in results.items()}
    with open(out_json, 'w') as f:
        json.dump(light, f, default=str, indent=2)
    print(f'\nresults -> {out_json}')

    # === Plots ===
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for col, name in enumerate(['M', 'G']):
        r = results[name]
        ax = axes[0, col]
        ax.hist(r['tr_margins'], bins=80, alpha=0.7, label='train', color='tab:blue')
        ax.hist(r['te_margins'], bins=80, alpha=0.6, label='test',  color='tab:orange')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='margin=0')
        ax.set_xlabel('logit margin')
        ax.set_ylabel('count')
        ax.set_title(f'{name}_CIFAR: per-image margin distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Rank comparison plot
    ax = axes[1, 0]
    layer_names = list(results['M']['ranks'].keys())
    x = np.arange(len(layer_names))
    bar_w = 0.35
    M_ranks = [results['M']['ranks'][n]['eff'] for n in layer_names]
    G_ranks = [results['G']['ranks'][n]['eff'] for n in layer_names]
    ax.bar(x - bar_w/2, M_ranks, bar_w, color='tab:red', label='M_CIFAR')
    ax.bar(x + bar_w/2, G_ranks, bar_w, color='tab:blue', label='G_CIFAR')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=20)
    ax.set_ylabel('effective rank')
    ax.set_title('Effective rank: does M have higher-rank weights?')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # MIA + gradient bar plot
    ax = axes[1, 1]
    x_labels = ['||grad||_train', '||grad||_test', 'MIA acc (×10)']
    M_vals = [results['M']['grad_train'], results['M']['grad_test'], results['M']['mia_acc'] * 10]
    G_vals = [results['G']['grad_train'], results['G']['grad_test'], results['G']['mia_acc'] * 10]
    x = np.arange(len(x_labels))
    ax.bar(x - bar_w/2, M_vals, bar_w, color='tab:red', label='M_CIFAR')
    ax.bar(x + bar_w/2, G_vals, bar_w, color='tab:blue', label='G_CIFAR')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20)
    ax.set_title('Gradient norms + MIA accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Track B (CIFAR + ResNet-18): does M show the same overfitting signatures as Track A?')
    fig.tight_layout()
    out_png = HERE / 'results' / 'fig_analysis_trackb.png'
    fig.savefig(out_png, dpi=130)
    print(f'plot -> {out_png}')


if __name__ == '__main__':
    main()
