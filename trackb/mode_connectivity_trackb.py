"""Mode connectivity for Track B: barrier between M_CIFAR and G_CIFAR.

Linearly interpolate between M_CIFAR and G_CIFAR weight states. Evaluate
train+test loss/accuracy at each alpha.

If barrier exists, they're in different basins.
If barrier is small, they're in the same basin (just different points).

For benign overfitting (Track B), we might see a SMALLER barrier than Track A's
4e7 ratio -- if M and G are mostly similar, the linear interpolation might be
flat.

Usage:
    python trackb/mode_connectivity_trackb.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.models as models


def make_resnet18():
    m = models.resnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


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


def load_state(p):
    return torch.load(p, map_location='cpu', weights_only=True)['model']


def interp_state(s1, s2, alpha):
    out = {}
    for k in s1:
        if torch.is_floating_point(s1[k]):
            out[k] = (1 - alpha) * s1[k] + alpha * s2[k]
        else:
            out[k] = s1[k]   # leave integer buffers as-is
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    train_loader, test_loader = get_loaders()

    s_M = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')
    s_G = load_state(HERE / 'checkpoints' / 'G' / 'final.pt')

    model = make_resnet18().to(device)

    alphas = np.linspace(0, 1, 11)    # 11 points for speed; can extend if interesting
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    print(f'{"alpha":>6}  {"train_loss":>12}  {"train_acc":>10}  {"test_loss":>12}  {"test_acc":>10}')
    for a in alphas:
        s = interp_state(s_M, s_G, a)
        model.load_state_dict(s)
        model.eval()
        tr_l, tr_a = evaluate(model, train_loader, device)
        te_l, te_a = evaluate(model, test_loader, device)
        train_losses.append(tr_l); train_accs.append(tr_a)
        test_losses.append(te_l);  test_accs.append(te_a)
        print(f'{a:>6.2f}  {tr_l:>12.4f}  {tr_a:>10.4f}  {te_l:>12.4f}  {te_a:>10.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(alphas, train_losses, marker='o', label='train', color='tab:blue')
    axes[0].plot(alphas, test_losses,  marker='s', label='test',  color='tab:orange')
    axes[0].set_xlabel('alpha (0 = M, 1 = G)')
    axes[0].set_ylabel('loss')
    axes[0].set_title('Loss along M_CIFAR -> G_CIFAR interpolation')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(alphas, train_accs, marker='o', label='train acc', color='tab:blue')
    axes[1].plot(alphas, test_accs,  marker='s', label='test acc',  color='tab:orange')
    axes[1].set_xlabel('alpha (0 = M, 1 = G)')
    axes[1].set_ylabel('accuracy')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title('Accuracy along M_CIFAR -> G_CIFAR interpolation')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Mode connectivity in Track B: are M and G in the same basin in CIFAR?')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_mode_connectivity_trackb.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')

    # Summary
    mid = len(alphas) // 2
    print(f'\nMidpoint (alpha=0.5):  train_loss={train_losses[mid]:.4f}  test_acc={test_accs[mid]:.4f}')
    print(f'Endpoint M (alpha=0):  train_loss={train_losses[0]:.4f}  test_acc={test_accs[0]:.4f}')
    print(f'Endpoint G (alpha=1):  train_loss={train_losses[-1]:.4f}  test_acc={test_accs[-1]:.4f}')
    print(f'Barrier ratio (mid loss / max endpoint loss): {train_losses[mid] / max(train_losses[0], train_losses[-1]):.2e}')


if __name__ == '__main__':
    main()
