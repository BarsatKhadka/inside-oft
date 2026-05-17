"""Track B: train ResNet-18 on CIFAR-10 in three regimes.

Three modes via --mode flag:
  G  — standard training with weight decay and augmentation (generalizes)
  M  — no weight decay, no augmentation (overfits hard)
  rescue — load an M checkpoint, then continue with WD turned on

Train/test loss + accuracy logged each epoch. Checkpoints saved periodically.

The G+M comparison shows the standard overfitting curve (M test loss climbs).
The rescue mode tests whether the trajectory_rescue.py finding from Track A
generalizes: can we recover G-like generalization from an overfit M_CIFAR by
adding WD mid-training?

Usage:
    python trackb/train_cifar.py --mode G
    python trackb/train_cifar.py --mode M
    python trackb/train_cifar.py --mode rescue --rescue_from trackb/checkpoints/M/epoch_400.pt
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.models as models


def get_loaders(batch_size, augment):
    if augment:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        train_tf = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    root = './trackb/cifar_data'
    train = torchvision.datasets.CIFAR10(root=root, train=True,  download=True, transform=train_tf)
    test  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=512,        shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def make_resnet18():
    """Standard ResNet-18 adapted for 32x32 CIFAR (replace first conv + remove maxpool)."""
    m = models.resnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['G', 'M', 'rescue'])
    parser.add_argument('--rescue_from', default=None, help='checkpoint to load when mode=rescue')
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_every', type=int, default=25)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.mode == 'G':
        wd = 5e-4
        augment = True
        n_epochs = args.num_epochs or 200
        out_dir = Path('trackb/checkpoints/G')
    elif args.mode == 'M':
        wd = 0.0
        augment = False
        n_epochs = args.num_epochs or 400
        out_dir = Path('trackb/checkpoints/M')
    elif args.mode == 'rescue':
        assert args.rescue_from is not None, 'rescue mode requires --rescue_from'
        wd = 5e-4
        augment = False    # keep same data treatment as M to isolate effect of WD
        n_epochs = args.num_epochs or 400
        rescue_tag = Path(args.rescue_from).stem
        out_dir = Path(f'trackb/checkpoints/rescue_{rescue_tag}')

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'mode={args.mode}  wd={wd}  augment={augment}  n_epochs={n_epochs}  device={device}  out_dir={out_dir}')

    train_loader, test_loader = get_loaders(args.batch_size, augment)
    model = make_resnet18().to(device)
    if args.mode == 'rescue':
        state = torch.load(args.rescue_from, map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
        print(f'Loaded rescue checkpoint from {args.rescue_from} (epoch {state.get("epoch", "?")})')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=wd, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    history = {'epoch': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    t_start = time.time()
    for ep in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.numel()
            running_correct += (logits.argmax(dim=-1) == y).sum().item()
            running_total += y.numel()
        tr_loss = running_loss / running_total
        tr_acc = running_correct / running_total
        te_loss, te_acc = evaluate(model, test_loader, device)
        history['epoch'].append(ep)
        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        print(f'ep {ep:4d}  train_loss {tr_loss:.4f}  train_acc {tr_acc:.4f}  '
              f'test_loss {te_loss:.4f}  test_acc {te_acc:.4f}  '
              f'elapsed {time.time() - t_start:.1f}s')

        if (ep + 1) % args.save_every == 0 or ep == n_epochs - 1:
            torch.save({'model': model.state_dict(), 'epoch': ep + 1}, out_dir / f'epoch_{ep+1}.pt')

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    torch.save({'model': model.state_dict(), 'epoch': n_epochs}, out_dir / 'final.pt')
    print(f'done. history -> {out_dir / "history.json"}')


if __name__ == '__main__':
    main()
