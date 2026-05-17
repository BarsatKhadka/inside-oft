"""Train a 1-layer transformer on modular addition.

Usage:
    python train.py --config configs/G.yaml
    python train.py --config configs/M.yaml
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch as t
import torch.optim as optim
import yaml

from data import gen_train_test, to_tensors
from model import Transformer


def cross_entropy_high_precision(logits, labels):
    # Cast to float64 because float32 log-softmax loses precision past grokking
    logprobs = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -logprobs[t.arange(labels.shape[0]), labels].mean()


def full_loss(model, inputs, labels):
    logits = model(inputs)[:, -1]  # predict at the "=" position
    return cross_entropy_high_precision(logits, labels)


def accuracy(model, inputs, labels):
    with t.no_grad():
        logits = model(inputs)[:, -1]
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print(f'config: {cfg}')

    # Seed everything
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    t.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        t.cuda.manual_seed_all(cfg['seed'])

    # Data
    train_pairs, test_pairs = gen_train_test(
        p=cfg['p'], frac_train=cfg['frac_train'], seed=cfg['seed']
    )
    train_in, train_lab = to_tensors(train_pairs, cfg['p'], device)
    test_in, test_lab = to_tensors(test_pairs, cfg['p'], device)
    print(f'train: {len(train_pairs)}   test: {len(test_pairs)}')

    # Model
    model = Transformer(
        p=cfg['p'],
        d_model=cfg['d_model'],
        num_heads=cfg['num_heads'],
        n_ctx=3,
        num_layers=cfg['num_layers'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params:,}')

    # Optimizer + LR warmup (first 10 steps)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        betas=(0.9, 0.98),
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(s / 10, 1.0))

    # Output dir
    out_dir = Path(cfg['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'saving checkpoints to {out_dir}')

    # Save initial checkpoint + config
    t.save({'model': model.state_dict(), 'epoch': 0}, out_dir / 'init.pt')
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f)

    history = {'epoch': [], 'train_loss': [], 'test_loss': [],
               'train_acc': [], 'test_acc': []}

    log_every = cfg.get('log_every', 100)
    save_every = cfg.get('save_every', 1000)

    for epoch in range(cfg['num_epochs']):
        train_loss = full_loss(model, train_in, train_lab)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == cfg['num_epochs'] - 1:
            test_loss = full_loss(model, test_in, test_lab)
            tr_acc = accuracy(model, train_in, train_lab)
            te_acc = accuracy(model, test_in, test_lab)
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss.item())
            history['test_loss'].append(test_loss.item())
            history['train_acc'].append(tr_acc)
            history['test_acc'].append(te_acc)
            print(f'epoch {epoch:6d}  '
                  f'train_loss {train_loss.item():.4e}  '
                  f'test_loss {test_loss.item():.4e}  '
                  f'train_acc {tr_acc:.4f}  test_acc {te_acc:.4f}')

        if epoch > 0 and epoch % save_every == 0:
            t.save({'model': model.state_dict(), 'epoch': epoch},
                   out_dir / f'epoch_{epoch}.pt')

    # Final save
    t.save({'model': model.state_dict(), 'epoch': cfg['num_epochs']},
           out_dir / 'final.pt')
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    print(f'done. history -> {out_dir / "history.json"}')


if __name__ == '__main__':
    main()
