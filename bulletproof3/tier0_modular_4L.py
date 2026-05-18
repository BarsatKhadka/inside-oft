"""Tier 0: 4-layer Transformer on (a+b) mod 113.
3 signatures (rank, Hessian top/bot, gradient angle) at depth.
5 seeds M (wd=0) and 5 seeds G (wd=1.0).
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.optim as optim

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from taska.data import gen_train_test, to_tensors
from taska.model import Transformer
from bulletproof3._signatures import compute_full_battery, effective_rank, all_ranks

P = 113
NUM_EPOCHS = 25000
LR = 1e-3
NUM_SEEDS = 5


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


@t.no_grad()
def per_example_loss(model, inp, lab):
    logits = model(inp)[:, -1, :]
    return t.nn.functional.cross_entropy(logits, lab, reduction='none').cpu().numpy()


@t.no_grad()
def acc(model, inp, lab):
    return (model(inp)[:, -1, :].argmax(dim=-1) == lab).float().mean().item()


def run(seed, wd, device):
    t.manual_seed(seed); np.random.seed(seed)
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=4).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))
    tr, te = gen_train_test(p=P, frac_train=0.3, seed=seed)
    tr_in, tr_lab = to_tensors(tr, P, device=device)
    te_in, te_lab = to_tensors(te, P, device=device)
    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(tr_in)[:, -1, :], tr_lab)
        opt.zero_grad(); loss.backward(); opt.step()
    train_loss_fn = lambda: cross_entropy_hp(model(tr_in)[:, -1, :], tr_lab)
    test_loss_fn = lambda: cross_entropy_hp(model(te_in)[:, -1, :], te_lab)
    train_losses = per_example_loss(model, tr_in, tr_lab)
    test_losses = per_example_loss(model, te_in, te_lab)
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                train_losses, test_losses,
                                lanczos_k=20, verbose=True)
    bat['train_acc'] = acc(model, tr_in, tr_lab)
    bat['test_acc'] = acc(model, te_in, te_lab)
    bat['seed'] = seed; bat['wd'] = wd
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    out_path = HERE / 'results' / 'tier0_modular_4L.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for label, wd in [('M', 0.0), ('G', 1.0)]:
        for seed in range(NUM_SEEDS):
            print(f'\n=== {label} seed={seed} ===')
            try:
                entry = run(seed, wd, device)
                results[label].append(entry)
                print(f'  test_acc={entry["test_acc"]:.4f}, '
                      f'top_eig={entry["hessian_top_full"]:.3f}, '
                      f'bot_eig={entry["hessian_bot_full"]:.3f}, '
                      f'cos={entry["cos_grad_train_test"]:.4f}')
            except Exception as e:
                print(f'  error: {e}')
                results[label].append({'seed': seed, 'wd': wd, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
