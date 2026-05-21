"""Experiment 3: algorithmic weight-decay sweep with Petzka (mech11 for Track A).

Trains the 1-layer modular-addition transformer from scratch at five weight
decays, three seeds each, 30k epochs, and at convergence records:

    top Hessian eigenvalue (Lanczos) of the training loss,
    weight L2 norm  ||theta||,
    Petzka relative flatness  =  lambda_max * ||theta||^2,
    whether the model grokked (test acc >= 0.95) and at which epoch.

WD grid {0.1, 0.25, 0.5, 1.0, 2.0} straddles the escape threshold WD ~ 0.376
(Entry 65): 0.1 and 0.25 below it, 0.5/1.0/2.0 above.

What it tests: does Petzka have a corresponding threshold -- a value below
which no run grokks and above which they all do? A sharp Petzka threshold at
the escape boundary is a strong mechanistic result. If Petzka instead varies
smoothly with WD and shows no threshold, it adds little beyond WD itself.

Output: taska/results/petzka_wd_sweep.json
Run:    python taska/analysis/petzka_wd_sweep.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import torch as t
import torch.optim as optim

from data import gen_train_test, to_tensors
from model import Transformer
from bulletproof3._signatures import (hessian_top_bot, weight_l2_norm,
                                      relative_flatness)

P = 113
D_MODEL = 128
LR = 1e-3
FRAC_TRAIN = 0.3
NUM_EPOCHS = 30000
LANCZOS_K = 20
WD_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0]
SEEDS = [0, 1, 2]


def ce(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1)
            == labels).float().mean().item()


def effective_rank(W):
    s = t.linalg.svdvals(W.detach().cpu())
    p = (s ** 2)
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def run(wd, seed, device):
    t.manual_seed(seed)
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3,
                         num_layers=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd,
                      betas=(0.9, 0.98))
    tr_pairs, te_pairs = gen_train_test(p=P, frac_train=FRAC_TRAIN, seed=seed)
    tr_in, tr_lab = to_tensors(tr_pairs, P, device=device)
    te_in, te_lab = to_tensors(te_pairs, P, device=device)

    grok_epoch = None
    for ep in range(NUM_EPOCHS):
        loss = ce(model(tr_in)[:, -1, :], tr_lab)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 500 == 0:
            te = eval_acc(model, te_in, te_lab)
            if grok_epoch is None and te >= 0.95:
                grok_epoch = ep + 1

    test_acc = eval_acc(model, te_in, te_lab)
    train_acc = eval_acc(model, tr_in, tr_lab)
    rank = effective_rank(model.blocks[0].mlp.W_out)
    # Petzka == relative_flatness_full: Hessian of the FULL-data loss
    # (0.5*(train+test)). The train-only Hessian degenerates to ~0 on an
    # interpolating model -- recorded for transparency.
    loss_full = lambda: 0.5 * (ce(model(tr_in)[:, -1, :], tr_lab)
                               + ce(model(te_in)[:, -1, :], te_lab))
    loss_train = lambda: ce(model(tr_in)[:, -1, :], tr_lab)
    top, bot, _ = hessian_top_bot(model, loss_full, k=LANCZOS_K)
    top_train, _bt, _ = hessian_top_bot(model, loss_train, k=LANCZOS_K)
    wn = weight_l2_norm(model)
    petzka = relative_flatness(top, wn)
    grokked = test_acc >= 0.95
    print(f'  wd={wd:<5} seed={seed} test={test_acc:.4f} grok={grokked} '
          f'(@{grok_epoch}) rank={rank:6.2f} top_full={top:9.2f} '
          f'top_train={top_train:7.2f} ||th||={wn:7.2f} petzka={petzka:11.1f}')
    return {'wd': wd, 'seed': seed, 'test_acc': test_acc,
            'train_acc': train_acc, 'rank_W_out': rank, 'top_eig_full': top,
            'top_eig_train': top_train, 'bot_eig': bot, 'theta_norm': wn,
            'petzka': petzka, 'grokked': grokked, 'grok_epoch': grok_epoch}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    out_path = HERE / 'results' / 'petzka_wd_sweep.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for wd in WD_VALUES:
        print(f'\n=== WD = {wd} ===')
        for seed in SEEDS:
            results.append(run(wd, seed, device))
            with open(out_path, 'w') as f:
                json.dump({'escape_threshold_ref': 0.376,
                           'wd_values': WD_VALUES, 'runs': results}, f,
                          indent=2)

    print('\n=== summary (mean over seeds) ===')
    for wd in WD_VALUES:
        rs = [r for r in results if r['wd'] == wd]
        n = len(rs)
        m = lambda k: sum(r[k] for r in rs) / n
        ng = sum(r['grokked'] for r in rs)
        print(f'  WD={wd:<5} grokked {ng}/{n}  test={m("test_acc"):.3f}  '
              f'top_full={m("top_eig_full"):.1f}  '
              f'||th||={m("theta_norm"):.2f}  petzka={m("petzka"):.1f}')
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
