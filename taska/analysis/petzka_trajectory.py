"""Experiment 1: Petzka relative flatness along the grokking trajectory.

NO NEW TRAINING. Loads every saved checkpoint of the existing M and G runs
(taska/checkpoints/{M,M_seed1,M_seed2,G,G_seed1,G_seed2}) and computes:

    top Hessian eigenvalue (Lanczos) of the TRAINING loss,
    weight L2 norm  ||theta||,
    Petzka relative flatness  =  lambda_max * ||theta||^2.

Petzka eigenvalue and weight-norm use the shared bulletproof3/_signatures.py
implementation, so the numbers are directly comparable to the relative_flatness
values already reported for the vision/LM tiers.

Pairs with the effective-rank trajectory: shows whether Petzka flatness moves
through the pre-grokking plateau the way rank does.

Output: taska/results/petzka_trajectory.json
Run:    python taska/analysis/petzka_trajectory.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # taska/
ROOT = HERE.parent                                      # repo root
sys.path.insert(0, str(HERE))                           # data, model
sys.path.insert(0, str(ROOT))                           # bulletproof3

import torch as t
from data import gen_train_test, to_tensors
from model import Transformer
from bulletproof3._signatures import (hessian_top_bot, weight_l2_norm,
                                      relative_flatness)

P = 113
D_MODEL = 128
LANCZOS_K = 12
SEED_DIRS = ['M', 'M_seed1', 'M_seed2', 'G', 'G_seed1', 'G_seed2']
CKPT = HERE / 'checkpoints'


def read_cfg(d):
    """Minimal config.yaml reader (avoids a yaml dependency)."""
    cfg = {}
    for line in (d / 'config.yaml').read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            cfg[k.strip()] = v.strip()
    return cfg


def ce(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def epochs_for(d):
    eps = sorted(int(p.stem.split('_')[1]) for p in d.glob('epoch_*.pt'))
    if (d / 'init.pt').exists():
        eps = [0] + eps
    if (d / 'final.pt').exists():
        eps = eps + [50000]
    return eps


def state_for(d, ep):
    f = (d / 'init.pt') if ep == 0 else \
        (d / 'final.pt') if ep == 50000 else (d / f'epoch_{ep}.pt')
    return t.load(f, map_location='cpu', weights_only=True)['model']


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    out = {'lanczos_k': LANCZOS_K, 'runs': {}}
    out_path = HERE / 'results' / 'petzka_trajectory.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for name in SEED_DIRS:
        d = CKPT / name
        if not d.exists():
            print(f'skip missing {name}')
            continue
        cfg = read_cfg(d)
        frac = float(cfg.get('frac_train', 0.3))
        seed = int(cfg.get('seed', 0))
        tr_pairs, _ = gen_train_test(p=P, frac_train=frac, seed=seed)
        tr_in, tr_lab = to_tensors(tr_pairs, P, device=device)

        model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3,
                             num_layers=1).to(device)
        eps = epochs_for(d)
        rec = {'seed': seed, 'frac_train': frac, 'epochs': eps,
               'top_eig': [], 'theta_norm': [], 'petzka': []}
        print(f'\n=== {name} (seed={seed}, frac={frac}, {len(eps)} ckpts) ===')
        for ep in eps:
            model.load_state_dict(state_for(d, ep))
            loss_fn = lambda: ce(model(tr_in)[:, -1, :], tr_lab)
            top, _bot, _ = hessian_top_bot(model, loss_fn, k=LANCZOS_K)
            wn = weight_l2_norm(model)
            pz = relative_flatness(top, wn)
            rec['top_eig'].append(top)
            rec['theta_norm'].append(wn)
            rec['petzka'].append(pz)
            print(f'  ep={ep:>6}: top={top:>10.3f}  ||theta||={wn:>8.3f}  '
                  f'petzka={pz:>12.1f}')
        out['runs'][name] = rec
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)

    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
