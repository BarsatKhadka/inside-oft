"""Dump the multi-seed effective-rank trajectory to JSON for the paper figure.

Computes the effective rank (entropy of normalized SQUARED singular values --
the definition used everywhere else in the study) of W_E, W_in, W_out at every
saved checkpoint, for all M seeds and all G seeds, and writes:

    taska/results/rank_trajectory.json

paper/fig.py picks this file up automatically and redraws the grokking panel
with a mean line and a per-seed min/max band. Run wherever torch is available:

    python taska/analysis/rank_trajectory_dump.py
"""
import json
import sys
from pathlib import Path

import torch as t

HERE = Path(__file__).resolve().parent.parent          # taska/
CKPT = HERE / 'checkpoints'
OUT  = HERE / 'results' / 'rank_trajectory.json'
P = 113

M_SEEDS = ['M', 'M_seed1', 'M_seed2']
G_SEEDS = ['G', 'G_seed1', 'G_seed2']
MATRICES = [('W_E', 'embed.W_E'), ('W_in', 'blocks.0.mlp.W_in'),
            ('W_out', 'blocks.0.mlp.W_out')]


def effective_rank(W):
    s = t.linalg.svdvals(W.float())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def load_state(path):
    return t.load(path, map_location='cpu', weights_only=True)['model']


def epochs_for(d):
    eps = sorted(int(p.stem.split('_')[1]) for p in d.glob('epoch_*.pt'))
    if (d / 'init.pt').exists():
        eps = [0] + eps
    if (d / 'final.pt').exists():
        eps = eps + [50000]
    return eps


def state_for(d, ep):
    if ep == 0:
        return load_state(d / 'init.pt')
    if ep == 50000:
        return load_state(d / 'final.pt')
    return load_state(d / f'epoch_{ep}.pt')


def main():
    # epochs present in every seed directory
    all_dirs = [CKPT / s for s in M_SEEDS + G_SEEDS if (CKPT / s).exists()]
    if not all_dirs:
        sys.exit('No checkpoint directories found under taska/checkpoints/.')
    common = set(epochs_for(all_dirs[0]))
    for d in all_dirs[1:]:
        common &= set(epochs_for(d))
    epochs = sorted(common)
    print(f'{len(epochs)} common epochs across {len(all_dirs)} seed dirs')

    out = {'epochs': epochs,
           'note': 'effective rank = exp(entropy of normalized squared '
                   'singular values); multi-seed.'}
    for name, _ in MATRICES:
        out[name] = {'M': [], 'G': []}

    for group, seeds in (('M', M_SEEDS), ('G', G_SEEDS)):
        for s in seeds:
            d = CKPT / s
            if not d.exists():
                print(f'  skip missing seed dir {s}')
                continue
            per = {name: [] for name, _ in MATRICES}
            for ep in epochs:
                st = state_for(d, ep)
                for name, key in MATRICES:
                    W = st[key]
                    if name == 'W_E':
                        W = W[:, :P]
                    per[name].append(effective_rank(W))
            for name, _ in MATRICES:
                out[name][group].append(per[name])
            print(f'  {s}: done')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
