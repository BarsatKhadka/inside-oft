"""Experiment 2: Petzka over the (weight decay x train fraction) phase diagram.

THE FALSIFICATION TEST. phase_diagram.py recorded only test_acc and rank; it
did NOT save model weights, so this script retrains the same 24-cell grid
(1-layer transformer, modular addition, 20k epochs/cell, single seed -- exactly
the phase_diagram.py protocol) and additionally measures, at convergence:

    top Hessian eigenvalue (Lanczos) of the training loss,
    weight L2 norm  ||theta||,
    Petzka relative flatness  =  lambda_max * ||theta||^2.

Each cell's final weights are saved this time (taska/checkpoints/phase/), so
the measurement need never be repeated.

What it decides: if two cells with near-identical test accuracy (e.g.
WD=0,frac=0.9 at ~99.8% vs WD=1.0,frac=0.9 at ~100%) have very different
Petzka values, then Petzka measures COMPRESSION, not generalization -- it
fails as a generalization indicator even if it orders M vs G correctly.

Output: taska/results/petzka_phase_diagram.json
        taska/results/fig_petzka_phase_diagram.png
Run:    python taska/analysis/petzka_phase_diagram.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:                              # matplotlib optional
    plt = None
import torch as t
import torch.optim as optim

from data import gen_train_test, to_tensors
from model import Transformer
from bulletproof3._signatures import (hessian_top_bot, weight_l2_norm,
                                      relative_flatness)

P = 113
D_MODEL = 128
SEED = 0
LR = 1e-3
NUM_EPOCHS = 20000
LANCZOS_K = 20
WD_VALUES = [0.0, 0.01, 0.1, 1.0]
FRAC_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


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


def run_cell(wd, frac, device):
    t.manual_seed(0)
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3,
                         num_layers=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd,
                      betas=(0.9, 0.98))
    tr_pairs, te_pairs = gen_train_test(p=P, frac_train=frac, seed=SEED)
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
    # (0.5*(train+test)), matching the vision-tier convention. The train-only
    # Hessian degenerates to ~0 on an interpolating model -- recorded for
    # transparency.
    loss_full = lambda: 0.5 * (ce(model(tr_in)[:, -1, :], tr_lab)
                               + ce(model(te_in)[:, -1, :], te_lab))
    loss_train = lambda: ce(model(tr_in)[:, -1, :], tr_lab)
    top, bot, _ = hessian_top_bot(model, loss_full, k=LANCZOS_K)
    top_train, _bt, _ = hessian_top_bot(model, loss_train, k=LANCZOS_K)
    wn = weight_l2_norm(model)
    petzka = relative_flatness(top, wn)

    # save weights so this never has to be recomputed
    ck_dir = HERE / 'checkpoints' / 'phase'
    ck_dir.mkdir(parents=True, exist_ok=True)
    t.save({'model': model.state_dict()},
           ck_dir / f'wd{wd}_frac{frac}.pt')

    print(f'  wd={wd:<5} frac={frac:<4} test={test_acc:.4f} rank={rank:6.2f} '
          f'top_full={top:9.2f} top_train={top_train:7.2f} ||th||={wn:7.2f} '
          f'petzka={petzka:11.1f} grok@{grok_epoch}')
    return {'test_acc': test_acc, 'train_acc': train_acc, 'rank_W_out': rank,
            'top_eig_full': top, 'top_eig_train': top_train, 'bot_eig': bot,
            'theta_norm': wn, 'petzka': petzka, 'grok_epoch': grok_epoch}


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    out_path = HERE / 'results' / 'petzka_phase_diagram.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid = {}
    for wd in WD_VALUES:
        for frac in FRAC_VALUES:
            print(f'\n--- wd={wd}, frac={frac} ---')
            grid[f'{wd}_{frac}'] = run_cell(wd, frac, device)
            with open(out_path, 'w') as f:
                json.dump(grid, f, indent=2)

    # Petzka vs test accuracy, coloured by weight decay -- the falsification plot
    if plt is None:
        print(f'\nwrote {out_path}')
        print('matplotlib unavailable; skipping plot. Build it later from the '
              'JSON (e.g. via paper/fig.py).')
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cmap = plt.cm.viridis
    for i, wd in enumerate(WD_VALUES):
        xs = [grid[f'{wd}_{fr}']['test_acc'] for fr in FRAC_VALUES]
        ys = [grid[f'{wd}_{fr}']['petzka'] for fr in FRAC_VALUES]
        ax.scatter(xs, ys, s=130, color=cmap(i / (len(WD_VALUES) - 1)),
                   edgecolor='0.3', label=f'WD = {wd:g}', zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel('test accuracy', fontsize=12)
    ax.set_ylabel('Petzka relative flatness  ($\\lambda_{\\max}\\,'
                  '\\|\\theta\\|^2$)', fontsize=12)
    ax.set_title('Does Petzka track generalization, or compression?',
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(title='weight decay', fontsize=9)
    fig.tight_layout()
    fig_path = HERE / 'results' / 'fig_petzka_phase_diagram.png'
    fig.savefig(fig_path, dpi=150)
    print(f'\nwrote {out_path}\nwrote {fig_path}')


if __name__ == '__main__':
    main()
