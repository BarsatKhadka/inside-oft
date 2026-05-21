"""Dump the 1-layer transformer's M->G mode-connectivity curve to JSON.

mode_connectivity.py computes and *plots* the linear-interpolation loss curve
but saves no data. The paper figure (paper/fig.py, Panel B of fig:grok_basin)
needs the raw curve. This script writes:

    taska/results/mode_connectivity.json

with the loss along alpha in [0,1] from M weights to G weights, so the
algorithmic transformer can be drawn as a loss-landscape row alongside the
vision / LM settings.

Note: the "~10^7 barrier" quoted elsewhere is a RATIO (midpoint loss / endpoint
loss) and is inflated by the near-zero endpoint training loss. This script also
records the absolute barrier (excess loss above the straight M--G line), which
is the quantity the figure draws and the one comparable across architectures.

Run wherever torch is available:  python taska/analysis/mode_connectivity_dump.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # taska/
sys.path.insert(0, str(HERE))

import torch as t
from data import gen_train_test, to_tensors
from model import Transformer

P = 113
D_MODEL = 128
DATA_SEED = 0
N_ALPHA = 21


def load_state(p):
    return t.load(p, map_location='cpu', weights_only=True)['model']


@t.no_grad()
def eval_loss(model, inputs, labels):
    logits = model(inputs)[:, -1, :].to(t.float64)
    lp = t.nn.functional.log_softmax(logits, dim=-1)
    loss = -lp[t.arange(labels.shape[0]), labels].mean().item()
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


def main():
    s_M = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')
    s_G = load_state(HERE / 'checkpoints' / 'G' / 'final.pt')

    tr_pairs, te_pairs = gen_train_test(p=P, frac_train=0.3, seed=DATA_SEED)
    tr_in, tr_lab = to_tensors(tr_pairs, P, device='cpu')
    te_in, te_lab = to_tensors(te_pairs, P, device='cpu')

    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3,
                        num_layers=1)
    model.eval()

    alphas = [i / (N_ALPHA - 1) for i in range(N_ALPHA)]
    train_loss, test_loss, train_acc, test_acc = [], [], [], []
    for a in alphas:
        model.load_state_dict({k: (1 - a) * s_M[k] + a * s_G[k] for k in s_M})
        trl, tra = eval_loss(model, tr_in, tr_lab)
        tel, tea = eval_loss(model, te_in, te_lab)
        train_loss.append(trl); test_loss.append(tel)
        train_acc.append(tra);  test_acc.append(tea)
        print(f'  alpha={a:.2f}  train_loss={trl:.4e}  test_loss={tel:.4e}')

    # absolute barrier = peak excess above the straight endpoint line
    def abs_barrier(curve):
        L0, L1 = curve[0], curve[-1]
        excess = [curve[i] - ((1 - alphas[i]) * L0 + alphas[i] * L1)
                  for i in range(len(alphas))]
        return max(excess)

    out = {
        'tier': 'taska_1L_transformer',
        'alphas': alphas,
        'curve': [{'alpha': alphas[i], 'train_loss': train_loss[i],
                   'test_loss': test_loss[i]} for i in range(len(alphas))],
        'train_loss': train_loss, 'test_loss': test_loss,
        'train_acc': train_acc, 'test_acc': test_acc,
        'abs_barrier_train': abs_barrier(train_loss),
        'abs_barrier_test': abs_barrier(test_loss),
        'ratio_barrier_train': (train_loss[len(alphas) // 2]
                                / max(train_loss[0], train_loss[-1])),
        'note': 'abs_barrier_* is the figure-comparable barrier; '
                'ratio_barrier_train is the inflated midpoint/endpoint ratio.',
    }
    out_path = HERE / 'results' / 'mode_connectivity.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nabs barrier (test) = {out["abs_barrier_test"]:.3f}')
    print(f'abs barrier (train) = {out["abs_barrier_train"]:.3f}')
    print(f'ratio barrier (train, inflated) = {out["ratio_barrier_train"]:.2e}')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
