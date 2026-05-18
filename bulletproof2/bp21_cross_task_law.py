"""bp21: Cross-task WD-rank law.

WD sweep on subtract and multiply. Does the law form generalize across ops?
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.optim as optim

from _common import (HERE, P, LR, cross_entropy_hp, effective_rank, eval_acc)
from taska.data import gen_train_test
from taska.model import Transformer

NUM_SEEDS = 5
NUM_EPOCHS = 20000
WD_VALUES = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
OPS = {
    'subtract': lambda a, b: (a - b) % P,
    'multiply': lambda a, b: (a * b) % P,
}


def to_tensors_op(pairs, fn, device):
    inp = t.tensor([(a, b, P) for a, b, _ in pairs], dtype=t.long, device=device)
    lab = t.tensor([fn(a, b) for a, b, _ in pairs], device=device)
    return inp, lab


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = []
    for op_name, fn in OPS.items():
        for wd in WD_VALUES:
            for seed in range(NUM_SEEDS):
                print(f'\n--- {op_name} wd={wd} seed={seed} ---')
                t.manual_seed(seed); np.random.seed(seed)
                model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
                opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))
                tr, te = gen_train_test(p=P, frac_train=0.3, seed=seed)
                tr_in, tr_lab = to_tensors_op(tr, fn, device)
                te_in, te_lab = to_tensors_op(te, fn, device)
                for ep in range(NUM_EPOCHS):
                    loss = cross_entropy_hp(model(tr_in)[:, -1, :], tr_lab)
                    opt.zero_grad(); loss.backward(); opt.step()
                te_a = eval_acc(model, te_in, te_lab)
                rank = effective_rank(model.blocks[0].mlp.W_out)
                entry = {
                    'op': op_name, 'wd': wd, 'seed': seed,
                    'final_test_acc': te_a, 'rank_W_out': rank,
                }
                results.append(entry)
                print(f'  test={te_a:.4f}, rank={rank:.2f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp21_cross_task_law.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Per-op fits
    for op in OPS:
        grokked = [r for r in results if r['op'] == op and r['final_test_acc'] > 0.9 and r['wd'] > 0]
        if len(grokked) > 4:
            x = np.log(np.array([r['wd'] for r in grokked]))
            y = np.log(np.array([r['rank_W_out'] for r in grokked]))
            slope, intercept = np.polyfit(x, y, 1)
            print(f'{op}: log(rank) = {slope:.3f}*log(WD) + {intercept:.3f}')


if __name__ == '__main__':
    main()
