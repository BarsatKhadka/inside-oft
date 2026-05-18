"""Task complexity vs rank: does the minimum generalizing rank depend on the task?

Sharp claim: there exists a task-specific minimum rank r*(task) such that any
model with effective rank below r* cannot represent the task generalizingly.
Different tasks have different r*. Generalizing models converge to rank ~ r*.

Tasks to test (all mod 113):
  - add:     c = (a + b) mod p     (current Track A task)
  - subtract: c = (a - b) mod p
  - mult:    c = (a * b) mod p     (harder?)
  - square:  c = (a^2 + b) mod p
  - poly:    c = (a^2 + a*b + b^2) mod p

For each task, train a fresh G-style model (WD=1.0) for 50k epochs.
Measure: converged W_out effective rank.

If rank correlates with task complexity → generalizes to "rank = task complexity"
hypothesis. Could even predict required rank from theory.

Usage:
    python taska/analysis/task_complexity_rank.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import json
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.optim as optim

from model import Transformer

P = 113
D_MODEL = 128
LR = 1e-3
WD = 1.0
NUM_EPOCHS = 30000
LOG_EVERY = 500
SEEDS = [0, 1, 2]

TASKS = {
    'add':       lambda a, b: (a + b) % P,
    'subtract':  lambda a, b: (a - b) % P,
    'mult':      lambda a, b: (a * b) % P,
    'square_plus_b': lambda a, b: (a * a + b) % P,
    'poly_quad': lambda a, b: (a * a + a * b + b * b) % P,
}


def gen_task_data(fn, seed, device, frac_train=0.3):
    pairs = [(i, j) for i in range(P) for j in range(P)]
    import random as r
    r.seed(seed)
    r.shuffle(pairs)
    div = int(frac_train * len(pairs))
    train, test = pairs[:div], pairs[div:]
    def to_t(ps):
        inp = t.tensor([(a, b, P) for a, b in ps], dtype=t.long, device=device)
        lab = t.tensor([fn(a, b) for a, b in ps], device=device)
        return inp, lab
    return to_t(train), to_t(test)


def effective_rank(W):
    s = t.linalg.svdvals(W.detach().cpu())
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def train_task(task_name, fn, seed, device):
    print(f'\n=== task={task_name}, seed={seed} ===')
    t.manual_seed(seed)
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))

    (train_in, train_lab), (test_in, test_lab) = gen_task_data(fn, seed, device)

    history = {'epoch': [], 'test_acc': [], 'train_loss': [],
               'rank_W_out': [], 'rank_W_in': [], 'rank_W_E': []}
    grok_epoch = None

    for ep in range(NUM_EPOCHS):
        loss = cross_entropy_hp(model(train_in)[:, -1, :], train_lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % LOG_EVERY == 0:
            te = eval_acc(model, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
            history['train_loss'].append(loss.item())
            history['rank_W_out'].append(effective_rank(model.blocks[0].mlp.W_out))
            history['rank_W_in'].append(effective_rank(model.blocks[0].mlp.W_in))
            history['rank_W_E'].append(effective_rank(model.embed.W_E[:, :P]))
            if grok_epoch is None and te >= 0.95:
                grok_epoch = ep + 1
                print(f'  GROK at {grok_epoch}, rank_W_out={history["rank_W_out"][-1]:.2f}')

    final = {
        'final_test_acc': history['test_acc'][-1],
        'grok_epoch': grok_epoch,
        'final_rank_W_out': history['rank_W_out'][-1],
        'final_rank_W_in': history['rank_W_in'][-1],
        'final_rank_W_E': history['rank_W_E'][-1],
    }
    print(f'  result: final_test_acc={final["final_test_acc"]:.4f}, grok@{grok_epoch}, '
          f'W_out_rank={final["final_rank_W_out"]:.2f}')
    return history, final


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    all_results = {}
    for task_name, fn in TASKS.items():
        all_results[task_name] = {}
        for seed in SEEDS:
            history, final = train_task(task_name, fn, seed, device)
            all_results[task_name][seed] = {'history': history, 'final': final}

    out_json = HERE / 'results' / 'task_complexity_rank.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({tn: {str(s): v for s, v in d.items()} for tn, d in all_results.items()}, f)

    # Aggregate
    task_names = list(TASKS.keys())
    mean_ranks_W_out = []
    std_ranks_W_out = []
    mean_grok = []
    final_accs = []
    for tn in task_names:
        ranks = [all_results[tn][s]['final']['final_rank_W_out'] for s in SEEDS]
        mean_ranks_W_out.append(np.mean(ranks))
        std_ranks_W_out.append(np.std(ranks))
        groks = [all_results[tn][s]['final']['grok_epoch'] or NUM_EPOCHS + 1 for s in SEEDS]
        mean_grok.append(np.mean(groks))
        final_accs.append(np.mean([all_results[tn][s]['final']['final_test_acc'] for s in SEEDS]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    x = np.arange(len(task_names))
    ax.bar(x, mean_ranks_W_out, yerr=std_ranks_W_out, capsize=4, color='tab:blue')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=20)
    ax.set_ylabel('converged W_out effective rank (mean ± std)')
    ax.set_title('Generalizing model rank vs task')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x, mean_grok, color='tab:red')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=20)
    ax.set_ylabel('mean epoch to grok (95% test acc)')
    ax.set_title('Time to grok vs task')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Does rank depend on task complexity?')
    fig.tight_layout()
    out = HERE / 'results' / 'fig_task_complexity_rank.png'
    fig.savefig(out, dpi=130)
    print(f'\nplot -> {out}')

    print('\n=== Per-task converged rank ===')
    for i, tn in enumerate(task_names):
        print(f'  {tn:20s}: rank={mean_ranks_W_out[i]:.2f}±{std_ranks_W_out[i]:.2f}, '
              f'grok_epoch={mean_grok[i]:.0f}, final_acc={final_accs[i]:.4f}')


if __name__ == '__main__':
    main()
