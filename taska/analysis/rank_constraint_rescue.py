"""KEY TEST: does FORCING low rank (WITHOUT weight decay) escape the saddle?

If rank compression IS the mechanism (not just WD's side effect), then
artificially forcing M's weights to low rank should escape the saddle even
without WD.

For each step:
  1. Standard gradient step (no WD)
  2. Project W_in, W_out (and optionally W_E) onto top-k SVD components

For each k in {15, 20, 30, 50}, run rescue starting from M_50000 with rank
projection at every step. Measure: does it grok?

If yes → rank compression is sufficient; WD's mechanism is verified.
If no → there's more to WD than just rank, our story needs refinement.

Usage:
    python taska/analysis/rank_constraint_rescue.py
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

from data import gen_train_test, to_tensors
from model import Transformer

P = 113
D_MODEL = 128
SEED = 0
LR = 1e-3
NUM_EPOCHS = 20000
LOG_EVERY = 500
K_VALUES = [15, 20, 30, 50]


def load_state(p):
    return t.load(p, map_location='cpu', weights_only=True)['model']


def make_model(state, device):
    m = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    m.load_state_dict(state)
    m.to(device)
    return m


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def full_loss(model, inputs, labels):
    return cross_entropy_hp(model(inputs)[:, -1, :], labels)


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def project_to_rank(W, k):
    U, S, Vt = t.linalg.svd(W, full_matrices=False)
    k_use = min(k, S.shape[0])
    return U[:, :k_use] @ t.diag(S[:k_use]) @ Vt[:k_use, :]


def run_with_rank_constraint(k, M_state, device, train_in, train_lab, test_in, test_lab):
    print(f'\n=== rank_constraint k={k} ===')
    model = make_model(M_state, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))

    history = {'epoch': [], 'test_acc': [], 'train_loss': []}

    for ep in range(NUM_EPOCHS):
        train_loss = full_loss(model, train_in, train_lab)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # After each step, project W_in, W_out, W_E to rank k
        with t.no_grad():
            model.blocks[0].mlp.W_in.copy_(project_to_rank(model.blocks[0].mlp.W_in, k))
            model.blocks[0].mlp.W_out.copy_(project_to_rank(model.blocks[0].mlp.W_out, k))
            W_E_part = model.embed.W_E[:, :P]
            model.embed.W_E[:, :P] = project_to_rank(W_E_part, k)

        if (ep + 1) % LOG_EVERY == 0:
            tr = train_loss.item()
            te = eval_acc(model, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['test_acc'].append(te)
            history['train_loss'].append(tr)
        if (ep + 1) % 4000 == 0:
            print(f'  ep={ep+1:5d}: train_loss={tr:.4e}, test_acc={te:.4f}')

    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    M_state = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    histories = {}
    for k in K_VALUES:
        histories[k] = run_with_rank_constraint(k, M_state, device, train_in, train_lab, test_in, test_lab)

    out_json = HERE / 'results' / 'rank_constraint_rescue.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({str(k): v for k, v in histories.items()}, f)
    print(f'\nhistories -> {out_json}')

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(K_VALUES)))
    for color, k in zip(cmap, K_VALUES):
        h = histories[k]
        ax.plot(h['epoch'], h['test_acc'], marker='.', label=f'project_to_rank_{k}', color=color)
    ax.set_xlabel('rescue epoch (no WD, only rank projection)')
    ax.set_ylabel('test accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Does forcing low rank (without WD) escape the saddle?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = HERE / 'results' / 'fig_rank_constraint_rescue.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')

    print('\n=== Outcomes ===')
    for k in K_VALUES:
        h = histories[k]
        grok = next((h['epoch'][i] for i, a in enumerate(h['test_acc']) if a >= 0.95), None)
        print(f'  k={k}: final={h["test_acc"][-1]:.4f}, grok @ {grok}')


if __name__ == '__main__':
    main()
