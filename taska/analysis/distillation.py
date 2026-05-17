"""β. Distillation from M: does M as a teacher help a fresh student learn faster?

Train a fresh student on the same task (a+b) mod p, with the loss:
    L = CE(student_output, true_label) + λ * KL(student_output || M_output)

M's outputs are wrong on test, but its activations preserve (a, b) at high
fidelity. If M's "output distribution" -- including its confidently-wrong test
predictions -- somehow encodes useful structure, the student might learn faster
than without distillation.

Sweep λ ∈ {0, 0.1, 0.5, 1.0, 2.0}. Compare time to grok.

Possible outcomes:
  - λ > 0 accelerates grokking: M's outputs carry useful information beyond
    just the right answers. Surprising and interesting.
  - λ > 0 slows grokking or fails to grok: M's outputs are misleading,
    distillation hurts. Confirms M as "useless teacher."
  - λ doesn't matter: structure of distillation is too weak to affect training.

Usage:
    python taska/analysis/distillation.py
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
WEIGHT_DECAY = 1.0
NUM_EPOCHS = 20000
LOG_EVERY = 200
LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0]


def load_state(p):
    return t.load(p, map_location='cpu', weights_only=True)['model']


def make_model(device):
    m = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    m.to(device)
    return m


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


def kl_div(student_logits, teacher_logits):
    """KL(student || teacher) where both are over the vocab dimension."""
    s_lp = t.nn.functional.log_softmax(student_logits.to(t.float64), dim=-1)
    t_p  = t.nn.functional.softmax(teacher_logits.to(t.float64),  dim=-1)
    # KL(P||Q) = sum P log P/Q.  We want student==P, teacher==Q? Actually conv
    # is: KL(s || t) = sum s log s/t.  Here we want student to learn from
    # teacher distribution -> minimize KL(student || teacher). Use:
    return (t.nn.functional.softmax(student_logits.to(t.float64), dim=-1) * (s_lp - t.log(t_p + 1e-12))).sum(dim=-1).mean()


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def run_config(lam, device, teacher_state):
    print(f'\n=== distillation with λ={lam} ===')

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    # Teacher predictions (precomputed, frozen)
    teacher = make_model(device)
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    with t.no_grad():
        teacher_logits_train = teacher(train_in)[:, -1, :]   # (N, vocab)

    # Student (fresh init)
    t.manual_seed(42)
    student = make_model(device)
    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))

    history = {'epoch': [], 'train_acc': [], 'test_acc': [], 'ce_loss': [], 'kl_loss': []}

    for ep in range(NUM_EPOCHS):
        student_logits = student(train_in)[:, -1, :]
        ce_loss = cross_entropy_hp(student_logits, train_lab)
        if lam > 0:
            kl = kl_div(student_logits, teacher_logits_train)
            loss = ce_loss + lam * kl
        else:
            kl = t.tensor(0.0)
            loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % LOG_EVERY == 0:
            tr = eval_acc(student, train_in, train_lab)
            te = eval_acc(student, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['train_acc'].append(tr)
            history['test_acc'].append(te)
            history['ce_loss'].append(ce_loss.item())
            history['kl_loss'].append(kl.item())
        if (ep + 1) % 2000 == 0:
            print(f'  ep={ep+1:5d}  train_acc={tr:.4f}  test_acc={te:.4f}  ce={ce_loss.item():.4e}  kl={kl.item():.4e}')

    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    teacher_state = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')

    histories = {}
    for lam in LAMBDAS:
        histories[lam] = run_config(lam, device, teacher_state)

    out_json = HERE / 'results' / 'distillation.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({str(k): v for k, v in histories.items()}, f)
    print(f'\nhistories -> {out_json}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(LAMBDAS)))
    for color, lam in zip(cmap, LAMBDAS):
        h = histories[lam]
        axes[0].plot(h['epoch'], h['test_acc'], label=f'λ={lam}', color=color)
        axes[1].plot(h['epoch'], h['train_acc'], label=f'λ={lam}', color=color)
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('test accuracy')
    axes[0].set_title('Student test accuracy with M-distillation')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('train accuracy')
    axes[1].set_title('Student train accuracy with M-distillation')
    axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(-0.05, 1.05)
    fig.suptitle("Distillation from M: does it accelerate or hinder student grokking?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_distillation.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')

    # Summary
    print('\n=== Summary: epoch at which test_acc first >= 0.95 ===')
    for lam in LAMBDAS:
        h = histories[lam]
        ep_reach = next((h['epoch'][i] for i, a in enumerate(h['test_acc']) if a >= 0.95), None)
        print(f'  λ={lam}: {"NEVER" if ep_reach is None else f"epoch {ep_reach}"}')


if __name__ == '__main__':
    main()
