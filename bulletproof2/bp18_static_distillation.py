"""bp18: Static distillation control.

Pre-compute G's training-set softmax (frozen). Train a fresh student (no WD)
against this static target. Compare:
  - Static distill: softmax-target only (no live teacher, no WD)
  - Live distill: teacher forward each step (Entry 26 style)
  - Hard-label only (M baseline)
  - G (with WD)

If static distill recovers G's structural signatures, soft labels ARE the
mechanism. If only live distill recovers them, the teacher's grad signal matters.
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.optim as optim
import torch.nn.functional as F

from _common import (HERE, P, LR, cross_entropy_hp, effective_rank, eval_acc, train_one)
from taska.data import gen_train_test, to_tensors
from taska.model import Transformer

NUM_SEEDS = 3
NUM_EPOCHS = 20000
T_DISTILL = 4.0


def train_distill(seed, teacher_logits, tr_in, tr_lab, te_in, te_lab, mode, teacher=None, device='cuda'):
    """Train student on (hard, soft) or (hard, live teacher) target."""
    t.manual_seed(seed + 999); np.random.seed(seed + 999)
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))
    soft_static = F.softmax(teacher_logits / T_DISTILL, dim=-1) if teacher_logits is not None else None
    for ep in range(NUM_EPOCHS):
        logits = model(tr_in)[:, -1, :]
        loss = cross_entropy_hp(logits, tr_lab)
        if mode == 'static':
            log_s = F.log_softmax(logits / T_DISTILL, dim=-1)
            loss = loss + (T_DISTILL ** 2) * F.kl_div(log_s, soft_static, reduction='batchmean')
        elif mode == 'live':
            with t.no_grad():
                t_logits = teacher(tr_in)[:, -1, :]
                soft_live = F.softmax(t_logits / T_DISTILL, dim=-1)
            log_s = F.log_softmax(logits / T_DISTILL, dim=-1)
            loss = loss + (T_DISTILL ** 2) * F.kl_div(log_s, soft_live, reduction='batchmean')
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 5000 == 0:
            te = eval_acc(model, te_in, te_lab)
            print(f'  ep={ep+1}: test={te:.4f}')
    return model


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {}
    for seed in range(NUM_SEEDS):
        print(f'\n=== seed={seed} ===')
        # 1) Train G as teacher
        print(' Train G (teacher)')
        G, meta_g, (tr_in, tr_lab, te_in, te_lab) = train_one(
            seed=seed, wd=1.0, num_epochs=NUM_EPOCHS, device=device)
        with t.no_grad():
            G_logits = G(tr_in)[:, -1, :].detach()
        # 2) Static distill student
        print(' Train static-distill student')
        S_static = train_distill(seed, G_logits, tr_in, tr_lab, te_in, te_lab,
                                 mode='static', device=device)
        # 3) Live distill student
        print(' Train live-distill student')
        S_live = train_distill(seed, None, tr_in, tr_lab, te_in, te_lab,
                               mode='live', teacher=G, device=device)
        # 4) Hard only student (M control)
        print(' Train hard-only student (M)')
        S_hard, meta_m, _ = train_one(seed=seed, wd=0.0, num_epochs=NUM_EPOCHS, device=device)
        for name, m in [('G_teacher', G), ('Static_distill', S_static),
                         ('Live_distill', S_live), ('M_hard', S_hard)]:
            r = effective_rank(m.blocks[0].mlp.W_out)
            te = eval_acc(m, te_in, te_lab)
            tr_a = eval_acc(m, tr_in, tr_lab)
            results.setdefault(name, []).append({
                'seed': seed, 'test_acc': te, 'train_acc': tr_a, 'rank_W_out': r,
            })
            print(f'  {name}: test={te:.4f}, rank={r:.2f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp18_static_distillation.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
