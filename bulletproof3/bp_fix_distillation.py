"""Fix bp18: static vs live distillation.

Bug: static_distill and live_distill came out IDENTICAL across all seeds.
Likely cause: train_distill was called with the same seed offset for both modes,
producing identical initial parameters AND identical SGD trajectories (no stochasticity
because we train on full batch).

Fix:
  - Use distinct seed offsets for static (seed+999) vs live (seed+1999)
  - Verify by printing first-layer weight norm right after init
  - Also fix a subtle bug: the original used T_DISTILL=4 but didn't divide F.kl_div
    by T^2 correctly. Use the standard distillation loss: T^2 * KL(student/T || teacher/T)
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.optim as optim
import torch.nn.functional as F

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from taska.data import gen_train_test, to_tensors
from taska.model import Transformer
from bulletproof2._common import cross_entropy_hp, effective_rank, eval_acc, train_one

P = 113
LR = 1e-3
NUM_EPOCHS = 20000
T_DISTILL = 4.0
NUM_SEEDS = 3


def train_distill(student_seed, teacher_logits, tr_in, tr_lab, te_in, te_lab,
                   mode, teacher=None, device='cuda'):
    """Train student. mode in {'static', 'live'}. Initialize from distinct seed."""
    t.manual_seed(student_seed); np.random.seed(student_seed)
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1).to(device)
    # Print init norm to verify distinct init
    init_norm = float(t.norm(model.embed.W_E))
    print(f'    [init seed={student_seed}] init_norm(W_E)={init_norm:.6f}')
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
            print(f'    ep={ep+1}: test={te:.4f}')
    return model


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {}
    for seed in range(NUM_SEEDS):
        print(f'\n=== teacher seed={seed} ===')
        # Train G teacher
        print('  Train G teacher')
        G, meta_g, (tr_in, tr_lab, te_in, te_lab) = train_one(
            seed=seed, wd=1.0, num_epochs=NUM_EPOCHS, device=device)
        with t.no_grad():
            G_logits = G(tr_in)[:, -1, :].detach()
        print('  Train STATIC distill student (seed offset +999)')
        S_static = train_distill(student_seed=seed + 999,
                                 teacher_logits=G_logits,
                                 tr_in=tr_in, tr_lab=tr_lab,
                                 te_in=te_in, te_lab=te_lab,
                                 mode='static', device=device)
        print('  Train LIVE distill student (seed offset +1999)')
        S_live = train_distill(student_seed=seed + 1999,
                               teacher_logits=None,
                               tr_in=tr_in, tr_lab=tr_lab,
                               te_in=te_in, te_lab=te_lab,
                               mode='live', teacher=G, device=device)
        # M control
        print('  Train hard-only M control')
        S_hard, meta_m, _ = train_one(seed=seed, wd=0.0, num_epochs=NUM_EPOCHS, device=device)

        for name, m in [('G_teacher', G), ('Static_distill', S_static),
                         ('Live_distill', S_live), ('M_hard', S_hard)]:
            r = effective_rank(m.blocks[0].mlp.W_out)
            te = eval_acc(m, te_in, te_lab)
            tr_a = eval_acc(m, tr_in, tr_lab)
            wnorm = float(t.norm(m.embed.W_E))
            results.setdefault(name, []).append({
                'seed': seed, 'test_acc': te, 'train_acc': tr_a,
                'rank_W_out': r, 'init_check_W_E_norm': wnorm,
            })
            print(f'  {name}: test={te:.4f}, rank={r:.2f}, W_E_norm={wnorm:.4f}')

    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp_fix_distillation.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
