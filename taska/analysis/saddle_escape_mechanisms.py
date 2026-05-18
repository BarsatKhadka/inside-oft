"""Test the saddle hypothesis: do alternative perturbations also escape M's saddle?

If M is genuinely at a saddle on the full-data loss surface (Entry 15), then
ANY force that systematically perturbs the weights should eventually escape
the saddle. Weight decay is just ONE such force. Other escape mechanisms:

  1. Pure noise injection (Gaussian noise added to weights each step)
  2. SAM (sharpness-aware minimization — finds and moves toward flat regions)
  3. Tiny additional held-out data (a few extra inputs that aren't memorized)
  4. WD (control - known to work)
  5. Nothing (control - known to fail; M stays at saddle)

For each, start from M_50000 and continue training for 20,000 epochs.
Track test accuracy.

If multiple mechanisms grok → saddle theory confirmed as general unifying mechanism.
If only WD groks → there's something special about WD's specific direction.
If only WD + held-out grok → it's specifically about full-data information.

Usage:
    python taska/analysis/saddle_escape_mechanisms.py
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


def train_with_mechanism(name, mechanism_fn, device, M_state, train_in, train_lab,
                          test_in, test_lab, extra_in=None, extra_lab=None):
    """Generic training loop. mechanism_fn(model, optimizer, step) modifies weights
    or gradients as needed per step."""
    print(f'\n=== {name} ===')
    model = make_model(M_state, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))

    history = {'epoch': [], 'train_acc': [], 'test_acc': [], 'train_loss': []}

    for ep in range(NUM_EPOCHS):
        train_loss = full_loss(model, train_in, train_lab)
        # add extra data if any
        if extra_in is not None:
            train_loss = train_loss + full_loss(model, extra_in, extra_lab)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # mechanism step (does e.g. noise injection AFTER gradient step)
        if mechanism_fn is not None:
            mechanism_fn(model, optimizer, ep)

        if (ep + 1) % LOG_EVERY == 0:
            tr = eval_acc(model, train_in, train_lab)
            te = eval_acc(model, test_in, test_lab)
            history['epoch'].append(ep + 1)
            history['train_acc'].append(tr)
            history['test_acc'].append(te)
            history['train_loss'].append(train_loss.item())
        if (ep + 1) % 4000 == 0:
            print(f'  ep={ep+1:5d}  train_acc={tr:.4f}  test_acc={te:.4f}')

    return history


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    M_state = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')

    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device=device)
    test_in,  test_lab  = to_tensors(test_pairs,  P, device=device)

    # Sanity baseline
    base_model = make_model(M_state, device)
    print(f'Baseline: train_acc={eval_acc(base_model, train_in, train_lab):.4f}, '
          f'test_acc={eval_acc(base_model, test_in, test_lab):.4f}')

    histories = {}

    # 1. Control: do nothing (WD=0, no noise) - M should stay memorizing
    histories['control_nothing'] = train_with_mechanism(
        'control_nothing (WD=0, no perturbation)',
        mechanism_fn=None,
        device=device, M_state=M_state,
        train_in=train_in, train_lab=train_lab,
        test_in=test_in, test_lab=test_lab,
    )

    # 2. WD=1.0 - known to work (Entry 18)
    def wd_mech(model, optimizer, step):
        # Apply weight decay manually by subtracting from each param
        with t.no_grad():
            for p in model.parameters():
                p.mul_(1 - LR * 1.0)
    histories['wd_1.0'] = train_with_mechanism(
        'wd_1.0 (control, known to work)',
        mechanism_fn=wd_mech,
        device=device, M_state=M_state,
        train_in=train_in, train_lab=train_lab,
        test_in=test_in, test_lab=test_lab,
    )

    # 3. Noise injection: add small Gaussian noise to all weights each step
    NOISE_STD = 1e-3
    def noise_mech(model, optimizer, step):
        with t.no_grad():
            for p in model.parameters():
                p.add_(t.randn_like(p) * NOISE_STD)
    histories[f'noise_std{NOISE_STD}'] = train_with_mechanism(
        f'noise injection (std={NOISE_STD})',
        mechanism_fn=noise_mech,
        device=device, M_state=M_state,
        train_in=train_in, train_lab=train_lab,
        test_in=test_in, test_lab=test_lab,
    )

    # 4. SAM-style: take a step in the direction of the gradient, compute loss
    #    again, take a step back, then take an additional WD-like step toward flat.
    #    Simplified SAM: at each step, add small adversarial perturbation, compute
    #    gradient on perturbed weights, restore weights, apply the perturbed gradient.
    SAM_RHO = 0.05
    def sam_mech(model, optimizer, step):
        # already did one step. Now perturb, recompute, gradient step from original.
        # Simpler: we do this AFTER the normal step, so it's a SAM-flavor.
        with t.no_grad():
            grads = [p.grad.detach().clone() if p.grad is not None else t.zeros_like(p) for p in model.parameters()]
            grad_norm = t.sqrt(sum((g ** 2).sum() for g in grads)) + 1e-12
            scale = SAM_RHO / grad_norm
            saved = [p.detach().clone() for p in model.parameters()]
            for p, g in zip(model.parameters(), grads):
                p.add_(g * scale)
        # compute perturbed gradient
        loss2 = full_loss(model, train_in, train_lab)
        optimizer.zero_grad()
        loss2.backward()
        # restore weights, step with perturbed grad
        with t.no_grad():
            for p, s in zip(model.parameters(), saved):
                p.copy_(s)
        optimizer.step()
    histories[f'sam_rho{SAM_RHO}'] = train_with_mechanism(
        f'SAM-style (rho={SAM_RHO})',
        mechanism_fn=sam_mech,
        device=device, M_state=M_state,
        train_in=train_in, train_lab=train_lab,
        test_in=test_in, test_lab=test_lab,
    )

    # 5. Add a few held-out pairs to training (tiny full-data signal injection)
    n_extra = 50  # 50 extra (a, b) pairs from the test set
    extra_idx = np.random.RandomState(42).choice(len(test_pairs), n_extra, replace=False)
    extra_pairs = [test_pairs[i] for i in extra_idx]
    extra_in_t  = t.tensor(extra_pairs, dtype=t.long, device=device)
    extra_lab_t = t.tensor([(a + b) % P for a, b, _ in extra_pairs], device=device)
    histories[f'plus{n_extra}heldout'] = train_with_mechanism(
        f'add {n_extra} held-out pairs to training (WD=0)',
        mechanism_fn=None,
        device=device, M_state=M_state,
        train_in=train_in, train_lab=train_lab,
        test_in=test_in, test_lab=test_lab,
        extra_in=extra_in_t, extra_lab=extra_lab_t,
    )

    # Save
    out_json = HERE / 'results' / 'saddle_escape_mechanisms.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(histories, f)
    print(f'\nhistories -> {out_json}')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10(np.arange(len(histories)))
    for color, (name, h) in zip(cmap, histories.items()):
        ax.plot(h['epoch'], h['test_acc'], marker='.', label=name, color=color)
    ax.set_xlabel('continued-training epoch (starting from M_50000)')
    ax.set_ylabel('test accuracy')
    ax.set_title("Alternative saddle-escape mechanisms: does anything besides WD escape M's saddle?")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out = HERE / 'results' / 'fig_saddle_escape_mechanisms.png'
    fig.savefig(out, dpi=130)
    print(f'plot -> {out}')

    print('\n=== Summary: final test_acc per mechanism ===')
    for name, h in histories.items():
        ep_grok = next((h['epoch'][i] for i, a in enumerate(h['test_acc']) if a >= 0.95), None)
        print(f'  {name:35s}: final={h["test_acc"][-1]:.4f}, grok @ {ep_grok}')


if __name__ == '__main__':
    main()
