"""mech10: Mode connectivity for tier5 CharLM Shakespeare.

Completes the basin-structure table across all tracks (algorithmic, vision, LM).

Trains one M (wd=0, dropout=0, 60k iters) and one G (wd=1e-3, dropout=0.1,
60k iters) CharLM, then linearly interpolates between them in weight space
and reports per-alpha val_loss.

Expected: barrier exists (LM-from-scratch behaves like algorithmic — different
basins). If no barrier, that's also informative.
"""
import json
import copy
import urllib.request
from pathlib import Path
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bulletproof3.tier5_charlm_shakespeare import (
    CharLM, get_shakespeare, get_batch, CTX, BATCH, DIM, DEPTH, HEADS
)

DATA_DIR = HERE.parent / 'data'
OUT_DIR = HERE / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ITERS = 60000  # reduced from tier5's 80k for time


def train(seed, mode, device, train_data, val_data, vocab):
    t.manual_seed(seed); np.random.seed(seed)
    wd = 1e-3 if mode == 'G' else 0.0
    drop = 0.1 if mode == 'G' else 0.0
    model = CharLM(vocab, dropout=drop).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
    for it in range(N_ITERS):
        x, y = get_batch(train_data, CTX, BATCH, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 10000 == 0:
            model.eval()
            with t.no_grad():
                x_v, y_v = get_batch(val_data, CTX, BATCH, device)
                l_v = F.cross_entropy(model(x_v).reshape(-1, vocab), y_v.reshape(-1)).item()
            print(f'  it={it+1}: train_loss={loss.item():.4f}, val_loss={l_v:.4f}')
            model.train()
    return model


@t.no_grad()
def eval_loss(model, train_data, val_data, vocab, device, n_batches=20):
    model.eval()
    tr_losses = []; te_losses = []
    for _ in range(n_batches):
        x, y = get_batch(train_data, CTX, BATCH, device)
        tr_losses.append(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())
        x, y = get_batch(val_data, CTX, BATCH, device)
        te_losses.append(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())
    return float(np.mean(tr_losses)), float(np.mean(te_losses))


def interpolate(m_state, g_state, alphas, vocab, train_data, val_data, device):
    interp = CharLM(vocab).to(device)
    results = []
    for a in alphas:
        new_sd = {}
        for k in m_state:
            if m_state[k].dtype.is_floating_point:
                new_sd[k] = (1 - a) * m_state[k].to(device) + a * g_state[k].to(device)
            else:
                new_sd[k] = m_state[k].to(device)
        interp.load_state_dict(new_sd)
        tr_l, te_l = eval_loss(interp, train_data, val_data, vocab, device)
        print(f'  alpha={a:.2f}: train={tr_l:.4f} val={te_l:.4f}')
        results.append({'alpha': a, 'train_loss': tr_l, 'test_loss': te_l})
    return results


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    train_data, val_data, vocab = get_shakespeare()
    print('=== Training M (CharLM, wd=0, no dropout) ===')
    m_model = train(0, 'M', device, train_data, val_data, vocab)
    print('=== Training G (CharLM, wd=1e-3, dropout=0.1) ===')
    g_model = train(0, 'G', device, train_data, val_data, vocab)
    m_state = {k: v.detach().cpu() for k, v in m_model.state_dict().items()}
    g_state = {k: v.detach().cpu() for k, v in g_model.state_dict().items()}
    alphas = np.linspace(0, 1, 11).tolist()
    print('\n=== Interpolating ===')
    curve = interpolate(m_state, g_state, alphas, vocab, train_data, val_data, device)
    barrier = max(r['test_loss'] for r in curve) - max(curve[0]['test_loss'], curve[-1]['test_loss'])
    print(f'\nBarrier height (val_loss): {barrier:.4f}')
    out = {
        'tier': 'tier5_charlm',
        'seed': 0,
        'iterations': N_ITERS,
        'alphas': alphas,
        'curve': curve,
        'barrier_height_test': barrier,
        'endpoint_m_test_loss': curve[0]['test_loss'],
        'endpoint_g_test_loss': curve[-1]['test_loss'],
        'midpoint_test_loss': curve[len(curve) // 2]['test_loss'],
    }
    with open(OUT_DIR / 'mech10_modeconn_tier5.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
