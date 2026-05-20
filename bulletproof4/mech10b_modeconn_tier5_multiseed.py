"""mech10b: Multi-seed mode connectivity for tier5 CharLM.

Validates the CharLM basin-structure claim (barrier=+1.35 from mech10 single seed).
Runs 3 M seeds × 1 G seed = 3 independent M-G interpolations and reports
mean ± std barrier height.

If the barrier is robust across M seeds (std << mean), the single-seed mech10
result stands and the paper claim is bulletproof. If std is large relative to
mean (>50%), the "different basins" verdict for CharLM is uncertain.
"""
import json
import copy
from pathlib import Path
import numpy as np
import torch as t
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

N_ITERS = 60000
M_SEEDS = [0, 1, 2]
G_SEED = 0
N_ALPHAS = 11


def train(seed, mode, device, train_data, val_data, vocab):
    t.manual_seed(seed)
    np.random.seed(seed)
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
            print(f'  [{mode} seed={seed}] it={it+1}: train={loss.item():.4f} val={l_v:.4f}')
            model.train()
    return model


@t.no_grad()
def eval_loss(model, train_data, val_data, vocab, device, n_batches=20):
    model.eval()
    tr_l, te_l = [], []
    for _ in range(n_batches):
        x, y = get_batch(train_data, CTX, BATCH, device)
        tr_l.append(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())
        x, y = get_batch(val_data, CTX, BATCH, device)
        te_l.append(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())
    return float(np.mean(tr_l)), float(np.mean(te_l))


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


def barrier_height(curve):
    peak = max(r['test_loss'] for r in curve)
    endpoint_max = max(curve[0]['test_loss'], curve[-1]['test_loss'])
    return peak - endpoint_max


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    train_data, val_data, vocab = get_shakespeare()
    alphas = np.linspace(0, 1, N_ALPHAS).tolist()

    print(f'=== Training G (seed={G_SEED}) ===')
    g_model = train(G_SEED, 'G', device, train_data, val_data, vocab)
    g_state = {k: v.detach().cpu() for k, v in g_model.state_dict().items()}
    del g_model

    all_pairs = []
    barriers = []

    for m_seed in M_SEEDS:
        print(f'\n=== Training M (seed={m_seed}) ===')
        m_model = train(m_seed, 'M', device, train_data, val_data, vocab)
        m_state = {k: v.detach().cpu() for k, v in m_model.state_dict().items()}
        del m_model

        print(f'\n=== Interpolating M_seed={m_seed} → G_seed={G_SEED} ===')
        curve = interpolate(m_state, g_state, alphas, vocab, train_data, val_data, device)
        bh = barrier_height(curve)
        barriers.append(bh)
        print(f'  barrier_height = {bh:.4f}')

        all_pairs.append({
            'm_seed': m_seed,
            'g_seed': G_SEED,
            'curve': curve,
            'barrier_height_test': bh,
            'endpoint_m_test_loss': curve[0]['test_loss'],
            'endpoint_g_test_loss': curve[-1]['test_loss'],
        })

    mean_barrier = float(np.mean(barriers))
    std_barrier = float(np.std(barriers))
    print(f'\n=== Summary ===')
    print(f'Barriers per M seed: {[f"{b:.4f}" for b in barriers]}')
    print(f'Mean barrier: {mean_barrier:.4f} ± {std_barrier:.4f}')
    verdict = 'DIFFERENT BASINS' if mean_barrier > 0.5 else ('UNCERTAIN' if mean_barrier > 0.1 else 'SAME BASIN')
    print(f'Verdict: {verdict}')

    out = {
        'tier': 'tier5_charlm',
        'n_iters': N_ITERS,
        'm_seeds': M_SEEDS,
        'g_seed': G_SEED,
        'n_alphas': N_ALPHAS,
        'pairs': all_pairs,
        'barriers': barriers,
        'mean_barrier_test': mean_barrier,
        'std_barrier_test': std_barrier,
        'verdict': verdict,
        'mech10_single_seed_barrier': 1.346,
    }
    with open(OUT_DIR / 'mech10b_modeconn_tier5_multiseed.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {OUT_DIR}/mech10b_modeconn_tier5_multiseed.json')


if __name__ == '__main__':
    main()
