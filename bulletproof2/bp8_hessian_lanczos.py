"""bp8: Hessian Lanczos spectrum at M and G.

Compute top-k eigenvalues of full-data Hessian via Lanczos iteration for 3 seeds
each of M and G. Reports eigenvalue distribution shape, degeneracy (count of
near-zero eigenvalues), condition number, and full spectrum.

Lanczos is more efficient and more numerically stable than power iteration for
multiple eigenvalues, and gives the spectrum, not just the top.
"""
import json
from pathlib import Path
import numpy as np
import torch as t

from _common import (HERE, P, cross_entropy_hp, train_one)

NUM_SEEDS = 3
NUM_EPOCHS = 20000
LANCZOS_K = 40  # Number of Lanczos iterations -> number of eigenvalues estimated


def hvp(model, inputs, labels, vec):
    """Hessian-vector product on full data."""
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    loss = cross_entropy_hp(model(inputs)[:, -1, :], labels)
    grads = t.autograd.grad(loss, list(model.parameters()), create_graph=True)
    flat_g = t.cat([g.flatten() for g in grads])
    Hv = t.autograd.grad(flat_g, list(model.parameters()), grad_outputs=vec, retain_graph=False)
    return t.cat([h.flatten() for h in Hv]).detach()


def lanczos(model, inputs, labels, k=LANCZOS_K, dtype=t.float32):
    """Lanczos tridiagonalization -> top-k eigenvalues."""
    device = next(model.parameters()).device
    n = sum(p.numel() for p in model.parameters())
    print(f'  Lanczos n={n}, k={k}')
    # Init
    q = t.randn(n, device=device, dtype=dtype)
    q /= q.norm()
    alphas, betas = [], []
    Q = [q]
    q_prev = t.zeros_like(q)
    beta_prev = 0.0
    for j in range(k):
        Hq = hvp(model, inputs, labels, q).to(dtype)
        alpha = float(t.dot(q, Hq))
        alphas.append(alpha)
        r = Hq - alpha * q - beta_prev * q_prev
        # Full reorth (mild cost, much more stable)
        for qi in Q:
            r = r - float(t.dot(r, qi)) * qi
        beta = float(r.norm())
        if beta < 1e-10:
            print(f'  Lanczos converged at j={j}')
            break
        betas.append(beta)
        q_prev = q
        q = r / beta
        Q.append(q)
        beta_prev = beta
    # Build tridiag
    m = len(alphas)
    T = np.diag(alphas) + np.diag(betas[:m-1], 1) + np.diag(betas[:m-1], -1)
    eigs = np.linalg.eigvalsh(T)
    return sorted(eigs.tolist(), reverse=True)


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    for seed in range(NUM_SEEDS):
        for label, wd in [('M', 0.0), ('G', 1.0)]:
            print(f'\n=== {label} seed={seed} ===')
            model, meta, (tr_in, tr_lab, te_in, te_lab) = train_one(
                seed=seed, wd=wd, num_epochs=NUM_EPOCHS, device=device)
            full_in = t.cat([tr_in, te_in]); full_lab = t.cat([tr_lab, te_lab])
            print('  computing Lanczos on full data...')
            eigs_full = lanczos(model, full_in, full_lab)
            print('  computing Lanczos on train data...')
            eigs_train = lanczos(model, tr_in, tr_lab)
            # Stats
            arr_full = np.array(eigs_full)
            arr_train = np.array(eigs_train)
            entry = {
                'seed': seed, 'wd': wd, 'final_test_acc': meta['final_test_acc'],
                'eigs_full': eigs_full,
                'eigs_train': eigs_train,
                'top_eig_full': float(arr_full[0]),
                'bot_eig_full': float(arr_full[-1]),
                'top_eig_train': float(arr_train[0]),
                'bot_eig_train': float(arr_train[-1]),
                'n_near_zero_full': int((np.abs(arr_full) < 1e-4 * arr_full[0]).sum()),
                'n_near_zero_train': int((np.abs(arr_train) < 1e-4 * abs(arr_train[0])).sum() if abs(arr_train[0]) > 0 else 0),
                'cond_full': float(abs(arr_full[0]) / max(abs(arr_full[arr_full != 0].min()), 1e-12)),
                'trace_full': float(arr_full.sum()),
            }
            results[label].append(entry)
            print(f'  top_full={entry["top_eig_full"]:.3f}, bot_full={entry["bot_eig_full"]:.3f}, '
                  f'near_zero_train={entry["n_near_zero_train"]}/{len(eigs_train)}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp8_hessian_lanczos.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
