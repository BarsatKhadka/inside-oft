"""Shared signature computation for the scale ladder.

Three core signatures we want to compute on EVERY (arch, dataset) pair:
  1. Effective rank of each weight matrix (Shannon entropy of squared SVs)
  2. Top + bottom Hessian eigenvalue (via Lanczos with reorthogonalization)
  3. cos(grad_L_train, grad_L_test) at converged parameters

Plus supporting metrics: gradient norms, loss-based MIA AUC, final accuracies.

This module is imported by every bp3 training script so all results are directly comparable.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch as t
import torch.nn.functional as F


# ---------- Effective rank ----------

def effective_rank(W: t.Tensor) -> float:
    """Shannon entropy of normalized squared singular values."""
    if W.ndim < 2:
        return float('nan')
    M = W.detach().cpu().float()
    if M.ndim > 2:
        M = M.reshape(M.shape[0], -1)
    try:
        s = t.linalg.svdvals(M)
    except Exception:
        return float('nan')
    p = s ** 2
    if p.sum() == 0:
        return float('nan')
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


def all_ranks(model) -> dict:
    """Effective rank of every 2D+ float parameter in the model."""
    out = {}
    for name, p in model.named_parameters():
        if p.ndim >= 2 and p.is_floating_point():
            out[name] = effective_rank(p)
    return out


# ---------- Hessian eigenvalues via Lanczos ----------

def _hvp(model, loss_fn, vec) -> t.Tensor:
    """Hessian-vector product given a loss function that returns a scalar."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss = loss_fn()
    grads = t.autograd.grad(loss, list(model.parameters()), create_graph=True)
    flat_g = t.cat([g.flatten() for g in grads])
    Hv = t.autograd.grad(flat_g, list(model.parameters()), grad_outputs=vec,
                          retain_graph=False, allow_unused=True)
    parts = []
    for h, p in zip(Hv, model.parameters()):
        if h is None:
            parts.append(t.zeros_like(p).flatten())
        else:
            parts.append(h.flatten())
    return t.cat(parts).detach()


def lanczos_hessian(model, loss_fn, k=30, dtype=t.float32, verbose=False) -> list:
    """Top-k Lanczos eigenvalues of the Hessian defined by loss_fn.

    loss_fn: callable returning a scalar loss with current model state (closure
             must NOT detach; gradients must flow).
    Returns: sorted descending list of approximate eigenvalues.
    """
    device = next(model.parameters()).device
    n = sum(p.numel() for p in model.parameters())
    q = t.randn(n, device=device, dtype=dtype)
    q /= q.norm()
    Q = [q]
    alphas, betas = [], []
    q_prev = t.zeros_like(q)
    beta_prev = 0.0
    for j in range(k):
        Hq = _hvp(model, loss_fn, q).to(dtype)
        alpha = float(t.dot(q, Hq))
        alphas.append(alpha)
        r = Hq - alpha * q - beta_prev * q_prev
        # full reorth
        for qi in Q:
            r = r - float(t.dot(r, qi)) * qi
        beta = float(r.norm())
        if beta < 1e-10:
            if verbose:
                print(f'  Lanczos converged at j={j}')
            break
        betas.append(beta)
        q_prev = q
        q = r / beta
        Q.append(q)
        beta_prev = beta
    m = len(alphas)
    T = np.diag(alphas) + np.diag(betas[:m-1], 1) + np.diag(betas[:m-1], -1)
    eigs = np.linalg.eigvalsh(T)
    return sorted(eigs.tolist(), reverse=True)


def hessian_top_bot(model, loss_fn, k=20):
    """Return (top, bottom, all_eigs)."""
    eigs = lanczos_hessian(model, loss_fn, k=k)
    return eigs[0], eigs[-1], eigs


# ---------- Gradient angle ----------

def flat_grad(model, loss_fn) -> t.Tensor:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss_fn().backward()
    g = t.cat([p.grad.detach().flatten() for p in model.parameters()
               if p.grad is not None])
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    return g


def gradient_angle(model, train_loss_fn, test_loss_fn) -> dict:
    g_tr = flat_grad(model, train_loss_fn)
    g_te = flat_grad(model, test_loss_fn)
    cos = float(t.dot(g_tr, g_te) / (g_tr.norm() * g_te.norm() + 1e-30))
    return {
        'cos_grad_train_test': cos,
        'angle_deg': float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
        'grad_train_norm': float(g_tr.norm()),
        'grad_test_norm': float(g_te.norm()),
        'grad_ratio_test_over_train': float(g_te.norm() / (g_tr.norm() + 1e-30)),
    }


# ---------- MIA AUC (loss-based) ----------

def _auc(scores, labels):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # np.trapz removed in numpy 2.x; use trapezoid if available, fall back to trapz
    trap = getattr(np, 'trapezoid', None) or np.trapz
    return float(trap(np.cumsum(y) / n_pos, np.cumsum(1 - y) / n_neg))


def mia_loss_auc(train_losses_arr: np.ndarray, test_losses_arr: np.ndarray) -> float:
    scores = np.concatenate([-train_losses_arr, -test_losses_arr])
    labels = np.concatenate([np.ones(len(train_losses_arr)),
                              np.zeros(len(test_losses_arr))])
    return _auc(scores, labels)


# ---------- Full battery ----------

def compute_full_battery(model, train_loss_fn, test_loss_fn,
                          train_losses_array=None, test_losses_array=None,
                          lanczos_k=20, verbose=False) -> dict:
    """Compute the standard battery on a converged model.

    train_loss_fn, test_loss_fn: closures returning scalar loss with grads enabled
    train_losses_array, test_losses_array: per-example loss arrays for MIA (optional)
    """
    out = {}
    out['ranks'] = all_ranks(model)
    if verbose:
        print('  computing Hessian top + bot on full data...')
    # Full-data Hessian: combine the two losses with proper weighting in closure
    # The simplest version: caller provides a "full_loss_fn" via combining train+test.
    # But we want top and bot of FULL loss landscape. Build it here:
    def full_loss_fn():
        return 0.5 * (train_loss_fn() + test_loss_fn())
    top, bot, all_eigs = hessian_top_bot(model, full_loss_fn, k=lanczos_k)
    out['hessian_top_full'] = top
    out['hessian_bot_full'] = bot
    out['hessian_eigs_full'] = all_eigs
    # Train-only Hessian (also useful)
    if verbose:
        print('  computing Hessian top + bot on train data...')
    try:
        top_tr, bot_tr, _ = hessian_top_bot(model, train_loss_fn, k=lanczos_k)
        out['hessian_top_train'] = top_tr
        out['hessian_bot_train'] = bot_tr
    except Exception as e:
        out['hessian_train_error'] = str(e)
    if verbose:
        print('  computing gradient angle...')
    out.update(gradient_angle(model, train_loss_fn, test_loss_fn))
    if train_losses_array is not None and test_losses_array is not None:
        out['mia_loss_auc'] = mia_loss_auc(train_losses_array, test_losses_array)
        out['mean_train_loss'] = float(train_losses_array.mean())
        out['mean_test_loss'] = float(test_losses_array.mean())
    return out
