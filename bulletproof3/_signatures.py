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

def _math_sdpa_ctx():
    """Force PyTorch's math attention backend during HVPs.

    The optimized scaled_dot_product_efficient_attention CUDA kernel does not
    implement double-backward, which we need for Hessian-vector products
    (create_graph=True then grad again). Forcing the math backend makes
    Hessian compute work on Transformer-based models (CharLM, ViT, Pythia).
    """
    # Try the new API first (PyTorch 2.3+), fall back to the deprecated one.
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        pass
    try:
        return t.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def _hvp(model, loss_fn, vec) -> t.Tensor:
    """Hessian-vector product given a loss function that returns a scalar."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    with _math_sdpa_ctx():
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
    Returns: sorted descending list of approximate eigenvalues. Returns empty
    list if the iteration produced no usable values (numerical failure).
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
        try:
            Hq = _hvp(model, loss_fn, q).to(dtype)
        except Exception as e:
            if verbose:
                print(f'  Lanczos HVP failed at j={j}: {e}')
            break
        # Bail out if HVP produced NaN/Inf
        if not t.isfinite(Hq).all():
            if verbose:
                print(f'  Lanczos: non-finite HVP at j={j}, stopping')
            break
        alpha = float(t.dot(q, Hq))
        if not np.isfinite(alpha):
            if verbose:
                print(f'  Lanczos: non-finite alpha at j={j}, stopping')
            break
        alphas.append(alpha)
        r = Hq - alpha * q - beta_prev * q_prev
        # full reorth
        for qi in Q:
            r = r - float(t.dot(r, qi)) * qi
        beta = float(r.norm())
        if beta < 1e-10 or not np.isfinite(beta):
            if verbose:
                print(f'  Lanczos converged/stopped at j={j} (beta={beta})')
            break
        betas.append(beta)
        q_prev = q
        q = r / beta
        Q.append(q)
        beta_prev = beta
    m = len(alphas)
    if m == 0:
        return []
    # Compute eigenvalues of the tridiagonal Lanczos matrix.
    # scipy.linalg.eigh_tridiagonal is specialized and far more robust
    # for ill-conditioned tridiagonals than np.linalg.eigvalsh on the dense
    # matrix; fall back to the dense path if scipy unavailable.
    alphas_arr = np.asarray(alphas, dtype=np.float64)
    betas_arr = np.asarray(betas[:m-1] if m > 1 else [], dtype=np.float64)
    try:
        from scipy.linalg import eigh_tridiagonal
        eigs = eigh_tridiagonal(alphas_arr, betas_arr, eigvals_only=True)
    except Exception:
        try:
            T = (np.diag(alphas_arr)
                 + np.diag(betas_arr, 1)
                 + np.diag(betas_arr, -1))
            eigs = np.linalg.eigvalsh(T)
        except np.linalg.LinAlgError:
            if verbose:
                print('  Lanczos: eigvalsh did not converge; returning alphas as estimate')
            # As a last resort, use the alphas themselves as eigenvalue estimates
            # (Ritz values along the diagonal of T).
            eigs = alphas_arr
    # Filter out any non-finite values
    eigs = eigs[np.isfinite(eigs)]
    return sorted(eigs.tolist(), reverse=True)


def hessian_top_bot(model, loss_fn, k=20):
    """Return (top, bottom, all_eigs). Returns (nan, nan, []) on numerical failure."""
    eigs = lanczos_hessian(model, loss_fn, k=k)
    if not eigs:
        return float('nan'), float('nan'), []
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

# ---------- Norm-based measurements ----------

def weight_l2_norm(model) -> float:
    """Total L2 norm of all parameters: sqrt(sum_i ||theta_i||^2)."""
    total = 0.0
    for p in model.parameters():
        if p.is_floating_point():
            total += float((p.detach() ** 2).sum())
    return float(np.sqrt(total))


def relative_flatness(top_hessian_eig: float, weight_l2: float) -> float:
    """Petzka et al. 2021 relative flatness: sharpness scaled by weight magnitude.

    Petzka, Kamp, Adilova, Sminchisescu, Boley (NeurIPS 2021).
    "Relative Flatness and Generalization."

    The standard top Hessian eigenvalue ("sharpness", Keskar 2017) is NOT
    invariant under network reparameterization: rescaling weights can make
    any minimum arbitrarily sharp or flat without changing the network's
    function (Dinh et al. 2017). Petzka's relative flatness scales sharpness
    by ||theta||^2, producing a reparameterization-invariant proxy that
    they show empirically correlates with generalization more reliably than
    raw top eigenvalue.

    Returns:  top_eig * (||theta||_2)^2
    """
    return float(top_hessian_eig) * (float(weight_l2) ** 2)


def distance_from_init(model, init_state_dict) -> dict:
    """||theta_final - theta_init|| and various relative measures.

    init_state_dict: dict of parameter names -> initial tensor values.
    """
    sq = 0.0
    sq_init = 0.0
    sq_final = 0.0
    n_params = 0
    for name, p in model.named_parameters():
        if not p.is_floating_point() or name not in init_state_dict:
            continue
        init = init_state_dict[name].to(p.device).to(p.dtype)
        if init.shape != p.shape:
            continue
        diff = p.detach() - init
        sq += float((diff ** 2).sum())
        sq_init += float((init ** 2).sum())
        sq_final += float((p.detach() ** 2).sum())
        n_params += p.numel()
    if sq_init <= 0 or n_params == 0:
        return {'dist_from_init': float(np.sqrt(sq)),
                'rel_dist_from_init': float('nan'),
                'init_norm': float(np.sqrt(sq_init)),
                'final_norm': float(np.sqrt(sq_final))}
    return {
        'dist_from_init': float(np.sqrt(sq)),
        'rel_dist_from_init': float(np.sqrt(sq) / np.sqrt(sq_init)),
        'init_norm': float(np.sqrt(sq_init)),
        'final_norm': float(np.sqrt(sq_final)),
        'norm_ratio_final_over_init': float(np.sqrt(sq_final) / np.sqrt(sq_init)),
    }


def path_norm_proxy(model) -> dict:
    """Proxy for L2 path norm: product of operator (spectral) norms of weight
    matrices in the model. Strict path-norm is intractable for transformers;
    product-of-spectral-norms is the standard Lipschitz upper bound used in
    Bartlett et al. 2017 norm-bound literature.

    Returns also sum of log spectral norms (more numerically stable when the
    product would overflow).
    """
    log_prod = 0.0
    spectral_norms = {}
    for name, p in model.named_parameters():
        if not p.is_floating_point() or p.ndim < 2:
            continue
        M = p.detach().cpu().float()
        if M.ndim > 2:
            M = M.reshape(M.shape[0], -1)
        try:
            s_max = float(t.linalg.svdvals(M)[0])
        except Exception:
            continue
        spectral_norms[name] = s_max
        if s_max > 0:
            log_prod += float(np.log(s_max))
    # Product can overflow; report the log explicitly
    return {
        'log_path_norm_proxy': float(log_prod),
        'spectral_norms': spectral_norms,
    }


# ---------- Full battery ----------

def compute_full_battery(model, train_loss_fn, test_loss_fn,
                          train_losses_array=None, test_losses_array=None,
                          init_state_dict=None,
                          lanczos_k=20, verbose=False) -> dict:
    """Compute the standard battery on a converged model.

    train_loss_fn, test_loss_fn: closures returning scalar loss with grads enabled
    train_losses_array, test_losses_array: per-example loss arrays for MIA (optional)
    init_state_dict: parameter-name -> tensor mapping of initial weights. If
        provided, distance-from-init metrics will be computed.
    """
    out = {}
    out['ranks'] = all_ranks(model)
    if verbose:
        print('  computing norm-based measurements...')
    out['weight_l2_norm'] = weight_l2_norm(model)
    if init_state_dict is not None:
        out.update(distance_from_init(model, init_state_dict))
    out.update(path_norm_proxy(model))
    if verbose:
        print('  computing Hessian top + bot on full data...')
    def full_loss_fn():
        return 0.5 * (train_loss_fn() + test_loss_fn())
    top, bot, all_eigs = hessian_top_bot(model, full_loss_fn, k=lanczos_k)
    out['hessian_top_full'] = top
    out['hessian_bot_full'] = bot
    out['hessian_eigs_full'] = all_eigs
    # Petzka relative flatness: sharpness * ||theta||^2 (reparameterization-invariant
    # fix to Dinh 2017's critique of top eigenvalue).
    if np.isfinite(top) and out.get('weight_l2_norm') is not None:
        out['relative_flatness_full'] = float(top) * (out['weight_l2_norm'] ** 2)
    if verbose:
        print('  computing Hessian top + bot on train data...')
    try:
        top_tr, bot_tr, _ = hessian_top_bot(model, train_loss_fn, k=lanczos_k)
        out['hessian_top_train'] = top_tr
        out['hessian_bot_train'] = bot_tr
        if np.isfinite(top_tr) and out.get('weight_l2_norm') is not None:
            out['relative_flatness_train'] = float(top_tr) * (out['weight_l2_norm'] ** 2)
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


# ---------- Helper: capture init state at start of training ----------

def capture_init_state(model) -> dict:
    """Snapshot a model's parameters right after init. Call BEFORE any training.
    Returns a CPU dict suitable for later passing to compute_full_battery."""
    return {name: p.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.is_floating_point()}
