"""Git Re-Basin permutation alignment.

The MLP has 512 hidden neurons. Their ORDER is arbitrary -- if you permute
the rows of W_in (and b_in) AND the columns of W_out the same way, the model
computes exactly the same function. So neuron #47 in M might be doing the
same job as neuron #213 in G.

Question: if we PERMUTE M's neurons to maximally align with G's neurons, does
the loss barrier between (M_permuted) and G shrink?

  - If barrier goes to ~0: M and G were in the SAME basin all along, our
    "different basins" finding was a basis (literally neuron-ordering) artifact.
    Spectral surgery negative result would also need reframing.

  - If barrier remains high: M and G are in genuinely different basins even
    after the field-standard alignment fix. Our story holds.

Method (per Ainsworth et al. 2023, "Git Re-Basin"):
  - Build cost matrix C of shape (512, 512). C[i, j] = similarity between
    M's neuron i and G's neuron j.
  - Solve the linear assignment problem to get optimal permutation pi.
  - Apply pi to M's MLP: W_in[pi], b_in[pi], W_out[:, pi].
  - Verify M's behavior is unchanged (sanity check).
  - Interpolate M_permuted -> G in weight space, plot loss.

Cost function: cosine similarity between (concatenated W_in row, W_out col)
for each neuron. We maximize total similarity = minimize negative similarity.

Usage:
    python taska/analysis/permutation_align.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from scipy.optimize import linear_sum_assignment

from data import gen_train_test, to_tensors
from model import Transformer

P = 113
D_MODEL = 128
SEED = 0


def load_state(ckpt):
    return t.load(ckpt, map_location='cpu', weights_only=True)['model']


def load_model_from_state(state):
    model = Transformer(p=P, d_model=D_MODEL, num_heads=4, n_ctx=3, num_layers=1)
    model.load_state_dict(state)
    model.eval()
    return model


@t.no_grad()
def eval_loss_acc(model, inputs, labels):
    logits = model(inputs)[:, -1, :].to(t.float64)
    log_probs = t.nn.functional.log_softmax(logits, dim=-1)
    loss = -log_probs[t.arange(labels.shape[0]), labels].mean().item()
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


def compute_neuron_features(state):
    """For each of 512 MLP neurons, build a feature vector = concat(W_in row,
    W_out column). Used to match neurons across models."""
    W_in  = state['blocks.0.mlp.W_in']    # (512, 128)
    W_out = state['blocks.0.mlp.W_out']   # (128, 512)
    feats = t.cat([W_in, W_out.T], dim=1)  # (512, 256)
    # L2-normalize for cosine similarity
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
    return feats


def find_best_permutation(feats_M, feats_G):
    """Hungarian algorithm to maximize sum of cosine similarities.

    Returns: array pi of length 512 such that M's neuron pi[j] is matched
    to G's neuron j.
    """
    # Cosine similarity matrix
    sim = (feats_M @ feats_G.T).numpy()    # (512, 512)
    # linear_sum_assignment MINIMIZES cost. Negate to maximize similarity.
    row_ind, col_ind = linear_sum_assignment(-sim)
    # row_ind[i] = i  (always); col_ind[i] = which G-neuron is matched to M-neuron i
    # We want a permutation pi such that M[pi] aligns with G[1..512] in order.
    # That is: pi[j] = which M-neuron should occupy position j after permutation.
    # If M-neuron i goes to position col_ind[i] in the aligned order, then
    # pi[col_ind[i]] = i.
    pi = np.empty(len(col_ind), dtype=int)
    for i, j in enumerate(col_ind):
        pi[j] = i
    # Average similarity achieved
    avg_sim = sim[row_ind, col_ind].mean()
    return pi, avg_sim


def apply_permutation(state, pi):
    """Return new state dict with M's MLP neurons permuted by pi."""
    pi_t = t.tensor(pi, dtype=t.long)
    new_state = {k: v.clone() for k, v in state.items()}
    new_state['blocks.0.mlp.W_in']  = state['blocks.0.mlp.W_in'][pi_t]
    new_state['blocks.0.mlp.b_in']  = state['blocks.0.mlp.b_in'][pi_t]
    new_state['blocks.0.mlp.W_out'] = state['blocks.0.mlp.W_out'][:, pi_t]
    return new_state


def interp_state(s_M, s_G, alpha):
    return {k: (1 - alpha) * s_M[k] + alpha * s_G[k] for k in s_M}


def main():
    s_M = load_state(HERE / 'checkpoints' / 'M' / 'final.pt')
    s_G = load_state(HERE / 'checkpoints' / 'G' / 'final.pt')

    train_pairs, _ = gen_train_test(p=P, frac_train=0.3, seed=SEED)
    train_in, train_lab = to_tensors(train_pairs, P, device='cpu')

    # Sanity check: original M
    m_loss, m_acc = eval_loss_acc(load_model_from_state(s_M), train_in, train_lab)
    print(f'Original M:  train_loss={m_loss:.4e}  train_acc={m_acc:.4f}')

    # Compute permutation
    feats_M = compute_neuron_features(s_M)
    feats_G = compute_neuron_features(s_G)
    pi, avg_sim = find_best_permutation(feats_M, feats_G)
    print(f'\nFound permutation. Avg neuron-pair cosine similarity = {avg_sim:.4f}')
    print(f'(Reference: random-pair cosine in 256-dim is ~0; identical neurons would be 1.0)')

    # Apply permutation to M
    s_M_perm = apply_permutation(s_M, pi)

    # Sanity check: M_perm should have SAME behavior as M (permutation is an exact symmetry)
    perm_loss, perm_acc = eval_loss_acc(load_model_from_state(s_M_perm), train_in, train_lab)
    print(f'\nPermuted M:  train_loss={perm_loss:.4e}  train_acc={perm_acc:.4f}')
    print(f'(should match original M exactly -- permutation is a symmetry of the model)')

    # ============================================================
    # Linear interpolation: M_perm -> G
    # ============================================================
    alphas = np.linspace(0, 1, 21)
    losses_orig = []     # original M -> G (for comparison)
    losses_perm = []     # permuted M -> G

    print(f'\n{"alpha":>6}  {"orig M->G loss":>18}  {"perm M->G loss":>18}')
    for a in alphas:
        # Original
        s_interp_orig = interp_state(s_M, s_G, a)
        l_o, _ = eval_loss_acc(load_model_from_state(s_interp_orig), train_in, train_lab)
        # Permuted
        s_interp_perm = interp_state(s_M_perm, s_G, a)
        l_p, _ = eval_loss_acc(load_model_from_state(s_interp_perm), train_in, train_lab)
        losses_orig.append(l_o)
        losses_perm.append(l_p)
        print(f'{a:>6.2f}  {l_o:>18.4e}  {l_p:>18.4e}')

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, losses_orig, marker='o', label='original M -> G (barrier)', color='tab:red')
    ax.plot(alphas, losses_perm, marker='s', label='permuted M -> G (after Git Re-Basin)', color='tab:blue')
    ax.set_xlabel('alpha (0 = M, 1 = G)')
    ax.set_ylabel('train loss (log scale)')
    ax.set_yscale('log')
    ax.set_title('Linear interpolation: does permutation alignment remove the barrier?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = HERE / 'results' / 'fig_permutation_align.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')

    # Summary
    mid_orig = losses_orig[len(alphas) // 2]
    mid_perm = losses_perm[len(alphas) // 2]
    endpoint = max(losses_orig[0], losses_orig[-1])
    print()
    print(f'Midpoint barrier (loss at alpha=0.5):')
    print(f'  Original M -> G:  {mid_orig:.4e}  (ratio vs endpoints: {mid_orig / endpoint:.2e})')
    print(f'  Permuted M -> G:  {mid_perm:.4e}  (ratio vs endpoints: {mid_perm / endpoint:.2e})')
    reduction = mid_orig / mid_perm
    print(f'  Barrier reduction factor: {reduction:.2f}x')
    if reduction > 10:
        print('  -> SIGNIFICANT reduction. Permutation alignment matters.')
    elif reduction > 2:
        print('  -> Moderate reduction. Partial story.')
    else:
        print('  -> Barrier essentially unchanged. M and G are different solutions, not just permuted ones.')


if __name__ == '__main__':
    main()
