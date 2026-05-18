"""bp17: Minimum-norm interpolator theory anchor.

For modular addition viewed as a classification problem with one-hot labels,
compute the literal minimum-Frobenius-norm solution among networks that perfectly
fit the training set. Compare its structural signatures (rank, Fourier content)
to G.

Approach: parameterize a 1-hidden-layer ReLU network with fixed (random) first
layer, learn second layer via least-squares (closed form). This gives the
min-norm interpolator in the second layer (the "neural tangent kernel" regime).
Then compute the min-norm solution in the kernel space, project to W_out, and
measure its rank/Fourier content.

If G ~ min-norm interpolator structurally, this connects our findings to the
implicit-bias literature (Belkin, Bartlett, Soudry).
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.nn.functional as F

from _common import HERE, effective_rank

P = 113
N_HIDDEN = 1024  # wide first layer = NTK regime


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {}
    for seed in range(5):
        print(f'\n--- seed={seed} ---')
        np.random.seed(seed); t.manual_seed(seed)
        # Train inputs: all (a, b) with a+b mod P; we use 30% train fraction
        all_pairs = [(a, b) for a in range(P) for b in range(P)]
        rng = np.random.RandomState(seed); rng.shuffle(all_pairs)
        train = all_pairs[:int(0.3 * len(all_pairs))]
        test = all_pairs[int(0.3 * len(all_pairs)):]
        def featurize(pairs):
            # one-hot embedding of (a, b) then random projection
            X = np.zeros((len(pairs), 2 * P))
            for i, (a, b) in enumerate(pairs):
                X[i, a] = 1.0; X[i, P + b] = 1.0
            return X
        W1 = rng.randn(2 * P, N_HIDDEN) / np.sqrt(2 * P)
        b1 = np.zeros(N_HIDDEN)
        def feats(pairs):
            X = featurize(pairs)
            return np.maximum(0, X @ W1 + b1)
        Phi_tr = feats(train)  # [N_tr, N_HIDDEN]
        Phi_te = feats(test)
        Y_tr = np.zeros((len(train), P))
        for i, (a, b) in enumerate(train):
            Y_tr[i, (a + b) % P] = 1.0
        Y_te = np.zeros((len(test), P))
        for i, (a, b) in enumerate(test):
            Y_te[i, (a + b) % P] = 1.0
        # Min-norm interpolator: W2 = Phi_tr^+ Y_tr (Moore-Penrose pseudoinverse)
        W2 = np.linalg.pinv(Phi_tr) @ Y_tr   # [N_HIDDEN, P]
        # Evaluate
        pred_te = (Phi_te @ W2).argmax(1)
        true_te = np.array([(a + b) % P for a, b in test])
        test_acc = float((pred_te == true_te).mean())
        # Rank of W2
        s = np.linalg.svd(W2, compute_uv=False)
        p = s ** 2; p = p / p.sum(); p = p[p > 0]
        eff_rank = float(np.exp(-(p * np.log(p)).sum()))
        # Fourier content of W2 row-aggregated
        # treat W2[:, k] as a function of the input space; compute Fourier coeffs
        # via the kernel basis. As a proxy: examine output dimensions' singular vectors.
        U, S, Vt = np.linalg.svd(W2, full_matrices=False)
        # Vt rows are right singular vectors in output space (size P)
        # Compute FFT of top singular vectors
        top_fft_concentration = []
        for k in range(min(20, len(S))):
            v = Vt[k]
            fft = np.abs(np.fft.fft(v))
            top_fft_concentration.append(float((fft[:5]**2).sum() / (fft**2).sum()))
        entry = {
            'seed': seed, 'N_HIDDEN': N_HIDDEN,
            'test_acc': test_acc,
            'rank_W2': eff_rank,
            'top_sv': float(S[0]),
            'stable_rank': float((S**2).sum() / (S[0]**2)),
            'top_singular_values': S[:20].tolist(),
            'fft_concentration_top20': top_fft_concentration,
        }
        results[f'seed{seed}'] = entry
        print(f'  test_acc={test_acc:.4f}, rank_W2={eff_rank:.2f}, top_sv={S[0]:.4f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp17_min_norm_interpolator.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
