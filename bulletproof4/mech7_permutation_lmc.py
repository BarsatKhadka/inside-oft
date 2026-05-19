"""mech7: Permutation-aligned linear mode connectivity (Ainsworth 2023 git re-basin).

For tier0 modular and tier2 ResNet, find a neuron permutation that aligns M
to G, then linearly interpolate M_permuted with G and measure the barrier.

If naive LMC has a barrier but permutation-aligned LMC doesn't, M and G are
in the same basin modulo permutation. If both show a barrier, they are
genuinely different solutions.

For MLP / Conv layers, Hungarian matching on per-neuron features.
For Transformer / ViT, this is much harder (attention heads' permutations
constrained by Q/K/V symmetry); we skip those and report it as future work.
"""
import json
import copy
from pathlib import Path
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

DATA_DIR = HERE.parent / 'data'
OUT_DIR = HERE / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- ResNet-18 builder ----------

def make_resnet18():
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def loaders(seed, augment):
    t.manual_seed(seed); np.random.seed(seed)
    if augment:
        tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    else:
        tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    tft = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    tr = torchvision.datasets.CIFAR10(str(DATA_DIR), train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10(str(DATA_DIR), train=False, download=True, transform=tft)
    return (t.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=2),
            t.utils.data.DataLoader(te, batch_size=128, shuffle=False, num_workers=2))


def train(seed, mode, device, epochs=80):
    augment = (mode == 'G'); wd = 5e-4 if mode == 'G' else 0.0
    tl, vl = loaders(seed, augment)
    model = make_resnet18().to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
    return model, tl, vl


# ---------- Permutation alignment (greedy Hungarian per conv layer) ----------

def align_conv_filters_to_target(W_source, W_target):
    """For 2D conv weights [C_out, C_in, kH, kW], find permutation of C_out
    dimension that best matches W_target. Returns permutation indices.
    """
    from scipy.optimize import linear_sum_assignment
    s = W_source.detach().cpu().float().reshape(W_source.shape[0], -1)
    t_ = W_target.detach().cpu().float().reshape(W_target.shape[0], -1)
    # Cosine similarity matrix [C_out_s, C_out_t]
    s_norm = s / (s.norm(dim=1, keepdim=True) + 1e-12)
    t_norm = t_ / (t_.norm(dim=1, keepdim=True) + 1e-12)
    cos = s_norm @ t_norm.T
    # Hungarian: maximize sum of cosines = minimize negative
    row_idx, col_idx = linear_sum_assignment(-cos.numpy())
    # row_idx[i] -> col_idx[i] means source neuron row_idx[i] maps to target neuron col_idx[i]
    # We want a permutation P such that perm[i] = source neuron that should become target neuron i
    perm = np.zeros(cos.shape[0], dtype=np.int64)
    for r, c in zip(row_idx, col_idx):
        perm[c] = r
    return t.tensor(perm)


def naive_lmc(m_state, g_state, model_template, tl, vl, device, alphas):
    """Linear interpolation of weights without alignment."""
    results = []
    interp = copy.deepcopy(model_template).to(device)
    for a in alphas:
        new_sd = {}
        for k in m_state:
            if m_state[k].dtype.is_floating_point:
                new_sd[k] = (1 - a) * m_state[k].to(device) + a * g_state[k].to(device)
            else:
                new_sd[k] = m_state[k].to(device)
        interp.load_state_dict(new_sd)
        interp.eval()
        with t.no_grad():
            tr_l = 0.0; tr_n = 0; te_l = 0.0; te_n = 0
            for i, (x, y) in enumerate(tl):
                if i >= 20: break
                x, y = x.to(device), y.to(device)
                tr_l += F.cross_entropy(interp(x), y, reduction='sum').item(); tr_n += y.size(0)
            for i, (x, y) in enumerate(vl):
                if i >= 20: break
                x, y = x.to(device), y.to(device)
                te_l += F.cross_entropy(interp(x), y, reduction='sum').item(); te_n += y.size(0)
        results.append({'alpha': a, 'train_loss': tr_l / tr_n, 'test_loss': te_l / te_n})
    return results


def compute_lmc_for_tier(seed=0, device='cuda', epochs=80):
    """Train M and G, compute naive LMC and report barrier.
    Permutation-aligned LMC for ResNet is complex due to BN, residual connections;
    for now we just report naive LMC and the Hungarian matching cost for conv1
    as a sanity check of alignment quality.
    """
    print('=== Training M (ResNet-18 CIFAR-10) ===')
    m_model, tl, vl = train(seed, 'M', device, epochs=epochs)
    print('=== Training G (ResNet-18 CIFAR-10) ===')
    g_model, _, _ = train(seed, 'G', device, epochs=epochs)

    m_state = {k: v.detach().cpu() for k, v in m_model.state_dict().items()}
    g_state = {k: v.detach().cpu() for k, v in g_model.state_dict().items()}

    # Naive LMC
    alphas = np.linspace(0, 1, 11).tolist()
    print('=== Naive LMC interpolation ===')
    template = make_resnet18().to(device)
    naive_results = naive_lmc(m_state, g_state, template, tl, vl, device, alphas)
    for r in naive_results:
        print(f'  alpha={r["alpha"]:.2f}: train={r["train_loss"]:.4f} test={r["test_loss"]:.4f}')
    naive_barrier = max(r['test_loss'] for r in naive_results) - max(
        naive_results[0]['test_loss'], naive_results[-1]['test_loss'])

    # Sanity check: conv1 alignment cost
    perm_conv1 = align_conv_filters_to_target(m_state['conv1.weight'],
                                                g_state['conv1.weight'])
    print(f'\nconv1 Hungarian alignment computed; perm[0:8] = {perm_conv1[:8].tolist()}')

    return {
        'seed': seed, 'epochs': epochs,
        'alphas': alphas,
        'naive_lmc_curve': naive_results,
        'naive_barrier_test': naive_barrier,
        'note': 'Full permutation-alignment for ResNet (with BN + residuals) is left as'
                ' future work; this script reports naive LMC barrier as the main result.',
    }


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print('mech7: LMC for ResNet-18 CIFAR-10')
    out = compute_lmc_for_tier(seed=0, device=device, epochs=80)
    with open(OUT_DIR / 'mech7_lmc_resnet18.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nNaive LMC barrier (test): {out["naive_barrier_test"]:.4f}')


if __name__ == '__main__':
    main()
