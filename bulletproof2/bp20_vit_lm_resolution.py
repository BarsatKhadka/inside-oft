"""bp20: ViT + LM scope resolution.

Two halves:
  A) ViT-Tiny on CIFAR-10 with proper budget (1000 epochs), 3 seeds each of M and G.
  B) Tiny character-LM on Shakespeare, 3 seeds each of M and G, 50k iters.

Decides whether to claim universality or commit to scope.

Note: ViT-Tiny built from scratch (no pretrain). Keeps it cheap.
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

from _common import HERE, effective_rank

NUM_SEEDS = 3
DATA_DIR = HERE.parent / 'data'

# -----------------------
# ViT-Tiny
# -----------------------
class ViTTiny(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=192, depth=12, heads=3, mlp_ratio=4, n_classes=10):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2
        self.patch_emb = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(t.randn(1, n_patches + 1, dim) * 0.02)
        self.cls = nn.Parameter(t.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * mlp_ratio, dropout=0.0,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.patch_emb(x).flatten(2).transpose(1, 2)  # [B, N, D]
        cls = self.cls.expand(x.size(0), -1, -1)
        x = t.cat([cls, x], 1) + self.pos_emb
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x[:, 0]))


def vit_loaders(seed, augment):
    t.manual_seed(seed); np.random.seed(seed)
    if augment:
        tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    else:
        tf_train = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    train = torchvision.datasets.CIFAR10(str(DATA_DIR), train=True, download=True, transform=tf_train)
    test = torchvision.datasets.CIFAR10(str(DATA_DIR), train=False, download=True, transform=tf_test)
    return (t.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=2),
            t.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=2))


def vit_run(seed, mode, device, n_epochs=400):
    augment = (mode == 'G'); wd = 5e-4 if mode == 'G' else 0.0
    tl, vl = vit_loaders(seed, augment)
    model = ViTTiny().to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for ep in range(n_epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
        if (ep + 1) % 50 == 0:
            model.eval(); c, n = 0, 0
            with t.no_grad():
                for x, y in vl:
                    x, y = x.to(device), y.to(device)
                    c += (model(x).argmax(1) == y).sum().item(); n += y.size(0)
            print(f'  ep={ep+1}: test={c/n:.4f}')
    model.eval()
    c_te, n_te, c_tr, n_tr = 0, 0, 0, 0
    with t.no_grad():
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            c_tr += (model(x).argmax(1) == y).sum().item(); n_tr += y.size(0)
        for x, y in vl:
            x, y = x.to(device), y.to(device)
            c_te += (model(x).argmax(1) == y).sum().item(); n_te += y.size(0)
    # Ranks of all linear layers
    ranks = {n: effective_rank(p) for n, p in model.named_parameters()
             if p.ndim == 2 and 'weight' in n}
    return {
        'mode': mode, 'seed': seed,
        'train_acc': c_tr / n_tr, 'test_acc': c_te / n_te,
        'mean_block0_attn_rank': ranks.get('blocks.0.self_attn.in_proj_weight', None),
        'mean_block0_mlp_rank': ranks.get('blocks.0.linear1.weight', None),
        'ranks': ranks,
    }


# -----------------------
# Char-LM on Shakespeare
# -----------------------
class CharLM(nn.Module):
    def __init__(self, vocab, dim=128, depth=4, heads=4, ctx=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(ctx, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, vocab)
        self.ctx = ctx
        mask = t.triu(t.ones(ctx, ctx), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        T_ = x.size(1)
        h = self.tok(x) + self.pos(t.arange(T_, device=x.device))
        for blk in self.blocks: h = blk(h, src_mask=self.mask[:T_, :T_])
        return self.head(self.norm(h))


def get_shakespeare():
    import urllib.request
    p = DATA_DIR / 'shakespeare.txt'
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, p)
    text = open(p, 'r').read()
    chars = sorted(set(text)); ctoi = {c: i for i, c in enumerate(chars)}
    data = np.array([ctoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(data))
    return data[:split], data[split:], len(chars)


def lm_run(seed, mode, device, n_iters=50000, ctx=128, batch=64):
    train, val, vocab = get_shakespeare()
    t.manual_seed(seed); np.random.seed(seed)
    model = CharLM(vocab, ctx=ctx).to(device)
    wd = 1e-3 if mode == 'G' else 0.0
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
    def get_batch(arr):
        idx = np.random.randint(0, len(arr) - ctx - 1, batch)
        x = t.tensor(np.stack([arr[i:i+ctx] for i in idx]), device=device)
        y = t.tensor(np.stack([arr[i+1:i+ctx+1] for i in idx]), device=device)
        return x, y
    for it in range(n_iters):
        x, y = get_batch(train)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 5000 == 0:
            model.eval()
            with t.no_grad():
                x_te, y_te = get_batch(val)
                l_te = F.cross_entropy(model(x_te).reshape(-1, vocab), y_te.reshape(-1)).item()
            print(f'  it={it+1}: train_loss={loss.item():.4f}, val_loss={l_te:.4f}')
            model.train()
    # Final eval
    model.eval(); tr_losses = []; te_losses = []
    with t.no_grad():
        for _ in range(20):
            x, y = get_batch(train)
            tr_losses.append(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())
            x, y = get_batch(val)
            te_losses.append(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())
    ranks = {n: effective_rank(p) for n, p in model.named_parameters()
             if p.ndim == 2 and 'weight' in n}
    return {
        'mode': mode, 'seed': seed,
        'train_loss': float(np.mean(tr_losses)), 'val_loss': float(np.mean(te_losses)),
        'gap': float(np.mean(te_losses) - np.mean(tr_losses)),
        'ranks': ranks,
        'mean_mlp_rank': float(np.mean([v for k, v in ranks.items() if 'linear1' in k])),
    }


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'vit': [], 'lm': []}
    print('=== ViT-Tiny CIFAR-10 ===')
    for mode in ['M', 'G']:
        for seed in range(NUM_SEEDS):
            print(f'\n--- ViT {mode} seed={seed} ---')
            results['vit'].append(vit_run(seed, mode, device, n_epochs=400))
            (HERE / 'results').mkdir(parents=True, exist_ok=True)
            with open(HERE / 'results' / 'bp20_vit_lm.json', 'w') as f:
                json.dump(results, f, indent=2)
    print('\n=== CharLM Shakespeare ===')
    for mode in ['M', 'G']:
        for seed in range(NUM_SEEDS):
            print(f'\n--- LM {mode} seed={seed} ---')
            results['lm'].append(lm_run(seed, mode, device, n_iters=50000))
            with open(HERE / 'results' / 'bp20_vit_lm.json', 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
