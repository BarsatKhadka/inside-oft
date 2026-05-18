"""Fix bp13: width x depth heatmap.

Bug: output was empty. Causes: likely OOM at d_model=1024 + 4 layers + 20k epochs
on small modular data (the issue is gradient bookkeeping holding refs).

Fix:
  - Catch OOM per cell and skip
  - Save incrementally after each cell
  - Smaller eval batch on big models
  - 15k epochs instead of 20k (still enough for d_model<=512)
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.optim as optim

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from taska.data import gen_train_test, to_tensors
from taska.model import Transformer
from bulletproof3._signatures import effective_rank


P = 113
LR = 1e-3
NUM_EPOCHS = 15000
NUM_SEEDS = 3
D_MODELS = [64, 128, 256, 512]   # dropped 1024 to avoid OOM
LAYERS = [1, 2, 4]


def cross_entropy_hp(logits, labels):
    lp = t.nn.functional.log_softmax(logits.to(t.float64), dim=-1)
    return -lp[t.arange(labels.shape[0]), labels].mean()


@t.no_grad()
def eval_acc(model, inputs, labels):
    return (model(inputs)[:, -1, :].argmax(dim=-1) == labels).float().mean().item()


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'bp_fix_widthdepth.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for d_model in D_MODELS:
        for n_layers in LAYERS:
            for seed in range(NUM_SEEDS):
                num_heads = max(1, d_model // 32)
                while d_model % num_heads != 0:
                    num_heads -= 1
                print(f'\n--- d_model={d_model} layers={n_layers} seed={seed} heads={num_heads} ---')
                try:
                    t.manual_seed(seed); np.random.seed(seed)
                    model = Transformer(p=P, d_model=d_model, num_heads=num_heads,
                                        n_ctx=3, num_layers=n_layers).to(device)
                    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0,
                                       betas=(0.9, 0.98))
                    tr, te = gen_train_test(p=P, frac_train=0.3, seed=seed)
                    tr_in, tr_lab = to_tensors(tr, P, device=device)
                    te_in, te_lab = to_tensors(te, P, device=device)
                    grok_ep = None
                    for ep in range(NUM_EPOCHS):
                        loss = cross_entropy_hp(model(tr_in)[:, -1, :], tr_lab)
                        opt.zero_grad(); loss.backward(); opt.step()
                        if (ep + 1) % 1000 == 0:
                            ta = eval_acc(model, te_in, te_lab)
                            if grok_ep is None and ta >= 0.95:
                                grok_ep = ep + 1
                    final_te = eval_acc(model, te_in, te_lab)
                    rank_out = effective_rank(model.blocks[-1].mlp.W_out)
                    entry = {
                        'd_model': d_model, 'num_layers': n_layers,
                        'num_heads': num_heads, 'seed': seed,
                        'final_test_acc': final_te, 'grok_epoch': grok_ep,
                        'rank_W_out_last': rank_out,
                        'rank_W_out_per_block': [effective_rank(b.mlp.W_out) for b in model.blocks],
                    }
                    results.append(entry)
                    print(f'  test={final_te:.4f}, rank={rank_out:.2f}')
                    del model, opt, tr_in, tr_lab, te_in, te_lab
                    t.cuda.empty_cache() if device.type == 'cuda' else None
                except t.cuda.OutOfMemoryError as e:
                    print(f'  OOM, skip')
                    t.cuda.empty_cache()
                    results.append({
                        'd_model': d_model, 'num_layers': n_layers, 'seed': seed,
                        'error': 'OOM',
                    })
                except Exception as e:
                    print(f'  error: {e}')
                    results.append({
                        'd_model': d_model, 'num_layers': n_layers, 'seed': seed,
                        'error': str(e),
                    })
                # Incremental save
                with open(out_path, 'w') as f:
                    json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
