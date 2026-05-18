"""bp13: Width x depth invariance heatmap.

d_model in {64, 128, 256, 512, 1024} x layers in {1, 2, 4} x 3 seeds = 45 G runs.

Prediction: G-rank ~flat across width and depth (it's task-determined).
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.optim as optim

from _common import (HERE, LR, P, cross_entropy_hp, effective_rank, eval_acc)
from taska.data import gen_train_test, to_tensors
from taska.model import Transformer

NUM_SEEDS = 3
NUM_EPOCHS = 20000
D_MODELS = [64, 128, 256, 512, 1024]
LAYERS = [1, 2, 4]


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = []
    for d_model in D_MODELS:
        for n_layers in LAYERS:
            for seed in range(NUM_SEEDS):
                num_heads = max(1, d_model // 32)
                # Ensure d_model divisible by num_heads
                while d_model % num_heads != 0:
                    num_heads -= 1
                print(f'\n--- d_model={d_model} layers={n_layers} seed={seed} ---')
                t.manual_seed(seed); np.random.seed(seed)
                try:
                    model = Transformer(p=P, d_model=d_model, num_heads=num_heads,
                                        n_ctx=3, num_layers=n_layers).to(device)
                except Exception as e:
                    print(f'  build failed: {e}')
                    continue
                opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0, betas=(0.9, 0.98))
                tr, te = gen_train_test(p=P, frac_train=0.3, seed=seed)
                tr_in, tr_lab = to_tensors(tr, P, device=device)
                te_in, te_lab = to_tensors(te, P, device=device)
                grok_ep = None
                for ep in range(NUM_EPOCHS):
                    loss = cross_entropy_hp(model(tr_in)[:, -1, :], tr_lab)
                    opt.zero_grad(); loss.backward(); opt.step()
                    if (ep + 1) % 1000 == 0:
                        ta = eval_acc(model, te_in, te_lab)
                        if grok_ep is None and ta >= 0.95: grok_ep = ep + 1
                final_te = eval_acc(model, te_in, te_lab)
                # Rank of deepest MLP W_out
                rank_out = effective_rank(model.blocks[-1].mlp.W_out)
                entry = {
                    'd_model': d_model, 'num_layers': n_layers, 'num_heads': num_heads,
                    'seed': seed, 'final_test_acc': final_te, 'grok_epoch': grok_ep,
                    'rank_W_out_last': rank_out,
                    'rank_W_out_all_layers': [effective_rank(b.mlp.W_out) for b in model.blocks],
                }
                results.append(entry)
                print(f'  test={final_te:.4f}, rank_W_out_last={rank_out:.2f}')
    (HERE / 'results').mkdir(parents=True, exist_ok=True)
    with open(HERE / 'results' / 'bp13_width_depth.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Heatmap summary
    print('\n=== G-rank heatmap (mean across grokked seeds) ===')
    print('  rows = layers, cols = d_model')
    print('  ' + ' '.join(f'{d:>7d}' for d in D_MODELS))
    for L in LAYERS:
        row = []
        for d in D_MODELS:
            cell = [r for r in results if r['d_model'] == d and r['num_layers'] == L
                    and r['final_test_acc'] >= 0.9]
            if cell:
                row.append(f'{np.mean([r["rank_W_out_last"] for r in cell]):>7.2f}')
            else:
                row.append('   -- ')
        print(f'L={L}: ' + ' '.join(row))


if __name__ == '__main__':
    main()
