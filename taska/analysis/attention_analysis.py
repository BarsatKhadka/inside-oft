"""D. Attention pattern analysis: are M and G using attention differently?

We haven't touched attention weights yet. Investigate:

  - Average attention pattern at position 2 (the "=" token output)
    across all training pairs and across all test pairs.
  - Does M's attention pattern differ between train and test inputs?
    (If yes, attention might be part of the "memorization detector".)
  - Compare M's attention to G's.

Also:
  - SVD of W_K, W_Q, W_V, W_O for each head.
  - Compare ranks. Does M have higher-rank attention weights too?

  - Attention pattern is shape (heads=4, src_pos=3, dst_pos=3). At position 2
    (above "="), the only meaningful patterns are the weights given to
    positions 0, 1, 2.

Usage:
    python taska/analysis/attention_analysis.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import torch as t

from data import gen_train_test, to_tensors
from model import Transformer

P = 113


def load_model(ckpt):
    model = Transformer(p=P, d_model=128, num_heads=4, n_ctx=3, num_layers=1)
    state = t.load(ckpt, map_location='cpu', weights_only=True)['model']
    model.load_state_dict(state)
    model.eval()
    return model


def effective_rank(W):
    s = t.linalg.svdvals(W)
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(t.exp(-(p * t.log(p)).sum()))


@t.no_grad()
def attention_pattern(model, inputs):
    """Return attention weights at position 2, averaged across inputs.
    Shape: (heads=4, src_pos=3)."""
    x = model.embed(inputs)
    x = model.pos_embed(x)
    block = model.blocks[0]
    attn = block.attn

    k = t.einsum('ihd,bpd->biph', attn.W_K, x)
    q = t.einsum('ihd,bpd->biph', attn.W_Q, x)
    attn_scores = t.einsum('biph,biqh->biqp', k, q)
    masked = t.tril(attn_scores) - 1e10 * (1 - attn.mask[:x.shape[-2], :x.shape[-2]])
    attn_matrix = t.nn.functional.softmax(masked / np.sqrt(attn.d_head), dim=-1)
    # attn_matrix shape: (batch, heads, query_pos=3, key_pos=3)
    # We want position 2 (query at "="), averaged across batch
    return attn_matrix[:, :, -1, :].mean(dim=0)   # (heads, src_pos=3)


def main():
    train_pairs, test_pairs = gen_train_test(p=P, frac_train=0.3, seed=0)
    train_in, _ = to_tensors(train_pairs, P, device='cpu')
    test_in,  _ = to_tensors(test_pairs,  P, device='cpu')

    print(f'{"model":>5}  {"matrix":>5}  {"effective_rank":>15}')
    for name, ckpt in [
        ('M', HERE / 'checkpoints' / 'M' / 'final.pt'),
        ('G', HERE / 'checkpoints' / 'G' / 'final.pt'),
    ]:
        model = load_model(ckpt)
        for matrix_name in ['W_K', 'W_Q', 'W_V', 'W_O']:
            W = getattr(model.blocks[0].attn, matrix_name).detach()
            # Reshape if needed (heads x d_head x d_model for W_K/Q/V, d_model x (heads*d_head) for W_O)
            W_flat = W.reshape(W.shape[0] * (W.shape[1] if W.ndim == 3 else 1), -1)
            er = effective_rank(W_flat)
            print(f'{name:>5}  {matrix_name:>5}  {er:>15.2f}')

    # Attention patterns
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row, name in enumerate(['M', 'G']):
        ckpt = HERE / 'checkpoints' / name / 'final.pt'
        model = load_model(ckpt)
        attn_train = attention_pattern(model, train_in).numpy()    # (heads, src_pos)
        attn_test  = attention_pattern(model, test_in).numpy()

        print(f'\n=== {name} attention pattern at position 2 (averaged) ===')
        print(f'{"head":>5}  {"src=0":>8}  {"src=1":>8}  {"src=2":>8}  (train vs test)')
        for h in range(4):
            print(f'{h:>5}  {attn_train[h, 0]:.4f}  {attn_train[h, 1]:.4f}  {attn_train[h, 2]:.4f}  | '
                  f'{attn_test[h, 0]:.4f}  {attn_test[h, 1]:.4f}  {attn_test[h, 2]:.4f}')

        # Plot bar chart for train and test
        for col, (split, attn) in enumerate([('train', attn_train), ('test', attn_test)]):
            ax = axes[row, col]
            x = np.arange(4)
            ax.bar(x - 0.25, attn[:, 0], 0.25, label='to pos 0 (a)')
            ax.bar(x,        attn[:, 1], 0.25, label='to pos 1 (b)')
            ax.bar(x + 0.25, attn[:, 2], 0.25, label='to pos 2 (=)')
            ax.set_xticks(x)
            ax.set_xticklabels([f'head {h}' for h in range(4)])
            ax.set_ylabel('avg attention weight')
            ax.set_title(f'{name} attention at pos 2 ({split})')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Attention patterns: do M and G route information differently?")
    fig.tight_layout()
    out = HERE / 'results' / 'fig_attention.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
