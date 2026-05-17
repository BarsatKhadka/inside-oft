"""Plot the train/test loss + accuracy curves for G and M from history.json.

Usage:
    python taska/analysis/plot_history.py

Output:
    taska/results/fig_grokking.png

Expected:
    G — train loss drops fast (~epoch 100), test loss stays high until ~epoch
        10-20k, then suddenly drops to ~0. Classic grokking shape.
    M — train loss drops fast and stays at 0. Test loss never drops; in fact
        slowly climbs as the model gets more confident in wrong answers.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent  # taska/
G_HIST = HERE / 'checkpoints' / 'G' / 'history.json'
M_HIST = HERE / 'checkpoints' / 'M' / 'history.json'
OUT = HERE / 'results' / 'fig_grokking.png'

def load(p):
    with open(p) as f:
        return json.load(f)

g = load(G_HIST)
m = load(M_HIST)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Loss panel (log-y)
ax = axes[0]
ax.plot(g['epoch'], g['train_loss'], label='G train', color='C0', linestyle='-')
ax.plot(g['epoch'], g['test_loss'],  label='G test',  color='C0', linestyle='--')
ax.plot(m['epoch'], m['train_loss'], label='M train', color='C3', linestyle='-')
ax.plot(m['epoch'], m['test_loss'],  label='M test',  color='C3', linestyle='--')
ax.set_yscale('log')
ax.set_xlabel('epoch')
ax.set_ylabel('loss (log scale)')
ax.set_title('Loss: G (with wd) vs M (no wd)')
ax.legend()
ax.grid(True, alpha=0.3)

# Accuracy panel
ax = axes[1]
ax.plot(g['epoch'], g['train_acc'], label='G train', color='C0', linestyle='-')
ax.plot(g['epoch'], g['test_acc'],  label='G test',  color='C0', linestyle='--')
ax.plot(m['epoch'], m['train_acc'], label='M train', color='C3', linestyle='-')
ax.plot(m['epoch'], m['test_acc'],  label='M test',  color='C3', linestyle='--')
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy: G vs M')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f'saved -> {OUT}')

# Print key milestones to console
def find_grok_epoch(epochs, test_accs, threshold=0.99):
    for e, a in zip(epochs, test_accs):
        if a >= threshold:
            return e
    return None

g_grok = find_grok_epoch(g['epoch'], g['test_acc'])
m_grok = find_grok_epoch(m['epoch'], m['test_acc'])

print()
print(f"G: final train_acc {g['train_acc'][-1]:.4f}  test_acc {g['test_acc'][-1]:.4f}")
print(f"M: final train_acc {m['train_acc'][-1]:.4f}  test_acc {m['test_acc'][-1]:.4f}")
print(f"G grokked at epoch: {g_grok}")
print(f"M grokked at epoch: {m_grok}  (None = never)")
