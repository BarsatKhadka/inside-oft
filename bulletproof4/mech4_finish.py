"""mech4 finish: only the missing cells.

The original mech4 (12 runs total) hit wall-time after 7. Missing:
  - (wd=5e-4, aug=False): need 2 more seeds (seeds 1, 2)
  - (wd=5e-4, aug=True): need 3 seeds (full G baseline)

This script runs ONLY those 5 missing runs and merges them into the existing
mech4_resnet_ablation.json file. Faster than rerunning the full mech4.
"""
import json
from pathlib import Path
import numpy as np
import torch as t

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bulletproof4.mech4_resnet_ablation import run_cell, CELLS  # reuse

# Only the missing cells
MISSING = [
    (5e-4, False, 'wd1_aug0', [1, 2]),       # already have seed 0
    (5e-4, True,  'wd1_aug1', [0, 1, 2]),    # all three
]


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    out_path = HERE / 'results' / 'mech4_resnet_ablation.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Load existing partial results
    if out_path.exists():
        results = json.load(open(out_path))
        print(f'Loaded existing results. Cells: {list(results.keys())}')
    else:
        results = {}
    for wd, aug, name, seeds in MISSING:
        results.setdefault(name, [])
        existing_seeds = {r['seed'] for r in results[name]
                          if 'seed' in r and 'error' not in r}
        for seed in seeds:
            if seed in existing_seeds:
                print(f'  {name} seed={seed} already complete, skipping')
                continue
            print(f'\n=== {name} seed={seed} (wd={wd}, aug={aug}) ===')
            try:
                entry = run_cell(seed, wd, aug, device)
                results[name].append(entry)
                print(f'  test={entry["test_acc"]:.4f} '
                      f'top={entry["hessian_top_full"]:.3f} '
                      f'bot={entry["hessian_bot_full"]:.3f} '
                      f'cos={entry["cos_grad_train_test"]:.4f} '
                      f'mia={entry.get("mia_loss_auc", 0):.4f}')
            except Exception as e:
                print(f'  error: {e}')
                import traceback; traceback.print_exc()
                results[name].append({'seed': seed, 'wd': wd, 'augment': aug, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
