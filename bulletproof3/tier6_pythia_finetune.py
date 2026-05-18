"""Tier 6: Pythia-160m fine-tuning on a tiny synthetic task.

The task: 'reverse 4-token sequences' or 'binary-string parity'. We use sequences
from a small constructed set so memorization vs generalization is cleanly defined.

M: full fine-tune, no WD, 100 epochs over a small set (overfits)
G: full fine-tune, WD=0.1, early-stopped
2 seeds each.

If transformers' library not available, falls back to GPT-2-small via torch.
"""
import json
from pathlib import Path
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bulletproof3._signatures import compute_full_battery

NUM_SEEDS = 2
EPOCHS = 100
BATCH = 8


def build_dataset(n_train=200, n_test=200, seed=0):
    """Generate a parity-by-character task. Given 8-char binary string,
    output 'E' if even parity, 'O' if odd. We can verify generalization
    by testing on unseen 8-bit strings."""
    rng = np.random.RandomState(seed)
    all_strings = []
    for i in range(256):
        s = format(i, '08b')
        label = 'E' if s.count('1') % 2 == 0 else 'O'
        all_strings.append((s, label))
    rng.shuffle(all_strings)
    train = all_strings[:n_train]
    test = all_strings[n_train:n_train + n_test]
    return train, test


def encode(text, tokenizer, max_len=32, device='cuda'):
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:max_len] + [tokenizer.eos_token_id] * (max_len - len(ids))
    return t.tensor(ids[:max_len], device=device)


def build_model_and_tokenizer(seed, device):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        name = 'EleutherAI/pythia-160m'
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name)
        model.to(device)
        return model, tok, name
    except Exception as e:
        print(f'  Pythia load failed: {e}; falling back to GPT-2')
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        tok.pad_token = tok.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.to(device)
        return model, tok, 'gpt2'


def format_example(s, label): return f'{s}={label}'


def get_loss(model, batch_texts, tokenizer, device, max_len=32):
    ids = t.stack([encode(txt, tokenizer, max_len, device) for txt in batch_texts])
    out = model(ids, labels=ids)
    return out.loss


def train_pythia(seed, mode, device):
    t.manual_seed(seed); np.random.seed(seed)
    wd = 0.1 if mode == 'G' else 0.0
    model, tok, name = build_model_and_tokenizer(seed, device)
    train, test = build_dataset(n_train=128, n_test=128, seed=seed)
    train_texts = [format_example(s, l) for s, l in train]
    test_texts = [format_example(s, l) for s, l in test]
    opt = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=wd)
    for ep in range(EPOCHS):
        model.train()
        np.random.shuffle(train_texts)
        for i in range(0, len(train_texts), BATCH):
            batch = train_texts[i:i+BATCH]
            opt.zero_grad()
            get_loss(model, batch, tok, device).backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with t.no_grad():
                test_loss = float(np.mean([
                    get_loss(model, test_texts[i:i+BATCH], tok, device).item()
                    for i in range(0, len(test_texts), BATCH)]))
            print(f'  ep={ep+1}: test_loss={test_loss:.4f}')
    return model, tok, train_texts, test_texts, name


def run(seed, mode, device):
    model, tok, train_texts, test_texts, name = train_pythia(seed, mode, device)
    model.eval()
    # Build batches for signature computation (small)
    tr_batch = train_texts[:16]; te_batch = test_texts[:16]
    train_loss_fn = lambda: get_loss(model, tr_batch, tok, device)
    test_loss_fn  = lambda: get_loss(model, te_batch, tok, device)
    # Per-example losses for MIA
    @t.no_grad()
    def per_ex(texts):
        losses = []
        for txt in texts:
            losses.append(get_loss(model, [txt], tok, device).item())
        return np.array(losses)
    tr_losses = per_ex(train_texts); te_losses = per_ex(test_texts)
    print('  computing signatures...')
    # Only top-k a few eigenvalues — model has ~160M params, Lanczos is expensive
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=5, verbose=True)
    bat['mode'] = mode; bat['seed'] = seed; bat['model_name'] = name
    bat['mean_train_loss'] = float(tr_losses.mean())
    bat['mean_test_loss'] = float(te_losses.mean())
    bat['gap_loss'] = float(te_losses.mean() - tr_losses.mean())
    return bat


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    results = {'M': [], 'G': []}
    out_path = HERE / 'results' / 'tier6_pythia_finetune.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for mode in ['M', 'G']:
        for seed in range(NUM_SEEDS):
            print(f'\n=== {mode} seed={seed} ===')
            try:
                entry = run(seed, mode, device)
                # Cull big lists from results before saving
                if 'hessian_eigs_full' in entry: entry['hessian_eigs_full'] = entry['hessian_eigs_full'][:20]
                if 'ranks' in entry:
                    entry['ranks'] = {k: v for k, v in entry['ranks'].items() if 'mlp' in k.lower() or 'embed' in k.lower() or 'fc' in k.lower() or 'attention' in k.lower()}
                results[mode].append(entry)
                print(f'  test_loss={entry["mean_test_loss"]:.4f} '
                      f'gap={entry["gap_loss"]:.4f} '
                      f'top={entry["hessian_top_full"]:.3f} '
                      f'bot={entry["hessian_bot_full"]:.3f} '
                      f'cos={entry["cos_grad_train_test"]:.4f}')
            except Exception as e:
                print(f'  error: {e}')
                results[mode].append({'mode': mode, 'seed': seed, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
