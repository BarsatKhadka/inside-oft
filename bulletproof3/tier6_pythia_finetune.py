"""Tier 6: Pythia-160m fine-tuning on a small text-domain (LANGUAGE MODEL task).

WHY this task:
- The other tiers cover supervised classification and algorithmic tasks. Tier 6
  needs to be a real language-model fine-tuning setup, because that's where the
  "structural signatures" claim becomes practically relevant (privacy in LLM
  fine-tuning, etc.).
- We fine-tune Pythia-160m on a small slice of Project Gutenberg's "Pride and
  Prejudice" — clean, well-known text. Train on 200 fixed-length chunks of the
  book, test on 200 different chunks from the same book.
- M (wd=0, many epochs): memorizes the 200 train chunks. Per-chunk perplexity
  on train will drop to ~1, on test it stays at Pythia's general novel
  perplexity.
- G (wd=0.1, same epochs): adapts to the style but doesn't memorize specific
  chunks; train and test perplexity stay similar.

Per-sequence loss difference (train vs test) gives clean MIA AUC.
Effective rank, Hessian top/bot, gradient angle are all computable on the
fine-tuned LM.
"""
import json
import urllib.request
from pathlib import Path
import numpy as np
import torch as t
import torch.nn.functional as F
import torch.optim as optim

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bulletproof3._signatures import compute_full_battery

NUM_SEEDS = 2
EPOCHS = 50
BATCH = 4
CHUNK_LEN = 256
N_TRAIN = 200
N_TEST = 200
DATA_DIR = HERE.parent / 'data'
GUTENBERG_URL = 'https://www.gutenberg.org/files/1342/1342-0.txt'  # Pride and Prejudice


def get_corpus():
    """Download Pride and Prejudice (clean Project Gutenberg text)."""
    p = DATA_DIR / 'pride_and_prejudice.txt'
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        print(f'  downloading {GUTENBERG_URL}')
        urllib.request.urlretrieve(GUTENBERG_URL, p)
    text = open(p, 'r', encoding='utf-8', errors='ignore').read()
    # Strip the Gutenberg header/footer roughly
    start_marker = '*** START OF THE PROJECT GUTENBERG'
    end_marker = '*** END OF THE PROJECT GUTENBERG'
    s = text.find(start_marker)
    e = text.find(end_marker)
    if s != -1: text = text[text.find('\n', s) + 1:]
    if e != -1: text = text[:text.find('\n', e) - 1] if text.find('\n', e) > 0 else text
    return text.strip()


def build_chunks(text, tokenizer, chunk_len, n_train, n_test, seed=0):
    """Tokenize full text and split into fixed-length chunks."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(f'  full corpus: {len(ids)} tokens')
    # Make chunks
    n_chunks = len(ids) // chunk_len
    chunks = [ids[i * chunk_len:(i + 1) * chunk_len] for i in range(n_chunks)]
    rng = np.random.RandomState(seed)
    rng.shuffle(chunks)
    train = chunks[:n_train]
    test = chunks[n_train:n_train + n_test]
    print(f'  train chunks: {len(train)}, test chunks: {len(test)}')
    return train, test


def build_model_and_tokenizer(device):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        name = 'EleutherAI/pythia-160m'
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name).to(device)
        return model, tok, name
    except Exception as e:
        print(f'  Pythia load failed: {e}; falling back to GPT-2')
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        tok.pad_token = tok.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        return model, tok, 'gpt2'


def batch_loss(model, chunks_batch, device):
    """Compute LM loss on a list of token-id chunks."""
    ids = t.tensor(chunks_batch, dtype=t.long, device=device)
    out = model(input_ids=ids, labels=ids)
    return out.loss


@t.no_grad()
def per_chunk_loss(model, chunks, device, batch_size=BATCH):
    """Return per-chunk LM loss (averaged over tokens in each chunk)."""
    losses = []
    for i in range(0, len(chunks), batch_size):
        b = chunks[i:i + batch_size]
        ids = t.tensor(b, dtype=t.long, device=device)
        out = model(input_ids=ids)
        logits = out.logits[:, :-1, :].contiguous()
        targets = ids[:, 1:].contiguous()
        # Per-token cross entropy
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none').reshape(targets.shape)
        per_chunk = per_tok.mean(dim=1).cpu().numpy()
        losses.append(per_chunk)
    return np.concatenate(losses)


def train_pythia(seed, mode, device):
    t.manual_seed(seed); np.random.seed(seed)
    wd = 0.1 if mode == 'G' else 0.0
    model, tok, name = build_model_and_tokenizer(device)
    text = get_corpus()
    train_chunks, test_chunks = build_chunks(text, tok, CHUNK_LEN, N_TRAIN, N_TEST, seed=seed)
    opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=wd)
    print(f'  starting fine-tune: mode={mode}, wd={wd}, epochs={EPOCHS}, lr=5e-5')
    for ep in range(EPOCHS):
        model.train()
        rng = np.random.RandomState(ep + seed * 1000)
        order = rng.permutation(len(train_chunks))
        epoch_loss = 0.0; n_batches = 0
        for i in range(0, len(train_chunks), BATCH):
            batch = [train_chunks[j] for j in order[i:i + BATCH]]
            opt.zero_grad()
            loss = batch_loss(model, batch, device)
            loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1
        if (ep + 1) % 5 == 0:
            model.eval()
            tr_l = per_chunk_loss(model, train_chunks[:64], device).mean()
            te_l = per_chunk_loss(model, test_chunks[:64], device).mean()
            print(f'  ep={ep+1}: train_loss={tr_l:.4f}, test_loss={te_l:.4f}, gap={te_l - tr_l:.4f}')
    return model, tok, train_chunks, test_chunks, name


def run(seed, mode, device):
    model, tok, train_chunks, test_chunks, name = train_pythia(seed, mode, device)
    model.eval()
    # Loss closures (small batch for memory in Hessian computation)
    tr_b = train_chunks[:8]; te_b = test_chunks[:8]
    train_loss_fn = lambda: batch_loss(model, tr_b, device)
    test_loss_fn  = lambda: batch_loss(model, te_b, device)
    # Per-chunk losses for MIA
    tr_losses = per_chunk_loss(model, train_chunks, device)
    te_losses = per_chunk_loss(model, test_chunks, device)
    print(f'  FINAL train_loss={tr_losses.mean():.4f}, test_loss={te_losses.mean():.4f}, '
          f'gap={te_losses.mean() - tr_losses.mean():.4f}')
    print('  computing signatures...')
    bat = compute_full_battery(model, train_loss_fn, test_loss_fn,
                                tr_losses, te_losses, lanczos_k=5, verbose=True)
    bat['mode'] = mode; bat['seed'] = seed; bat['model_name'] = name
    bat['mean_train_loss'] = float(tr_losses.mean())
    bat['mean_test_loss'] = float(te_losses.mean())
    bat['gap_loss'] = float(te_losses.mean() - tr_losses.mean())
    bat['n_train_chunks'] = len(train_chunks)
    bat['n_test_chunks'] = len(test_chunks)
    bat['chunk_len'] = CHUNK_LEN
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
                if 'hessian_eigs_full' in entry:
                    entry['hessian_eigs_full'] = entry['hessian_eigs_full'][:20]
                if 'ranks' in entry:
                    entry['ranks'] = {k: v for k, v in entry['ranks'].items()
                                       if any(s in k.lower() for s in ['mlp', 'embed', 'fc', 'attention', 'dense', 'query', 'key', 'value'])}
                results[mode].append(entry)
                print(f'  gap_loss={entry["gap_loss"]:.4f} '
                      f'mia={entry.get("mia_loss_auc",0):.4f} '
                      f'top={entry["hessian_top_full"]:.3f} '
                      f'bot={entry["hessian_bot_full"]:.3f} '
                      f'cos={entry["cos_grad_train_test"]:.4f}')
            except Exception as e:
                print(f'  error: {e}')
                import traceback; traceback.print_exc()
                results[mode].append({'mode': mode, 'seed': seed, 'error': str(e)})
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
