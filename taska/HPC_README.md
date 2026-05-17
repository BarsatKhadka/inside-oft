# HPC submission notes

Cluster path assumed: `~/inside-oft/` with the repo cloned and `venv/` at the repo root.

## One-time setup (on the cluster)

```bash
cd ~/inside-oft
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p logs taska/checkpoints/G taska/checkpoints/M
```

If `requirements.txt` doesn't pin a CUDA-matching torch build, install torch explicitly from the PyTorch index for your CUDA version, e.g.:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Check your cluster's CUDA version with `nvidia-smi` or `module avail cuda`.

## Update the SBATCH header

Open `taska/train.slurm` and check / set:

- `--time=04:00:00` — 50k epochs on a modern GPU is ~30–60 min. 4h is safety margin. Lower if your queue is tight.
- `--mem=32G` — overkill but cheap.
- `--gres=gpu:1` — adjust if your cluster wants `--gres=gpu:a100:1` or similar.
- `--partition=...` — uncomment if your cluster requires it. Common names: `gpu`, `gpu-short`, `gpu-long`.
- `--account=...` — uncomment if accounting is required.

Find what your cluster expects:
```bash
sinfo                       # see partitions
sacctmgr show assoc user=$USER format=Account,Partition,QOS    # see what you're allowed to use
```

## Submitting

From the repo root (`~/inside-oft`):

### Step 1 — smoke run first (1–2 min, ~500 epochs)

Always do this before the full job. Catches torch/CUDA/path issues in 2 minutes
instead of 1 hour.

```bash
sbatch --job-name=smoke --time=00:15:00 taska/train.slurm taska/configs/G_smoke.yaml
```

Watch the log until it finishes:

```bash
squeue -u $USER
tail -f logs/smoke_*.out
```

Expected output (memorization phase — no grokking in 500 epochs, that's fine):

```
epoch   0  train_loss 4.76e+00  test_loss 4.77e+00  train_acc 0.9%   test_acc 0.9%
epoch 150  train_loss 1.02e-01  test_loss 1.42e+01  train_acc 100%   test_acc 3.6%
epoch 499  train_loss 7.79e-04  test_loss 1.84e+01  train_acc 100%   test_acc 5.8%
```

Train loss should drop to near-zero, train accuracy hit 100%, test loss climb.
**Paste the log output back to claude before launching the full run** so we
verify nothing weird happened.

Then clean up the smoke checkpoints:

```bash
rm -rf taska/checkpoints/G_smoke
```

### Step 2 — full runs (50k epochs each, ~30–60 min on a modern GPU)

```bash
sbatch --job-name=G taska/train.slurm taska/configs/G.yaml
sbatch --job-name=M taska/train.slurm taska/configs/M.yaml
```

Both run independently. If GPUs are scarce, M waits in queue until G finishes.

## Checking on jobs

```bash
squeue -u $USER                       # queue status
tail -f logs/G_<jobid>.out            # live training log
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,AllocTRES   # post-mortem
```

## Outputs

After successful runs:

```
taska/checkpoints/G/
  init.pt
  epoch_1000.pt
  epoch_2000.pt
  ...
  final.pt
  config.yaml
  history.json

taska/checkpoints/M/    # same structure
```

Plus SLURM logs under `logs/`.

## Pulling results back to your laptop

From your laptop (PowerShell on Windows works the same):

```bash
# everything (~50 checkpoints x ~1 MB each = ~100 MB total for both runs)
rsync -avz magnolia1:~/inside-oft/taska/checkpoints/ ./taska/checkpoints/
rsync -avz magnolia1:~/inside-oft/logs/                ./logs/

# or just the histories + final + init if you want a quick look first (tiny)
scp 'magnolia1:~/inside-oft/taska/checkpoints/G/{init,final}.pt' taska/checkpoints/G/
scp magnolia1:~/inside-oft/taska/checkpoints/G/history.json       taska/checkpoints/G/
scp 'magnolia1:~/inside-oft/taska/checkpoints/M/{init,final}.pt' taska/checkpoints/M/
scp magnolia1:~/inside-oft/taska/checkpoints/M/history.json       taska/checkpoints/M/
```

## Troubleshooting

- **CUDA OOM:** unlikely for this tiny model. If it happens, you're probably on a shared GPU; ask for `--gres=gpu:1 --exclusive` or a bigger memory tier.
- **`source venv/bin/activate` fails:** wrong path — ensure venv is at `~/inside-oft/venv`, or update the path in `train.slurm`.
- **torch can't see CUDA:** wrong torch build for the cluster's CUDA. Reinstall with the matching `--index-url` from PyTorch.
- **Job stuck pending forever:** `squeue -u $USER` will show a reason code in the last column. Common ones: `Priority`, `Resources` (just wait), `QOSMaxJobsLimit` (queue policy), `BadConstraints` (check your SBATCH header).
