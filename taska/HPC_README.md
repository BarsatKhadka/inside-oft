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

## Quick sanity check before the big run

Before submitting a 4-hour job, test the pipeline with a tiny config (1000 epochs ≈ 1 min):

```bash
# on the cluster, in an interactive GPU session if available, or as a short sbatch
cp taska/configs/G.yaml taska/configs/G_smoke.yaml
# edit G_smoke.yaml: num_epochs: 1000, out_dir: taska/checkpoints/G_smoke
sbatch --time=00:15:00 --job-name=Gsmoke taska/train.slurm taska/configs/G_smoke.yaml
```

If that completes with non-NaN losses and saves checkpoints, the real run will work.

## Troubleshooting

- **CUDA OOM:** unlikely for this tiny model. If it happens, you're probably on a shared GPU; ask for `--gres=gpu:1 --exclusive` or a bigger memory tier.
- **`source venv/bin/activate` fails:** wrong path — ensure venv is at `~/inside-oft/venv`, or update the path in `train.slurm`.
- **torch can't see CUDA:** wrong torch build for the cluster's CUDA. Reinstall with the matching `--index-url` from PyTorch.
- **Job stuck pending forever:** `squeue -u $USER` will show a reason code in the last column. Common ones: `Priority`, `Resources` (just wait), `QOSMaxJobsLimit` (queue policy), `BadConstraints` (check your SBATCH header).
