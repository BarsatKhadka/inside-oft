#!/bin/bash
#SBATCH --job-name=grok_%x
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# #SBATCH --partition=gpu          # uncomment + set if your cluster needs an explicit partition
# #SBATCH --account=YOUR_ACCOUNT   # uncomment + set if accounting is required

# Usage:
#   sbatch --job-name=G train.slurm configs/G.yaml
#   sbatch --job-name=M train.slurm configs/M.yaml

set -euo pipefail

CONFIG=${1:?"usage: sbatch train.slurm <path-to-config.yaml>"}

echo "=== job info ==="
echo "job id:     $SLURM_JOB_ID"
echo "node:       $SLURMD_NODENAME"
echo "gpu(s):     ${CUDA_VISIBLE_DEVICES:-none}"
echo "config:     $CONFIG"
echo "started:    $(date)"
echo "=================="

# Move to repo root (the directory that contains taska/)
cd "$SLURM_SUBMIT_DIR"

# Activate venv (assumes you created it at ~/inside-oft/venv)
source venv/bin/activate

# Sanity print
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Run training. CONFIG path is relative to repo root, e.g. taska/configs/G.yaml
python -u taska/train.py --config "$CONFIG"

echo "=== finished: $(date) ==="
