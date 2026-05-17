#!/bin/bash
#SBATCH --job-name=grok
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Magnolia (Ole Miss MCSR) — L40S GPU nodes (45-50), 46GB VRAM.
#
# Usage:
#   sbatch --job-name=smoke --time=00:15:00 taska/train.slurm taska/configs/G_smoke.yaml
#   sbatch --job-name=G                     taska/train.slurm taska/configs/G.yaml
#   sbatch --job-name=M                     taska/train.slurm taska/configs/M.yaml

set -euo pipefail

CONFIG=${1:?"usage: sbatch train.slurm <path-to-config.yaml>"}

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source venv/bin/activate

echo "[start]  $(date)  job=$SLURM_JOB_ID  host=$(hostname)  config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python -u taska/train.py --config "$CONFIG"

echo "[done]   $(date)"
