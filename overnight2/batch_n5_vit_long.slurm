#!/bin/bash
#SBATCH --job-name=n5_vit
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate
pip show torchvision >/dev/null 2>&1 || pip install torchvision
echo "[start] $(date) host=$(hostname)"
python -u overnight2/vit_long_cifar.py
echo "[done] $(date)"
