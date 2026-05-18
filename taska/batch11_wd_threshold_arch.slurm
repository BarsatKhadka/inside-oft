#!/bin/bash
#SBATCH --job-name=batch11_wdth
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Batch 11: WD threshold across architectures.
# 4 architectures × 6 WD values = 24 fresh training runs × 15k epochs.
# Tests: is the 0.5 WD threshold universal or arch-specific?

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u taska/analysis/wd_threshold_per_arch.py

echo "[done] $(date)"
