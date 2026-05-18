#!/bin/bash
#SBATCH --job-name=batch_matrix
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 48 cells (3 archs × 4 tasks × 4 WD) × 5k epochs each. ~8-10 hours on L40S.
# Builds the master cross-product table for the paper.

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u overnight/full_matrix.py

echo "[done] $(date)"
