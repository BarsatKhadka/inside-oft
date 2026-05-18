#!/bin/bash
#SBATCH --job-name=batch_long_lm
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 6 runs (3 seeds × 2 WD) × 50k iters of 4-layer Char LM on tiny shakespeare.
# Tests H1 vs H2: do LMs eventually show signatures?

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u overnight/deep_long_lm.py

echo "[done] $(date)"
