#!/bin/bash
#SBATCH --job-name=batch8_task
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Batch 8: Task diversity sweep.
# 3 operations × 4 primes = 12 G models × 30k epochs.
# Tests: rank-vs-task-complexity scaling.

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u taska/analysis/task_diversity_sweep.py

echo "[done] $(date)"
