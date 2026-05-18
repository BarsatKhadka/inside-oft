#!/bin/bash
#SBATCH --job-name=batch10_xarch
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Batch 10: Cross-architecture escape mechanism test.
# 3 architectures × 4 mechanisms = 12 rescue runs, plus initial M training
# for each. Tests if the "only norm-based escapes" pattern holds across
# 1L Transformer, 4L Transformer, and MLP.

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u taska/analysis/cross_arch_escape.py

echo "[done] $(date)"
