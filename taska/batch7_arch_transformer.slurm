#!/bin/bash
#SBATCH --job-name=batch7_archtrans
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Batch 7: Transformer architecture sweep (depth × width).
# 12 architectures × 2 regimes (M/G) = 24 runs × 30k epochs.
# Builds the architecture-invariance table for the paper.

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u taska/analysis/arch_sweep_transformer.py

echo "[done] $(date)"
