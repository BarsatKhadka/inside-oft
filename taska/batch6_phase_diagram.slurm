#!/bin/bash
#SBATCH --job-name=batch6_phase
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Batch 6: Phase diagram.
# 2D grid in (WD, frac_train) — 24 runs × 20k epochs each.
# Maps the catastrophic / grokking / benign / confusion regimes.
# For each cell record: final test acc AND final rank, so we can show that
# the structural signature varies smoothly across the phase diagram.

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u taska/analysis/phase_diagram.py

echo "[done] $(date)"
