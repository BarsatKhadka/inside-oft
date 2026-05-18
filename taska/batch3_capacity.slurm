#!/bin/bash
#SBATCH --job-name=batch3_capacity
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Batch 3: Capacity + depth scaling.
# Train G models at d_model ∈ {64, 128, 256, 512} × num_layers ∈ {1, 2}
# Determines whether converged rank is task-determined or capacity-determined.

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u taska/analysis/capacity_depth_scaling.py

echo "[done] $(date)"
