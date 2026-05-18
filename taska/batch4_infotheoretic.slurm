#!/bin/bash
#SBATCH --job-name=batch4_info
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Batch 4: Information-theoretic measurements.
# Quantifies "preserved info" via per-pair MI, weight compressibility, and
# logit-distribution extraction. Should be quick (~30 min) -- no training.

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u taska/analysis/info_theoretic.py

echo "[done] $(date)"
