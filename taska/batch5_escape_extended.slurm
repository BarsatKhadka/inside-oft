#!/bin/bash
#SBATCH --job-name=batch5_escape
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Batch 5: Extended escape-mechanism comparison.
# 12 mechanisms × 15k epochs each = ~180k total epochs.
# Tests WD/L1/L2 family, SAM family (multiple rho), noise family (multiple std),
# label smoothing family. Comprehensive map of "which escape M's saddle."

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u taska/analysis/escape_mechanisms_extended.py

echo "[done] $(date)"
