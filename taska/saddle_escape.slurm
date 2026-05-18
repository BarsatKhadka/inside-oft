#!/bin/bash
#SBATCH --job-name=saddle_escape
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Test the saddle hypothesis: do alternative perturbation forces also escape
# M's saddle, or is WD specific?
#
# 5 mechanisms × 20k epochs each = ~80-100k epochs total. ~30-60 min on L40S.

set -uo pipefail

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u taska/analysis/saddle_escape_mechanisms.py

echo "[done] $(date)"
