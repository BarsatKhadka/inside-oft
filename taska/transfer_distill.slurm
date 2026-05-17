#!/bin/bash
#SBATCH --job-name=transfer_distill
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs both:
#   α — transfer_test.py (M/G/random body + fresh unembed on a-b mod p)
#   β — distillation.py  (student trained with M as teacher)
#
# Usage:
#   sbatch taska/transfer_distill.slurm

set -uo pipefail   # no -e -- run both even if one fails

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

echo
echo "=============================================="
echo "α. transfer_test.py"
echo "=============================================="
python -u taska/analysis/transfer_test.py

echo
echo "=============================================="
echo "β. distillation.py"
echo "=============================================="
python -u taska/analysis/distillation.py

echo
echo "[done] $(date)"
