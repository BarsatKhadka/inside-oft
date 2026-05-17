#!/bin/bash
#SBATCH --job-name=analysis_batch
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs the five new analysis scripts (A through E) sequentially.
# All operate on existing checkpoints in taska/checkpoints/; no training.
#
# Usage:
#   sbatch taska/analysis_batch.slurm

set -euo pipefail

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
echo "A. local_generalization.py"
echo "=============================================="
python -u taska/analysis/local_generalization.py

echo
echo "=============================================="
echo "B. memorization_quality.py"
echo "=============================================="
python -u taska/analysis/memorization_quality.py

echo
echo "=============================================="
echo "C. neuron_organization.py"
echo "=============================================="
python -u taska/analysis/neuron_organization.py

echo
echo "=============================================="
echo "D. attention_analysis.py"
echo "=============================================="
python -u taska/analysis/attention_analysis.py

echo
echo "=============================================="
echo "E. capacity_test.py"
echo "=============================================="
python -u taska/analysis/capacity_test.py

echo
echo "[done] $(date)"
