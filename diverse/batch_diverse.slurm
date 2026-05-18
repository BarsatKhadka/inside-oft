#!/bin/bash
#SBATCH --job-name=batch_diverse
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# DIVERSE DOMAIN BATCH: test signatures across 3 fundamentally different task types
#   1. MNIST classification (vision)
#   2. Shakespeare character-level LM (autoregressive language)
#   3. Synthetic tabular classification
#
# If the saddle/rank/gradient-asymmetry signatures hold across all three plus
# our existing modular addition and CIFAR-10, that's 5 distinct domains
# supporting a universal claim.

set -uo pipefail

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs diverse/mnist_data
source venv/bin/activate
pip show torchvision >/dev/null 2>&1 || pip install torchvision

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u diverse/diverse_tasks.py

echo "[done] $(date)"
