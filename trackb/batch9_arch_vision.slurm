#!/bin/bash
#SBATCH --job-name=batch9_archvis
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Batch 9: Vision architecture sweep on CIFAR-10.
# 4 architectures × 2 modes = 8 runs × 100 epochs each.
# ResNet-18, ResNet-50, ViT-Small, MLP. M and G variants.

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs trackb/cifar_data
source venv/bin/activate
pip show torchvision >/dev/null 2>&1 || pip install torchvision

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

python -u trackb/arch_sweep_vision.py

echo "[done] $(date)"
