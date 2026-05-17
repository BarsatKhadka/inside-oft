#!/bin/bash
#SBATCH --job-name=trackb_train
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Track B: CIFAR-10 + ResNet-18 in G or M regime.
# Run this script twice (with --export) to get both:
#   sbatch --export=ALL,MODE=G trackb/trackb_train.slurm
#   sbatch --export=ALL,MODE=M trackb/trackb_train.slurm
#
# After both finish and M shows overfitting, submit the rescue separately:
#   sbatch trackb/trackb_rescue.slurm

set -uo pipefail

: "${MODE:?Must set MODE=G or MODE=M via --export}"

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs trackb/cifar_data
source venv/bin/activate

# Install torchvision if missing
pip show torchvision >/dev/null 2>&1 || pip install torchvision

echo "[start] $(date)  host=$(hostname)  MODE=$MODE"
nvidia-smi --query-gpu=name --format=csv

python -u trackb/train_cifar.py --mode "$MODE"

echo "[done] $(date)"
