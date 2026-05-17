#!/bin/bash
#SBATCH --job-name=trackb_rescue
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Track B rescue: load an overfit M_CIFAR checkpoint, continue with weight
# decay turned on. Tests whether the rescue finding from Track A generalizes
# to standard CIFAR overfitting.
#
# Pass the rescue source via --export:
#   sbatch --export=ALL,RESCUE_FROM=trackb/checkpoints/M/final.pt trackb/trackb_rescue.slurm
#
# To compare rescue from different stages of overfitting:
#   sbatch --export=ALL,RESCUE_FROM=trackb/checkpoints/M/epoch_100.pt trackb/trackb_rescue.slurm
#   sbatch --export=ALL,RESCUE_FROM=trackb/checkpoints/M/epoch_400.pt trackb/trackb_rescue.slurm

set -uo pipefail
: "${RESCUE_FROM:?Must set RESCUE_FROM=path/to/checkpoint via --export}"

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs trackb/cifar_data
source venv/bin/activate
pip show torchvision >/dev/null 2>&1 || pip install torchvision

echo "[start] $(date)  host=$(hostname)  RESCUE_FROM=$RESCUE_FROM"
nvidia-smi --query-gpu=name --format=csv

python -u trackb/train_cifar.py --mode rescue --rescue_from "$RESCUE_FROM"

echo "[done] $(date)"
