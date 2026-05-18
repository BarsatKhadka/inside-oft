#!/bin/bash
#SBATCH --job-name=tier4_vit_s
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=bulletproof3/logs/%x_%j.out
#SBATCH --error=bulletproof3/logs/%x_%j.err
set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p bulletproof3/logs
source venv/bin/activate
echo "[start] $(date) host=$(hostname)"
python -u bulletproof3/tier4_vit_small_cifar100.py
echo "[done] $(date)"
