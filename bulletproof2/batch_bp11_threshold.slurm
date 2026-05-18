#!/bin/bash
#SBATCH --job-name=bp11_threshold
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=bulletproof2/logs/%x_%j.out
#SBATCH --error=bulletproof2/logs/%x_%j.err
set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p bulletproof2/logs
source venv/bin/activate
echo "[start] $(date) host=$(hostname)"
python -u bulletproof2/bp11_threshold_fine.py
echo "[done] $(date)"
