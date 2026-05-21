#!/bin/bash
#SBATCH --job-name=petzka_trajectory
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=taska/logs/%x_%j.out
#SBATCH --error=taska/logs/%x_%j.err
# Experiment 1: Petzka relative flatness along the grokking trajectory.
# No new training -- reads existing taska/checkpoints/*.
set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p taska/logs
source venv/bin/activate
echo "[start] $(date) host=$(hostname)"
python -u taska/analysis/petzka_trajectory.py
echo "[done] $(date)"
