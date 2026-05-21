#!/bin/bash
#SBATCH --job-name=petzka_phase
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=taska/logs/%x_%j.out
#SBATCH --error=taska/logs/%x_%j.err
# Experiment 2: Petzka over the 24-cell (WD x train-fraction) phase diagram.
# Retrains the grid (phase_diagram.py saved no weights) and saves checkpoints.
set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p taska/logs
source venv/bin/activate
echo "[start] $(date) host=$(hostname)"
python -u taska/analysis/petzka_phase_diagram.py
echo "[done] $(date)"
