#!/bin/bash
#SBATCH --job-name=petzka_wd_sweep
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=taska/logs/%x_%j.out
#SBATCH --error=taska/logs/%x_%j.err
# Experiment 3: algorithmic WD sweep with Petzka -- WD in {0.1,0.25,0.5,1,2},
# 3 seeds, 30k epochs from scratch on modular addition.
set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p taska/logs
source venv/bin/activate
echo "[start] $(date) host=$(hostname)"
python -u taska/analysis/petzka_wd_sweep.py
echo "[done] $(date)"
