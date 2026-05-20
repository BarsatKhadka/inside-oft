#!/bin/bash
#SBATCH --job-name=tier6_v5_validate_g
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
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
python -u bulletproof3/tier6_v5_validate_g.py
echo "[done] $(date)"
