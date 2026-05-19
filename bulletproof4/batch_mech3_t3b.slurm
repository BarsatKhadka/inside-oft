#!/bin/bash
#SBATCH --job-name=mech3_t3b
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=bulletproof4/logs/%x_%j.out
#SBATCH --error=bulletproof4/logs/%x_%j.err
set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p bulletproof4/logs
source venv/bin/activate
echo "[start] $(date) host=$(hostname)"
python -u bulletproof4/mech3_mode_connectivity.py --tier tier3b --epochs 120
echo "[done] $(date)"
