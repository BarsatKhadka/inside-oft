#!/bin/bash
#SBATCH --job-name=analysis_trackb
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run all Track B analyses on the existing M_CIFAR and G_CIFAR checkpoints.
# No training -- only analysis. Should be ~30-60 min on L40S.

set -uo pipefail

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate

echo "[start] $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv

echo
echo "============================================================"
echo "1. analysis_trackb.py  — margins, ranks, gradients, MIA"
echo "============================================================"
python -u trackb/analysis_trackb.py

echo
echo "============================================================"
echo "2. rank_trajectory_trackb.py  — rank over training epochs"
echo "============================================================"
python -u trackb/rank_trajectory_trackb.py

echo
echo "============================================================"
echo "3. mode_connectivity_trackb.py  — barrier between M and G"
echo "============================================================"
python -u trackb/mode_connectivity_trackb.py

echo
echo "[done] $(date)"
