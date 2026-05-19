#!/bin/bash
#SBATCH --job-name=mech9_clustering
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=bulletproof4/logs/%x_%j.out
#SBATCH --error=bulletproof4/logs/%x_%j.err
set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p bulletproof4/logs
source venv/bin/activate
python -u bulletproof4/mech9_tier_clustering.py
