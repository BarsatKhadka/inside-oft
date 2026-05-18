#!/bin/bash
#SBATCH --job-name=quant_batch
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Quantitative experiments for the rigor pass.
# These produce numerical claims with error bars across 3 seeds each:
#   1. wd_rank_quantitative.py     — quantitative WD→rank→escape relationship (3 seeds × 11 WD = 33 runs × 30k epochs)
#   2. per_layer_rank.py            — which layer is the memorization bottleneck?
#   3. task_complexity_rank.py      — does converged rank depend on task complexity?
#
# Total compute: heavy. ~12-20 hours on L40S.

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
echo "1. per_layer_rank.py  (which layer is the bottleneck?)"
echo "============================================================"
python -u taska/analysis/per_layer_rank.py

echo
echo "============================================================"
echo "2. task_complexity_rank.py  (does rank scale with task?)"
echo "============================================================"
python -u taska/analysis/task_complexity_rank.py

echo
echo "============================================================"
echo "3. wd_rank_quantitative.py  (quantitative WD-rank-escape; LONG)"
echo "============================================================"
python -u taska/analysis/wd_rank_quantitative.py

echo "[done] $(date)"
