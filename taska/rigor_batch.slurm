#!/bin/bash
#SBATCH --job-name=rigor_batch
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Four rigor experiments back-to-back. Cumulative ~3-5 hours on L40S.
# These nail down the mechanism: "WD escapes M's saddle via rank compression"
#   1. rank_during_rescue  — does WD compress rank, while SAM and noise don't?
#   2. rank_constraint_rescue  — does forcing low rank (without WD) escape?
#   3. wd_sweep  — quantitative WD-strength threshold
#   4. alternative_regularizers  — L1, spectral, label smoothing — which escape?

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
echo "1. rank_during_rescue.py  (~1 hour)"
echo "============================================================"
python -u taska/analysis/rank_during_rescue.py

echo
echo "============================================================"
echo "2. rank_constraint_rescue.py  (~1 hour)"
echo "============================================================"
python -u taska/analysis/rank_constraint_rescue.py

echo
echo "============================================================"
echo "3. wd_sweep.py  (~2 hours -- 9 runs, possibly long)"
echo "============================================================"
python -u taska/analysis/wd_sweep.py

echo
echo "============================================================"
echo "4. alternative_regularizers.py  (~1 hour)"
echo "============================================================"
python -u taska/analysis/alternative_regularizers.py

echo "[done] $(date)"
