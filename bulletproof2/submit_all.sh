#!/bin/bash
# Submit all bp7-bp22 jobs. Run from project root.
# Usage: bash bulletproof2/submit_all.sh

mkdir -p bulletproof2/logs

for slurm in bulletproof2/batch_bp*.slurm; do
    echo "Submitting $slurm"
    sbatch "$slurm"
done
