#!/bin/bash
# Submit all bulletproof3 jobs. Run from project root: bash bulletproof3/submit_all.sh
# Order: cheap fixes first, then scale ladder bottom-up.

mkdir -p bulletproof3/logs

# Fixes first (cheap, quick wins)
echo "=== Bug fixes ==="
sbatch bulletproof3/batch_fix_distill.slurm
sbatch bulletproof3/batch_fix_probe.slurm
sbatch bulletproof3/batch_fix_widthdepth.slurm

# Scale ladder
echo "=== Tier 0-1 (toy / small) ==="
sbatch bulletproof3/batch_tier0_4L.slurm
sbatch bulletproof3/batch_tier1_mnist.slurm
sbatch bulletproof3/batch_tier1b_fmnist.slurm

echo "=== Tier 2-3 (vision medium) ==="
sbatch bulletproof3/batch_tier2_r18.slurm
sbatch bulletproof3/batch_tier3_r50.slurm
sbatch bulletproof3/batch_tier3b_vit_t.slurm

echo "=== Tier 4-6 (large) ==="
sbatch bulletproof3/batch_tier4_vit_s.slurm
sbatch bulletproof3/batch_tier5_charlm.slurm
sbatch bulletproof3/batch_tier6_pythia.slurm

echo "Submitted all bulletproof3 jobs. Check squeue -u \$USER"
