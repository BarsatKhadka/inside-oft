#!/bin/bash
# Submit all bulletproof4 mechanistic experiments.
# Usage: bash bulletproof4/submit_all.sh

mkdir -p bulletproof4/logs

# Free analysis first (no compute, just reads existing JSONs)
echo "=== Free analyses (run these first; they take seconds) ==="
sbatch bulletproof4/batch_mech1_corr.slurm
sbatch bulletproof4/batch_mech2_perlayer.slurm

# Mode connectivity per tier — the critical experiment
echo "=== Mode connectivity ==="
sbatch bulletproof4/batch_mech3_t2.slurm
sbatch bulletproof4/batch_mech3_t3b.slurm
sbatch bulletproof4/batch_mech3_t4.slurm

# Heavy ablations
echo "=== Mechanistic ablations ==="
sbatch bulletproof4/batch_mech4_abl.slurm
sbatch bulletproof4/batch_mech5_noise.slurm
sbatch bulletproof4/batch_mech6_force.slurm

echo "Submitted. Watch: squeue -u \$USER"
