# 10 new jobs to submit

Submission instructions for the next batch of experiments. Run from `~/inside-oft` on HPC after `git pull origin main`.

## Setup (one-time)

```bash
cd ~/inside-oft
git pull origin main
source venv/bin/activate
pip install scipy   # needed for robust Lanczos + Hungarian matching in mech7
```

## Step 1 — free analyses (run first, finish in seconds)

These read existing tier JSONs and compute correlations / per-layer breakdowns. No training. Will give immediate insight on whether MIA is the universal anchor and where the M-vs-G signal lives per layer.

```bash
sbatch bulletproof4/batch_mech1_corr.slurm       # MIA vs structural signature correlations
sbatch bulletproof4/batch_mech2_perlayer.slurm   # per-layer rank decomposition
```

Check output:
```bash
tail -f bulletproof4/logs/mech1_corr_*.out
tail -f bulletproof4/logs/mech2_perlayer_*.out
```

## Step 2 — tier5 (CharLM) rerun, EXCLUDE bad node

The previous tier5 runs hit CUDA ECC hardware errors. Find which node was bad and exclude it:

```bash
# Find which node had ECC errors
grep -h "host=" bulletproof3/logs/tier5_charlm_*.out | sort -u
```

Replace `NODE_X` below with whatever node name you saw (e.g. `node041`, `node049`):

```bash
sed -i '/#SBATCH --time=/i #SBATCH --exclude=NODE_X' bulletproof3/batch_tier5_charlm.slurm
head bulletproof3/batch_tier5_charlm.slurm   # verify --exclude line is present
sbatch bulletproof3/batch_tier5_charlm.slurm
```

## Step 3 — tier6 (Pythia) rerun, float32 + NaN fix applied

The fix is already in the latest commit (float32 load + lr=1e-5 + gradient clipping + NaN guard).

```bash
sbatch bulletproof3/batch_tier6_pythia.slurm
tail -f bulletproof3/logs/tier6_pythia_*.out
```

Watch the first `ep=5` print. Expected: `train_loss ≈ 4.0`. If NaN again, kill the job — we need to debug further.

## Step 4 — mode connectivity per tier (the highest-value new experiment)

Tells us whether M and G are in the same basin (no barrier) or different basins (barrier). Critical for explaining why ViT signatures decouple — if same basin, they're just different points in one basin.

```bash
sbatch bulletproof4/batch_mech3_t2.slurm    # ResNet-18 CIFAR-10
sbatch bulletproof4/batch_mech3_t3b.slurm   # ViT-Tiny CIFAR-10
sbatch bulletproof4/batch_mech3_t4.slurm    # ViT-Small CIFAR-100
sbatch bulletproof4/batch_mech7_lmc.slurm   # permutation-aligned LMC for ResNet
```

## Step 5 — heavy mechanistic ablations

These explain WHY specific decouplings happen.

```bash
sbatch bulletproof4/batch_mech4_abl.slurm    # ResNet 2x2: WD vs aug → explains sharpness reversal
sbatch bulletproof4/batch_mech5_noise.slurm  # CIFAR with 30% random labels → does noise memorize differently?
sbatch bulletproof4/batch_mech6_force.slurm  # ViT-Tiny on 500-example CIFAR → does forced memorization make signatures fire?
```

## Total jobs queued

If you submit everything: **10 jobs**
- 2 free analyses (mech1, mech2)
- 1 tier5 rerun
- 1 tier6 rerun
- 4 mode-connectivity (mech3_t2, mech3_t3b, mech3_t4, mech7)
- 2 expensive ablations (mech4, mech5)
- 1 ViT forced grokking (mech6)

## Check what's queued

```bash
squeue -u $USER
```

## Pull results back when done

```bash
# From laptop
rsync -avz magnolia1:~/inside-oft/bulletproof3/results/ ./bulletproof3/results/
rsync -avz magnolia1:~/inside-oft/bulletproof4/results/ ./bulletproof4/results/
```

## What each experiment answers

| Job | Cost | Question it answers |
|---|---|---|
| mech1_corr | 0 GPU-hr | Q4: is MIA AUC just measuring the same property as other signatures? |
| mech2_perlayer | 0 GPU-hr | Q3, Q7: where (which layer) does the M-vs-G signal live? |
| tier5_charlm | ~25 GPU-hr | Final scale ladder point for autoregressive LM (from scratch) |
| tier6_pythia | ~35 GPU-hr | Final scale ladder point for pretrained LM fine-tuning |
| mech3_t2 | ~12 GPU-hr | Q5: are ResNet M and G in different basins? |
| mech3_t3b | ~18 GPU-hr | Q5: are ViT-Tiny M and G in different basins? |
| mech3_t4 | ~36 GPU-hr | Q5: are ViT-Small M and G in different basins? |
| mech7_lmc | ~12 GPU-hr | Naive LMC barrier for ResNet + permutation cost |
| mech4_abl | ~48 GPU-hr | Q1: why does sharpness REVERSE in CNN? (WD only / aug only / both) |
| mech5_noise | ~24 GPU-hr | Q6: does benign overfit differ from random-label memorization? |
| mech6_force | ~24 GPU-hr | Q2: does ViT show clean signatures under extreme memorization (500 ex)? |

Total compute: ~230 GPU-hours. Wall-clock with HPC parallelism: 3-5 days.

## Watch the most informative ones first

```bash
# mech1 finishes in <1 minute — might immediately confirm "MIA is the universal anchor"
tail -f bulletproof4/logs/mech1_corr_*.out

# mech3_t3b (ViT-Tiny mode connectivity) determines whether the ViT signature
# decoupling is "same basin different points" or "different basins different signatures"
tail -f bulletproof4/logs/mech3_t3b_*.out
```

## If anything errors

Paste the `.out` and `.err` tail back to Claude. Most likely issues:
- mech3_t4 (ViT-Small) might OOM during Hessian — if so, drop probe set from 1000 to 300
- mech4 is 12 runs total — might hit 48h wall-time, may need to split into 2 jobs
- tier6 NaN again — would need lr=5e-6 or bfloat16 native support
- mech7 needs scipy — make sure `pip install scipy` was done in the venv
