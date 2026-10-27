#!/bin/bash
# M1: one solve (bombpot/experiments/m1.py solve). Arguments go to m1.py, e.g.
#   sbatch scripts/ionic/m1_solve.sbatch --config bombpot/experiments/configs/holdem/m1_kmeans2d.yaml
#   sbatch --time=3-00:00:00 --mem=<2 x the pilot's peak> scripts/ionic/m1_solve.sbatch --config ... --resume
# The time budget (run.budget_hours, or --budget-hours) must end at least 30 minutes before the
# walltime; m1.py checks. CPUs (Step 0): 64, with THREADS numba threads, by default half the CPUs
# (one per physical core); THREADS=64 uses the hyperthreads too.
#SBATCH --job-name=m1_solve
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/m1/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
THREADS=${THREADS:-$((SLURM_CPUS_PER_TASK / 2))}
export NUMBA_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
echo "threads $THREADS"

$PY -m bombpot.experiments.m1 solve "$@"
