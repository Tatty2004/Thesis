#!/bin/bash
# M1: the exact check of one checkpoint's average strategy (bombpot/experiments/m1.py evaluate), e.g.
#   sbatch scripts/ionic/m1_eval.sbatch --checkpoint results/m1/m1_kmeans2d/solve/ckpt_t000001234.npz
#   sbatch --time=00:30:00 scripts/ionic/m1_eval.sbatch --checkpoint ... --max-minutes 20   (timing only)
# It writes ckpt_t<t>.eval.json next to the checkpoint (or .timing.json when stopped early).
# CPUs as for m1_solve.sbatch: 64, with THREADS (default half the CPUs) numba threads.
# Split into N shards, as an array job whose task i runs shard i, then m1_combine.sbatch:
#   sbatch --array=0-7 scripts/ionic/m1_eval.sbatch --checkpoint ... --shards 8
#SBATCH --job-name=m1_eval
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/m1/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
THREADS=${THREADS:-$((SLURM_CPUS_PER_TASK / 2))}
export NUMBA_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
echo "threads $THREADS"

$PY -m bombpot.experiments.m1 evaluate "$@" ${SLURM_ARRAY_TASK_ID:+--shard $SLURM_ARRAY_TASK_ID}
