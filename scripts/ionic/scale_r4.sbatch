#!/bin/bash
# S0 scaling test: 25 R4 iterations at the job's CPU count. Set the CPUs (and node) when submitting:
#   sbatch -c 32 scripts/ionic/scale_r4.sbatch
#   sbatch -c 128 --exclusive --mem=0 -w node403 scripts/ionic/scale_r4.sbatch
# THREADS=64 overrides NUMBA_NUM_THREADS (default: the CPU count), e.g. to leave hyperthreads idle.
#SBATCH --job-name=scale_r4
#SBATCH --partition=all
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/scaling/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
if [[ -n "${THREADS:-}" ]]; then
    export NUMBA_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
    echo "threads overridden: $THREADS"
fi

$PY scripts/ionic/scale_r4.py --iterations 25 --out "logs/scaling/$SLURM_JOB_ID.json"
