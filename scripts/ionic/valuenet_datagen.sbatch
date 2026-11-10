#!/bin/bash
# Value-net training data (bombpot/experiments/valuenet/datagen.py): one HDF5 shard per array
# task, seed = task id. Submit from the repo root, e.g.
#   sbatch --array=0-39 scripts/ionic/valuenet_datagen.sbatch            # N=200 per shard
#   N=50 TOL=1e-3 sbatch --array=0-3 scripts/ionic/valuenet_datagen.sbatch
# Memory from a local measurement on the real deck (see the subgame stream's report): peak RSS
# about 2 GB per process; 6G leaves room. Time: --time must cover N x seconds per sample.
#SBATCH --job-name=vn_datagen
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=logs/valuenet/%x-%A_%a.out

PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
N=${N:-200}
TOL=${TOL:-1e-3}
OUT=${OUT:-results/valuenet/data}
mkdir -p "$OUT"
TASK=${SLURM_ARRAY_TASK_ID:-0}
# Write to node-local disk and copy once at the end (one large write to scratch).
LOCAL="${TMPDIR:-/tmp}/$USER-vn-$SLURM_JOB_ID-$TASK.h5"
$PY -m bombpot.experiments.valuenet.datagen --n "$N" --seed "$TASK" --tol "$TOL" --out "$LOCAL" --log-every 5
cp "$LOCAL" "$OUT/shard_$(printf %04d "$TASK").h5" && rm -f "$LOCAL"
