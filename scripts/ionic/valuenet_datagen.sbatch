#!/bin/bash
# Value-net training data (bombpot/experiments/valuenet/datagen.py): one HDF5 shard per array
# task, seed = task id. Submit from the repo root, e.g.
#   sbatch --array=0-39 scripts/ionic/valuenet_datagen.sbatch            # N=200 per shard
#   N=50 TOL=1e-3 sbatch --array=0-3 scripts/ionic/valuenet_datagen.sbatch
# Memory from a local measurement on the real deck (5 samples, 2026-11-09): peak RSS 1.8 GB;
# 4G is about 2x that. Time: N x seconds per sample, measured on a loaded login host (load ~110
# on 80 cores, the process got 4.3 of its 16 threads) at tol 1e-3 pots, limit river: 50 s with
# DCFR (the default), 146 s with CFR+. N=200 with DCFR: about 3 h there; 12 h leaves room.
#SBATCH --job-name=vn_datagen
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
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
