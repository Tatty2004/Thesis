#!/bin/bash
# Value-net training data with the torch river solver on one GPU (datagen.py --solver gpu):
# one HDF5 shard per array task, seed = task id. Submit from the repo root, e.g.
#   N=700 TOL=1e-2 OUT=results/valuenet/data/real_onepair EXTRA="--flops Ah7c2d,KsKd9h" \
#       sbatch --array=0-39 scripts/ionic/valuenet_datagen_gpu.sbatch
# RTX 2080 Ti only (agreed 2026-11-15: up to 40 at once; they are the least contended, and the
# solver needs 4 GB of the card). Host memory measured 2.1 GB (job 32308115), about 3.7 s per sample at tol 1e-2.
#SBATCH --job-name=vn_dg_gpu
#SBATCH --partition=all
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
# Jobs over 1 h are not accepted on the RTX 2080 nodes (2026-11-15): keep N x 3.7 s under about 45 min.
#SBATCH --time=01:00:00
#SBATCH --output=logs/valuenet/%x-%A_%a.out

PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
N=${N:-700}
TOL=${TOL:-1e-2}
OUT=${OUT:-results/valuenet/data}
mkdir -p "$OUT"
TASK=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${OFFSET:-0} ))  # OFFSET: seeds past the 999 array-index cap
if [ -n "${SEEDS:-}" ]; then  # SEEDS="812 840 ...": rerun these seeds, task i -> the i-th
    L=($SEEDS); TASK=${L[${SLURM_ARRAY_TASK_ID:-0}]}
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Write to node-local disk and copy once at the end (one large write to scratch).
LOCAL="${TMPDIR:-/tmp}/$USER-vn-$SLURM_JOB_ID-$TASK.h5"
# Copy only a finished shard; a failed run (e.g. a full /tmp, node016 on 2026-11-20) leaves nothing behind.
if $PY -m bombpot.experiments.valuenet.datagen --n "$N" --seed "$TASK" --tol "$TOL" --solver gpu --dtype float32 \
    --out "$LOCAL" --log-every "${LOG:-25}" ${EXTRA:-}; then
    cp "$LOCAL" "$OUT/shard_$(printf %04d "$TASK").h5" && rm -f "$LOCAL"
else
    rc=$?; rm -f "$LOCAL"; exit $rc
fi
