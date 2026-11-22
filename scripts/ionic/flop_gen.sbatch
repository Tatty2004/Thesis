#!/bin/bash
# Flop-end labels for the flop-net test (bombpot/experiments/valuenet/flop_gen.py), one shard per
# array task, seed = SEED0 + task id. Submit from the worktree root, e.g.
#   ARM=a SEED0=0 GEN="--side train --deals 50 --iters 100 --per-pair 8 --minutes 50 --first-seconds 30" \
#       sbatch --array=0-79%12 --time=01:00:00 scripts/ionic/flop_gen.sbatch
# Writes node-local, then copies the finished shard to $OUT/$ARM/shard_<seed>.h5 (also after
# SIGUSR1/SIGTERM: --signal makes flop_gen stop early and keep its finished states).
# Jobs <= 1 h can land on RTX 2080 / 3090 nodes; longer ones only on the larger GPUs
# (node007's RTX 6000s are Turing-slow: excluded for those via EXCLUDE_MORE=node007).
# Memory: main ~8 GB at --deals all (pilot MaxRSS 7.6 GB) + ~0.5 GB per equity worker.
#SBATCH --job-name=vn_fgen
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --exclude=node806,node807,node808,node206,node805,node011,node012,node013,node014,node016,node021
#SBATCH --output=logs/flopnet_test/%x-%A_%a.out

PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
MODEL=${MODEL:-/n/fs/ugrad/ug27/rb5709/Thesis/results/valuenet/net/sweep_multi_d192L4/best.pt}
OUT=${OUT:-/n/fs/scratch/rb5709/valuenet/flopnet_test/data}
SEED=$(( ${SEED0:-0} + ${SLURM_ARRAY_TASK_ID:-0} ))
mkdir -p "$OUT/$ARM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
LOCAL="${TMPDIR:-/tmp}/$USER-fgen-$SLURM_JOB_ID-$SEED.h5"
$PY -m bombpot.experiments.valuenet.flop_gen --model "$MODEL" --seed "$SEED" --workers $(( SLURM_CPUS_PER_TASK - 1 )) \
    $GEN --out "$LOCAL" &
PID=$!
trap 'kill -USR1 $PID 2>/dev/null' USR1
trap 'kill -TERM $PID 2>/dev/null' TERM
rc=0
wait $PID || rc=$?
while kill -0 $PID 2>/dev/null; do wait $PID || rc=$?; done  # a trapped signal interrupts the first wait
if [[ -s $LOCAL ]]; then
    cp "$LOCAL" "$OUT/$ARM/shard_$(printf %05d "$SEED").h5" && rm -f "$LOCAL"
fi
rm -rf "$NUMBA_CACHE_DIR"
exit $rc
