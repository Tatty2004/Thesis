#!/bin/bash
# Time the viewer's live mode (bombpot/experiments/valuenet/bench_live.py) on one GPU, or on CPUs
# with --gres=gpu:0 and DEVICE=cpu. Caches go to node-local /tmp (removed at exit).
#   sbatch scripts/ionic/valuenet_live_bench.sbatch <trunk dir> <out.json> [more args]
#   sbatch --gres=gpu:0 --cpus-per-task=8 --export=ALL,DEVICE=cpu scripts/ionic/valuenet_live_bench.sbatch ...
#SBATCH --job-name=vn_live_bench
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=node806,node807,node808,node206,node805,node011,node012,node013,node014,node016,node021
#SBATCH --output=logs/valuenet/%x-%j.out

export PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
DEVICE=${DEVICE:-cuda}
[[ $DEVICE == cuda ]] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CACHE="${TMPDIR:-/tmp}/$USER-live-bench-$SLURM_JOB_ID"
mkdir -p "$CACHE"
trap 'rm -rf "$CACHE" "$NUMBA_CACHE_DIR"' EXIT
TRUNK=$1
OUT=$2
shift 2
$PY -m bombpot.experiments.valuenet.bench_live --trunk "$TRUNK" --out "$OUT" --cache "$CACHE" --device "$DEVICE" "$@"
