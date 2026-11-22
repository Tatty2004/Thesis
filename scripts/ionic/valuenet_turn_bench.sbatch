#!/bin/bash
# Time safe turn re-solves on one GPU (bombpot/experiments/valuenet/bench_turn_resolve.py).
#   sbatch --gres=gpu:rtx_2080:1 scripts/ionic/valuenet_turn_bench.sbatch <trunk dir> <out.json> [more args]
# The network runs on the GPU, the turn tree (one deal, one line) on the CPU.
#SBATCH --job-name=vn_turn_bench
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=node806,node807,node808,node206,node805,node011,node012,node013,node014,node016,node021
#SBATCH --output=logs/valuenet/%x-%j.out

export PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TRUNK=$1
OUT=$2
shift 2
$PY -m bombpot.experiments.valuenet.bench_turn_resolve --trunk "$TRUNK" --out "$OUT" "$@"
