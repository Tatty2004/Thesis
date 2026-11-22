#!/bin/bash
# Flop-end sample cost pilot on one GPU: `python -m bombpot.experiments.valuenet.$MODULE $ARGS`
# (MODULE "flop_pilot run" by default: one state's reference, variants and analysis; or flop_datagen).
# Submit from the worktree root, the GPU type in --gres, e.g.
#   ARGS="--flops Ah7c2d,KsKd9h --equity ... --seed 0 --out ..." \
#       sbatch --gres=gpu:a5000:1 --time=03:00:00 scripts/ionic/flop_pilot.sbatch
# MODEL (default the sweep's d192L4 HandSetNet) is passed as --model.
# RTX 2080 nodes take only jobs <= 1 h. Excluded nodes: see the shared-host notes (slow, full /tmp,
# GPUs busy, GTX 1080).
#SBATCH --job-name=vn_flop
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --exclude=node806,node807,node808,node206,node805,node011,node012,node013,node014,node016,node021
#SBATCH --output=logs/%x-%j.out

PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY -m bombpot.experiments.valuenet.${MODULE:-flop_pilot run} --model "${MODEL:-/n/fs/ugrad/ug27/rb5709/Thesis/results/valuenet/net/sweep_multi_d192L4/best.pt}" $ARGS
