#!/bin/bash
# The torch river-subgame solver against the compiled one, on one GPU
# (bombpot/experiments/valuenet/bench_gpu_subgame.py). Arguments go to the script, e.g.
#   sbatch scripts/ionic/valuenet_gpu_bench.sbatch --n 6 --tol 1e-2 --out results/valuenet/datagen/gpu_bench.json
#SBATCH --job-name=vn_gpu_bench
#SBATCH --partition=all
#SBATCH --gres=gpu:1
# node806-808 are GTX 1080 / 1080 Ti (sm_61); the cu132 torch build only has sm_75+ kernels
#SBATCH --exclude=node806,node807,node808
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/valuenet/%x-%j.out

export PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY -m bombpot.experiments.valuenet.bench_gpu_subgame "$@"
