#!/bin/bash
# Value-net training on one GPU (bombpot/experiments/valuenet/train.py). Arguments go to
# train.py, e.g.
#   sbatch scripts/ionic/valuenet_train.sbatch --config bombpot/experiments/configs/valuenet/small_deck.yaml
#   sbatch --time=00:15:00 scripts/ionic/valuenet_train.sbatch \
#       --config bombpot/experiments/configs/valuenet/synthetic_smoke.yaml --synthetic --data @LOCAL@/data
#       (GPU path check, synthetic shards written to node-local disk)
# @LOCAL@ in the arguments becomes a node-local directory for this job, removed at exit (for
# data that shouldn't sit on shared storage, e.g. shards staged with STAGE=<shard dir>, which
# copies the *.h5 files of that directory to @LOCAL@/data first).
# Uses the torch environment on scratch (PY=... overrides it).
#SBATCH --job-name=valuenet_train
#SBATCH --partition=all
#SBATCH --gres=gpu:1
# node806-808 are GTX 1080 / 1080 Ti (sm_61); the cu132 torch build only has sm_75+ kernels
#SBATCH --exclude=node806,node807,node808
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/valuenet/%x-%j.out

export PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"  # this checkout, not the editable install's
LOCAL="${TMPDIR:-/tmp}/$USER-valuenet-$SLURM_JOB_ID"
mkdir -p "$LOCAL"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$LOCAL"' EXIT
if [[ -n ${STAGE:-} ]]; then
    mkdir -p "$LOCAL/data" && cp "$STAGE"/*.h5 "$LOCAL/data/" && echo "staged $(ls "$LOCAL/data" | wc -l) shards from $STAGE"
fi
ARGS=("${@//@LOCAL@/$LOCAL}")
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
$PY -c "import torch, h5py; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.version.cuda, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu', '| h5py', h5py.__version__)"
# Fail fast (instead of mid-training) if this GPU's architecture isn't in the torch build.
$PY -c "
import sys, torch
if torch.cuda.is_available():
    cc = 'sm_%d%d' % torch.cuda.get_device_capability(0)
    if cc not in torch.cuda.get_arch_list():
        sys.exit(f'{torch.cuda.get_device_name(0)} is {cc}; torch {torch.__version__} has kernels for {torch.cuda.get_arch_list()} only')
" || exit 1

$PY -m bombpot.experiments.valuenet.train "${ARGS[@]}"
