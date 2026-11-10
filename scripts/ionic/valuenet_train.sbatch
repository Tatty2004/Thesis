#!/bin/bash
# Value-net training on one GPU (bombpot/experiments/valuenet/train.py). Arguments go to
# train.py, e.g.
#   sbatch scripts/ionic/valuenet_train.sbatch --config bombpot/experiments/configs/valuenet/small_deck.yaml
#   sbatch --time=00:15:00 scripts/ionic/valuenet_train.sbatch \
#       --config bombpot/experiments/configs/valuenet/synthetic_smoke.yaml --synthetic   (GPU path check)
# Uses the torch environment on scratch (PY=... overrides it).
#SBATCH --job-name=valuenet_train
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/valuenet/%x-%j.out

export PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"  # this checkout, not the editable install's
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
$PY -c "import torch, h5py; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.version.cuda, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu', '| h5py', h5py.__version__)"

$PY -m bombpot.experiments.valuenet.train "$@"
