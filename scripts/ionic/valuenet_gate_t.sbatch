#!/bin/bash
# Gate T: safe turn re-solving on the small deck against the exact three-street grade
# (bombpot/experiments/valuenet/gate_t.py). Arguments go to gate_t (--out is required).
#   sbatch scripts/ionic/valuenet_gate_t.sbatch --trunk-iters 100 --out logs/valuenet/gate_t.json
#SBATCH --job-name=vn_gate_t
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --exclude=node806,node807,node808,node206,node805,node011,node012,node013,node014,node016,node021
#SBATCH --output=logs/valuenet/%x-%j.out

export PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
$PY -m bombpot.experiments.valuenet.gate_t "$@"
