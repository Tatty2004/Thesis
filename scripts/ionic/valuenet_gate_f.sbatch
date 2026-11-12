#!/bin/bash
# Gate F for one configuration: train the value network on the small deck's shards (CPU; the
# network and the 36-hand samples are small), then solve flop+turn with it and grade exactly
# (bombpot/experiments/valuenet/gate_f.py). Argument: a config name under
# bombpot/experiments/configs/valuenet/ (without .yaml); more arguments go to gate_f.
#   sbatch scripts/ionic/valuenet_gate_f.sbatch small_limit_j2d_br
# SKIP_TRAIN=1 grades an existing results/valuenet/net/<name>/best.pt.
# Memory: gate C1 peaks at 4.9 GB on this game (2026-11-12); training reads shards in blocks.
#SBATCH --job-name=gate_f
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=logs/valuenet/%x-%j.out

export PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
NAME=$1
shift
RUN=results/valuenet/net/$NAME
if [[ -z ${SKIP_TRAIN:-} ]]; then
    $PY -m bombpot.experiments.valuenet.train --config "bombpot/experiments/configs/valuenet/$NAME.yaml" --run "$NAME"
fi
$PY -m bombpot.experiments.valuenet.gate_f --model "$RUN/best.pt" --out "$RUN/gate_f.json" "$@"
