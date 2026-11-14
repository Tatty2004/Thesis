#!/bin/bash
# Gate F's noise study (bombpot/experiments/valuenet/gate_f_noise.py): the exact river leaf with
# a fixed random error, one level per job. Arguments go to gate_f_noise; give --noise and --out.
#   sbatch scripts/ionic/gate_f_noise.sbatch --noise 0.05 --mode rel --out results/valuenet/noise/rel_0.05.json
# Memory: gate C1 (the same solve, exact leaf) peaks at 4.9 GB on this game (2026-11-12).
#SBATCH --job-name=gf_noise
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=logs/valuenet/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR"

$PY -m bombpot.experiments.valuenet.gate_f_noise "$@"
