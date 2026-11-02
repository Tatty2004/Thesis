#!/bin/bash
# M1: a split exact check's result (bombpot/experiments/m1.py combine), after its shards, e.g.
#   sbatch --dependency=afterok:<array job> scripts/ionic/m1_combine.sbatch --checkpoint ... --shards 8
# It loads the checkpoint, adds up the shards' river sums and writes ckpt_t<t>.eval.json.
#SBATCH --job-name=m1_combine
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/m1/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"

$PY -m bombpot.experiments.m1 combine "$@"
