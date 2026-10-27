#!/bin/bash
# One experiment config through the runner. Arguments go to bombpot.experiments.run, e.g.
#   sbatch scripts/ionic/experiment.sbatch bombpot/experiments/configs/holdem/flop_turn.yaml
# Override resources on the command line: sbatch --mem=32G --time=4:00:00 scripts/ionic/experiment.sbatch ...
#SBATCH --job-name=experiment
#SBATCH --partition=all
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"

# Peak memory afterwards: jobstats <jobid>, or sacct -j <jobid> --format=MaxRSS,Elapsed
$PY -m bombpot.experiments.run "$@"
