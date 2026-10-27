#!/bin/bash
# M1: fit the river buckets of one config, then check them (bombpot/experiments/m1.py fit), e.g.
#   sbatch scripts/ionic/m1_fit.sbatch --config bombpot/experiments/configs/holdem/m1_kmeans2d.yaml
# The fit is sequential (one random generator runs through the turn deals in order) and mostly
# single-threaded numpy, so it asks for 4 CPUs, not Step 0's 64. Measured on spin: kmeans_2d
# 2.3 s per turn deal (1.3 h for 2,070), avg_1d 0.25 s (10 min).
#SBATCH --job-name=m1_fit
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/m1/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"

$PY -m bombpot.experiments.m1 fit "$@"
