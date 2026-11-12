#!/bin/bash
# Gate C1 on the small deck (bombpot/experiments/valuenet/gate_c1.py): one configuration per job.
# Arguments go to gate_c1; give --out. From the worktree root, with PY set (no .venv here):
#   PY=/n/fs/ugrad/ug27/rb5709/Thesis/.venv/bin/python sbatch scripts/ionic/gate_c1.sbatch \
#       --trunk dcfr --iters 200 --checks 10,20,50,100,200 --leaf-tol 1e-3 --resolve-tol 1e-3 --inner dcfr --out <json>
# Memory: 3 iterations and one check on the limit game peak at 4.9 GB resident (measured 2026-11-12).
#SBATCH --job-name=gate_c1
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR"

$PY -m bombpot.experiments.valuenet.gate_c1 "$@"
