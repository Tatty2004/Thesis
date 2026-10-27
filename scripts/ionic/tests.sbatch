#!/bin/bash
# The full test suite on a compute node. Extra arguments go to pytest, e.g.
#   sbatch scripts/ionic/tests.sbatch --runslow
#   sbatch scripts/ionic/tests.sbatch tests/holdem
#SBATCH --job-name=tests
#SBATCH --partition=all
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"

$PY -m pytest -ra --durations=15 "$@"
