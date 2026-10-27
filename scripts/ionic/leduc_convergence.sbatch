#!/bin/bash
# leduc_convergence.py (the failing Leduc LP test's setup, run to 3000 iterations), in .venv
# and in the laptop-matched Python 3.11 env.
#   sbatch scripts/ionic/leduc_convergence.sbatch
#SBATCH --job-name=leduc_convergence
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
PY311=${PY311:-$HOME/envs/py311/bin/python}

for py in "$PY" "$PY311"; do
    echo "====="
    $py scripts/ionic/leduc_convergence.py --checkpoints 1500 2000 3000
done
