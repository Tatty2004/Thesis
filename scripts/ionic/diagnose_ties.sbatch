#!/bin/bash
# The 4 tests that failed on Ionic (job 31993952), in the laptop-matched Python 3.11 env and in
# .venv (3.13), then diagnose_ties.py in both: where the tree and range solvers part ways.
#   sbatch scripts/ionic/diagnose_ties.sbatch
#SBATCH --job-name=diagnose_ties
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
PY311=${PY311:-$HOME/envs/py311/bin/python}

FAILED=(
    "tests/test_range_cfr.py::test_iterates_match_tree_solver[dcfr-holdem-4x2-1-b2]"
    "tests/test_range_cfr.py::test_iterates_match_tree_solver[cfr+-holdem-4x2-1-b2]"
    "tests/test_range_cfr.py::test_iterates_match_tree_solver[cfr-holdem-4x2-1-b2]"
    "tests/test_range_cfr.py::test_converges_to_the_lp_value[leduc]"
)

set +e  # a failing test shouldn't stop the remaining steps
for py in "$PY311" "$PY"; do
    echo "===== tests: $($py -c 'import sys, numpy; print("python", sys.version.split()[0], "numpy", numpy.__version__)')"
    $py -m pytest -p no:cacheprovider -rA "${FAILED[@]}"
done
for py in "$PY311" "$PY"; do
    echo "===== diagnose_ties"
    $py scripts/ionic/diagnose_ties.py
done
