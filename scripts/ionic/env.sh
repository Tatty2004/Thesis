# Shared setup for Ionic job scripts. Sourced by the .sbatch scripts in this folder;
# submit them from the repo root (sbatch scripts/ionic/<name>.sbatch ...).
# PY=<python> picks another environment (default .venv/bin/python), e.g.
#   PY=$HOME/envs/py311/bin/python sbatch scripts/ionic/tests.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
PY=${PY:-.venv/bin/python}
if [[ ! -f pyproject.toml || ! -x $PY ]]; then
    echo "submit from the repo root, with $PY set up" >&2
    exit 1
fi

# One thread per allocated CPU, for numba and for numpy's BLAS. An --exclusive job that
# didn't set --cpus-per-task gets the whole node.
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE}
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# numba's on-disk cache goes to node-local disk, one per job, removed at exit.
# Nodes differ in CPU, so a shared cache would just be recompiled and overwritten.
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/$USER-numba-$SLURM_JOB_ID"
mkdir -p "$NUMBA_CACHE_DIR"
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT

echo "job $SLURM_JOB_ID on $(hostname), $(date '+%Y-%m-%d %H:%M:%S')"
echo "cpus $SLURM_CPUS_PER_TASK (affinity $(taskset -cp $$ | sed 's/.*: //')), mem ${SLURM_MEM_PER_NODE:-?} MB"
lscpu | grep -E "Model name|Thread\(s\) per core"
echo "git $(git rev-parse --short HEAD)$(git status --porcelain -- bombpot pyproject.toml | grep -q . && echo ' dirty')"
$PY -c "import sys, numpy, scipy, numba; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| numba', numba.__version__)"
echo
