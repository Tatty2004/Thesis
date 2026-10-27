#!/bin/bash
# S0 scaling test on one whole node: each argument CPUS[:THREADS] runs scale_r4.py as its own
# job step, confined to CPUS CPUs (whole cores, hyperthreads included) with THREADS numba threads
# (default CPUS). Steps run one after another, so they don't compete with each other.
#   sbatch -w node202 scripts/ionic/scale_node.sbatch 8 16 32 64 104 104:52
#SBATCH --job-name=scale_node
#SBATCH --partition=all
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/scaling/%x-%j.out

source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"

for spec in "$@"; do
    cpus=${spec%%:*}
    threads=${spec##*:}
    echo "== step: $cpus CPUs, $threads threads"
    srun --exact -n 1 -c "$cpus" --mem=24G \
        env NUMBA_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" \
        $PY scripts/ionic/scale_r4.py --iterations 25 --out "logs/scaling/$SLURM_JOB_ID-c$cpus-t$threads.json"
done
