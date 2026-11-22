#!/bin/bash
# One learning-curve point of the flop-net test (PLAN_flopnet_test.md): stage arm $ARM's shards
# and the exact eval shards to node-local disk, then train flopnet_test.yaml on the first $N
# samples with label set $TARGET (cfv, k50cfv, k400cfv) and seed $SEED. Run name
# flopnet_${NAME:-$ARM}_n${N}_s${SEED} under results/valuenet/net/. Submit from the worktree root:
#   ARM=a N=1000 SEED=0 sbatch --time=01:00:00 scripts/ionic/flop_train.sbatch
#SBATCH --job-name=vn_ftr
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=node806,node807,node808,node206,node805,node011,node012,node013,node014,node016,node021
#SBATCH --output=logs/flopnet_test/%x-%j.out

export PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
DATA=${DATA:-/n/fs/scratch/rb5709/valuenet/flopnet_test/data}
LOCAL="${TMPDIR:-/tmp}/$USER-ftr-$SLURM_JOB_ID"
mkdir -p "$LOCAL/train" "$LOCAL/eval"
trap 'rm -rf "$NUMBA_CACHE_DIR" "$LOCAL"' EXIT
cp "$DATA/$ARM"/*.h5 "$LOCAL/train/" && cp "$DATA/eval"/*.h5 "$LOCAL/eval/"
echo "staged $(ls "$LOCAL/train" | wc -l) $ARM shards, $(ls "$LOCAL/eval" | wc -l) eval shards"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY -m bombpot.experiments.valuenet.train --config bombpot/experiments/configs/valuenet/flopnet_test.yaml \
    --run "flopnet_${NAME:-$ARM}_n${N}_s${SEED}" --data "$LOCAL/train" --eval-dir "$LOCAL/eval" \
    --limit "$N" --epochs-for "$N" --target "${TARGET:-cfv}" --seed "$SEED"
