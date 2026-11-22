#!/bin/bash
# A short real-deck trunk with --turn-cfv on node-local /tmp (nothing written to scratch), then
# the turn re-solve bench on it with both T-value sources; only the bench JSONs and the trunk's
# trunk.json are copied back to logs/valuenet/.
#   sbatch scripts/ionic/valuenet_turn_dev.sbatch [iters]
#SBATCH --job-name=vn_turn_dev
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --exclude=node806,node807,node808,node206,node805,node011,node012,node013,node014,node016,node021
#SBATCH --output=logs/valuenet/%x-%j.out

export PY=${PY:-/n/fs/scratch/rb5709/envs/torch/bin/python}
source "$SLURM_SUBMIT_DIR/scripts/ionic/env.sh"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
ITERS=${1:-20}
REAL=/n/fs/ugrad/ug27/rb5709/Thesis/results/valuenet/real
NET=/n/fs/ugrad/ug27/rb5709/Thesis/results/valuenet/net/net2_attn_op1/best.pt
WORK="${TMPDIR:-/tmp}/$USER-turn-dev-$SLURM_JOB_ID"
mkdir -p "$WORK"
trap 'rm -rf "$WORK" "$NUMBA_CACHE_DIR"' EXIT
df -h "$WORK" | tail -1
cp "$REAL/equity_Ah7c2d_KsKd9h_13x4.npy" "$WORK/"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -m bombpot.experiments.valuenet.real_trunk --model "$NET" --flops Ah7c2d,KsKd9h --iters "$ITERS" \
    --checks "$ITERS" --samples 0 --defer --amp --turn-cfv --no-river-inputs --out "$WORK/it$ITERS"
ls -la "$WORK/it$ITERS"
cp "$WORK/it$ITERS/trunk.json" "logs/valuenet/turn_dev_it${ITERS}_trunk.json"
for SRC in avg br; do
    CUDA_VISIBLE_DEVICES=0 $PY -m bombpot.experiments.valuenet.bench_turn_resolve --trunk "$WORK/it$ITERS" \
        --cfv $SRC --spots 4 --modes alt --history 300 --out "logs/valuenet/turn_dev_it${ITERS}_bench_$SRC.json"
done
