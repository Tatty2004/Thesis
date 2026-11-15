"""Evaluate saved value networks on one common shard directory (every sample, no split), so runs
trained on different data sizes compare like for like: train.evaluate's held-out loss, zero
baseline and range-weighted per-hand error (pots).

    python -m bombpot.experiments.valuenet.eval_common --data results/valuenet/data/curve_common_heldout \
        results/valuenet/net/small_limit_hand_curve_*/best.pt

Needs torch (the scratch environment).
"""
from __future__ import annotations

import argparse
import json

from bombpot.experiments.valuenet.train import evaluate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+", help="saved ValueNets (best.pt)")
    ap.add_argument("--data", required=True, help="the common shard directory")
    ap.add_argument("--target", default="cfv")
    ap.add_argument("--weight", default="range")
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out", help="write the results here as JSON")
    a = ap.parse_args(argv)
    from bombpot.valuenet.data import ShardSet
    from bombpot.valuenet.model import load_model

    out = {}
    for path in a.models:
        model = load_model(path, "cpu")
        shards = ShardSet(a.data, model.features, 0.0, 1024, target=a.target, weight=a.weight, split="sample")
        ev = evaluate(model, shards, "all", "cpu", a.huber_delta, a.batch_size)
        out[path] = {**ev, "samples": shards.count("train")}
        print(f"{path}: loss {ev['loss']:.6g} (zero {ev['zero_loss']:.6g}) hand MAE {ev['hand_mae']:.6g} pots",
              flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
