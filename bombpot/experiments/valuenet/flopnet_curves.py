"""Read-out of the flop-net label test (PLAN_flopnet_test.md): per-arm label cost from the shards,
learning curves from the training runs, a power-law fit, and the matched-budget comparison.

    python -m bombpot.experiments.valuenet.flopnet_curves --data /n/fs/scratch/rb5709/valuenet/flopnet_test/data \
        --runs flopnet_ --out logs/flopnet_test/curves.json

Cost: each state's `seconds` (the solve plus the control variate) on its shard's GPU, converted
to A5000-seconds by the measured speed of that GPU type on the same arm (median seconds per
label relative to the A5000's); the arm's A5000-s per label is the mean of those. Runs are
results/valuenet/net/<prefix><arm>_n<N>_s<seed>[...]: their summary.json `best` row (best epoch
by the arm's own validation loss) gives eval_hand_mae on the exact held-out set.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def arm_costs(data: Path) -> dict:
    import h5py

    out = {}
    for d in sorted(p for p in data.iterdir() if p.is_dir()):
        per_gpu: dict = {}
        n = 0
        for f in sorted(d.glob("*.h5")):
            with h5py.File(f, "r") as h:
                if h.attrs.get("n", 0) == 0 or "seconds" not in h:
                    continue
                per_gpu.setdefault(str(h.attrs["gpu"]), []).extend(h["seconds"][:].tolist())
                n += int(h.attrs["n"])
        if not n:
            continue
        med = {g: float(np.median(s)) for g, s in per_gpu.items()}
        ref = med.get("NVIDIA RTX A5000") or min(med.values())
        a5000_s = sum(sum(s) * ref / med[g] for g, s in per_gpu.items())
        out[d.name] = {"labels": n, "a5000_seconds_per_label": a5000_s / n, "a5000_hours": a5000_s / 3600,
                       "gpu_seconds_per_label_median": med,
                       "labels_by_gpu": {g: len(s) for g, s in per_gpu.items()}}
    return out


def runs(prefix: str) -> list:
    from bombpot.experiments.run import RESULTS

    rows = []
    for d in sorted((RESULTS / "valuenet" / "net").glob(prefix + "*")):
        m = re.match(re.escape(prefix) + r"(.+)_n(\d+)_s(\d+)$", d.name)
        s = d / "summary.json"
        if not m or not s.exists():
            continue
        js = json.loads(s.read_text())
        b = js["best"]
        rows.append({"run": d.name, "arm": m.group(1), "n": int(m.group(2)), "seed": int(m.group(3)),
                     "train": js["train_samples"], "val": js["heldout_samples"], "epochs": js["epochs"],
                     "best_epoch": b["epoch"], "eval_mae": b.get("eval_hand_mae"),
                     "eval_mae_last": js["last"].get("eval_hand_mae"), "val_mae": b["heldout_hand_mae"],
                     "minutes": js["seconds"] / 60, "gpu": js.get("gpu")})
    return rows


def fit(ns, maes) -> dict | None:
    """log MAE = log c + slope log N, least squares (needs 3 points)."""
    if len(ns) < 3:
        return None
    x, y = np.log(ns), np.log(maes)
    slope, ic = np.polyfit(x, y, 1)
    return {"slope": float(slope), "c": float(np.exp(ic)),
            "at": {str(n): float(np.exp(ic) * n ** slope) for n in (10_000, 30_000, 100_000, 300_000)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="flopnet_")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    costs = arm_costs(Path(a.data))
    rs = runs(a.runs)
    curves = {}
    for r in rs:
        curves.setdefault(r["arm"], {}).setdefault(r["n"], []).append(r["eval_mae"])
    table = {}
    for arm, by_n in curves.items():
        base = arm.split("_")[0]  # e.g. "b_k50": states of arm b, labels k50
        c = costs.get(base, {}).get("a5000_seconds_per_label")
        pts = [{"n": n, "eval_mae": float(np.mean(v)), "seeds": len(v), "spread": float(np.ptp(v)) if len(v) > 1 else 0.0,
                "label_a5000_hours": None if c is None or "_" in arm else n * c / 3600}
               for n, v in sorted(by_n.items())]
        table[arm] = {"points": pts, "fit": fit([p["n"] for p in pts], [p["eval_mae"] for p in pts])}
    res = {"costs": costs, "curves": table, "runs": rs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for arm, t in sorted(table.items()):
        print(arm, " ".join(f"N={p['n']}:{p['eval_mae']:.4f}" + (f"({p['label_a5000_hours']:.0f}h)"
                                                                  if p["label_a5000_hours"] else "")
                            for p in t["points"]), "| fit", t["fit"] and {k: round(v, 4) if isinstance(v, float)
                                                                         else v for k, v in t["fit"].items()})
    print(json.dumps(costs, indent=1))


if __name__ == "__main__":
    main()
