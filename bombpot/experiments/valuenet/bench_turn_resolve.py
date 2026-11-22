"""Time safe turn re-solves (turn_resolve.py) on one device from a real_trunk directory.

    python -m bombpot.experiments.valuenet.bench_turn_resolve --trunk results/valuenet/real/<name> \
        --spots 6 --iters 100,300 --history 300 --out logs/valuenet/bench_turn.json

For `--spots` random (turn deal, flop line) roots (plus the checked-down flop 'cc' first), for
each player: the spot's setup (the one-deal network leaf, the T values), then a fresh re-solve
of each length in --iters timed without intermediate checks (one evaluation at the end), and
one re-solve of --history iterations evaluated every --check-every iterations (convergence).
The first re-solve is a warm-up and is reported separately. Reports wall time, the share in
network calls, the gadget's exploitability, GPU memory and host RSS.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from bombpot.experiments.valuenet.turn_resolve import Spot, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--spots", type=int, default=6)
    ap.add_argument("--iters", default="100,300")
    ap.add_argument("--history", type=int, default=300)
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--cfv", default="auto", choices=["auto", "avg", "br"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch

    spot = Spot(a.trunk, a.device)
    lens = [int(x) for x in a.iters.split(",")]
    rng = np.random.default_rng(a.seed)
    L = spot.upper.classes[1][0].num_lines
    picks = [(int(rng.integers(spot.D)), 0)] + [(int(rng.integers(spot.D)), int(rng.integers(L))) for _ in range(a.spots - 1)]
    dev = torch.cuda.get_device_name(0) if a.device == "cuda" else a.device
    res = {"args": vars(a), "device": dev, "setup_seconds": spot.setup_seconds, "runs": [], "histories": []}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = spot.setup(picks[0][0], 0, picks[0][1], a.cfv)
    t0 = time.perf_counter()
    spot.resolve(s, 0, 20)
    res["warmup_seconds"] = time.perf_counter() - t0
    print(f"{dev}: setup {spot.setup_seconds:.1f}s, warm-up {res['warmup_seconds']:.1f}s", flush=True)
    for d, l in picks:
        t0 = time.perf_counter()
        s = spot.setup(d, 0, l, a.cfv)
        setup = time.perf_counter() - t0
        for p in (0, 1):
            if float(s["reach"][p].sum()) <= 0:
                continue
            for n in lens:
                if a.device == "cuda":
                    torch.cuda.synchronize()
                g, ev = spot.resolve(s, p, n)
                r = report(spot, s, g, ev, p)
                r.update(spot_setup_seconds=setup, tree_seconds=g.seconds - g.leaf_seconds)
                res["runs"].append(r)
                print(f"deal {d} line {r['flop_line']:9s} p{p} {n:4d} its: {g.seconds:6.2f}s (net {g.leaf_seconds:5.2f}s, "
                      f"{g.leaf_calls} calls), expl {r['exploitability_pots']:.2e} pots, setup {setup:.2f}s", flush=True)
            if a.history:
                g, ev = spot.resolve(s, p, a.history, tol=0.0, check_every=a.check_every)
                r = report(spot, s, g, ev, p)
                res["histories"].append(r)
                print(f"   history p{p}: " + ", ".join(f"{t}:{x / s['pot']:.2e}" for t, x in r["history"]), flush=True)
        res["gpu_max_allocated_GB"] = torch.cuda.max_memory_allocated() / 2 ** 30 if a.device == "cuda" else None
        res["host_max_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        out.write_text(json.dumps(res, indent=1, default=float))
    for n in lens:
        t = [r["seconds"] for r in res["runs"] if r["iterations"] == n]
        e = [r["exploitability_pots"] for r in res["runs"] if r["iterations"] == n]
        res.setdefault("summary", {})[n] = {"seconds_mean": float(np.mean(t)), "seconds_max": float(np.max(t)),
                                            "net_share": float(np.mean([r["leaf_seconds"] / r["seconds"] for r in res["runs"]
                                                                        if r["iterations"] == n])),
                                            "exploitability_pots_median": float(np.median(e)),
                                            "exploitability_pots_max": float(np.max(e))}
        print(f"{n} its: {np.mean(t):.2f}s mean, {np.max(t):.2f}s max; gadget expl median {np.median(e):.2e} pots", flush=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"GPU {res['gpu_max_allocated_GB']} GB, host RSS {res['host_max_rss_GB']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
