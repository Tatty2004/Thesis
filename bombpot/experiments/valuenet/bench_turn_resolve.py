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


def profile_leaf(spot, s: dict, n: int, compile_modes=()) -> dict:
    """Seconds per network leaf call at one spot (13 rows): the whole values_both call, the
    network's forward alone on the same inputs, and with torch.compile modes."""
    import torch

    from bombpot.core.solvers.range_cfr import class_reach, stack_closing

    leaf, cls = s["leaf"], s["cls"]
    tp = cls.template
    sig = [np.full(s["reach"].shape[1:] + (len(a),), 1.0 / len(a)) for a in tp.actions]
    _, closing = class_reach(tp, sig, (s["reach"][0], s["reach"][1]))
    cl = stack_closing([closing])
    sync = (lambda: torch.cuda.synchronize()) if spot.device == "cuda" else (lambda: None)

    def timed(f):
        f()
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        sync()
        return (time.perf_counter() - t0) / n

    out = {"call": timed(lambda: leaf.values_both(1, cl)), "call_one_player": timed(lambda: leaf.values(0, 1, cl))}
    rep = leaf.replicas[0]
    dev, H = rep["device"], leaf.H
    rows = cl[0][0].shape[1] * cl[0][0].shape[2]
    d = torch.zeros(rows, dtype=torch.long, device=dev)
    hist = torch.full((rows, H), 1.0 / H, device=dev, dtype=leaf.dtype)
    board = rep["board"][d] if leaf.cfg.board == "ids" else rep["board"][d].to(leaf.dtype)
    scal = torch.zeros(rows, 2, device=dev, dtype=leaf.dtype)
    hand = {"hx": rep["hx"][d], "hc": rep["hc"].expand(rows, H, 2)}
    model = rep["model"]

    def fwd(m):
        def f():
            with torch.no_grad(), torch.autocast("cuda", enabled=leaf.amp and dev.type == "cuda"):
                m(board, scal, hist, hist, **hand)
        return f

    out["forward"] = timed(fwd(model))
    for mode in compile_modes:
        try:
            t0 = time.perf_counter()
            cm = torch.compile(model, mode=mode)
            fwd(cm)()
            out[f"compile_{mode}_seconds"] = time.perf_counter() - t0
            out[f"forward_compiled_{mode}"] = timed(fwd(cm))
        except Exception as e:  # report, don't fail the bench
            out[f"compile_{mode}_error"] = repr(e)[:300]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--spots", type=int, default=6)
    ap.add_argument("--iters", default="100,300")
    ap.add_argument("--history", type=int, default=300)
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--cfv", default="auto", choices=["auto", "avg", "br"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="alt,sim", help="alt (alternating updates), sim (simultaneous)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--profile", type=int, default=0, help="also time N network calls at the first spot")
    ap.add_argument("--compile", default="", help="torch.compile modes to time in --profile, e.g. reduce-overhead")
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
    if a.profile:
        res["profile"] = profile_leaf(spot, s, a.profile, [m for m in a.compile.split(",") if m])
        print("profile (seconds per call):", json.dumps(res["profile"]), flush=True)
        out.write_text(json.dumps(res, indent=1, default=float))
    for d, l in picks:
        t0 = time.perf_counter()
        s = spot.setup(d, 0, l, a.cfv)
        setup = time.perf_counter() - t0
        for p in (0, 1):
            if float(s["reach"][p].sum()) <= 0:
                continue
            for mode in a.modes.split(","):
                sim = mode == "sim"
                for n in lens:
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    g, ev = spot.resolve(s, p, n, simultaneous=sim)
                    r = report(spot, s, g, ev, p)
                    r.update(spot_setup_seconds=setup, tree_seconds=g.seconds - g.leaf_seconds, mode=mode)
                    res["runs"].append(r)
                    print(f"deal {d} line {r['flop_line']:9s} p{p} {mode} {n:4d} its: {g.seconds:6.2f}s (net "
                          f"{g.leaf_seconds:5.2f}s, {g.leaf_calls} calls), expl {r['exploitability_pots']:.2e} pots, "
                          f"setup {setup:.2f}s", flush=True)
                if a.history:
                    g, ev = spot.resolve(s, p, a.history, tol=0.0, check_every=a.check_every, simultaneous=sim)
                    r = report(spot, s, g, ev, p)
                    r["mode"] = mode
                    res["histories"].append(r)
                    print(f"   history p{p} {mode}: " + ", ".join(f"{t}:{x / s['pot']:.2e}" for t, x in r["history"]),
                          flush=True)
        res["gpu_max_allocated_GB"] = torch.cuda.max_memory_allocated() / 2 ** 30 if a.device == "cuda" else None
        res["host_max_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        out.write_text(json.dumps(res, indent=1, default=float))
    for mode in a.modes.split(","):
        for n in lens:
            rs = [r for r in res["runs"] if r["iterations"] == n and r["mode"] == mode]
            t = [r["seconds"] for r in rs]
            e = [r["exploitability_pots"] for r in rs]
            res.setdefault("summary", {})[f"{mode}_{n}"] = {
                "seconds_mean": float(np.mean(t)), "seconds_max": float(np.max(t)),
                "net_share": float(np.mean([r["leaf_seconds"] / r["seconds"] for r in rs])),
                "exploitability_pots_median": float(np.median(e)), "exploitability_pots_max": float(np.max(e))}
            print(f"{mode} {n} its: {np.mean(t):.2f}s mean, {np.max(t):.2f}s max; gadget expl median "
                  f"{np.median(e):.2e} pots, max {np.max(e):.2e}", flush=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"GPU {res['gpu_max_allocated_GB']} GB, host RSS {res['host_max_rss_GB']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
