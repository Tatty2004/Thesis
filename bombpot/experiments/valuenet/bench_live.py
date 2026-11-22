"""Time the viewer's live mode end to end (viewer_server.Viewer.state, as the page calls it) on
one device: for `--spots` random turn deals after a few flop lines, the live turn re-solve
(both players) in each --configs setting, then for the first setting the live river re-solve
of the checked-down turn (exact: every river card of the line; and quick: one river).

    python -m bombpot.experiments.valuenet.bench_live --trunk <dir> --cache /tmp/x --out bench.json \
        [--configs alt300p,alt300u,sim300p,sim150p] [--device cuda|cpu|mps]

A config is alt|sim (update mode), the iterations, and p (paired: both players share network
calls) or u (one after the other). Caches go to --cache (never the trunk directory), one
subdirectory per config, so every solve is timed cold.
"""
from __future__ import annotations

import argparse
import json
import re
import resource
import time
from pathlib import Path

import numpy as np

from bombpot.experiments.valuenet import viewer_server as vs


def parse_config(c: str) -> dict:
    m = re.fullmatch(r"(alt|sim)(\d+)([pu])", c)
    if not m:
        raise SystemExit(f"bad config {c!r}")
    return {"simultaneous": m[1] == "sim", "iters": int(m[2]), "paired": m[3] == "p"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model")
    ap.add_argument("--cfv", default="auto")
    ap.add_argument("--configs", default="alt300p,alt300u,sim300p,sim150p")
    ap.add_argument("--spots", type=int, default=4)
    ap.add_argument("--rivers", type=int, default=2, help="spots whose river is also timed (first config)")
    ap.add_argument("--flop-lines", default="0,0;1,1;0,1,1", help="flop action paths (indices), ';'-separated")
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--sub-iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.cuda.get_device_name(0) if a.device == "cuda" else a.device
    res = {"args": vars(a), "device": dev, "threads": torch.get_num_threads(), "runs": [], "rivers": []}
    rng = np.random.default_rng(a.seed)
    lines = a.flop_lines.split(";")
    first = None
    for ci, cname in enumerate(a.configs.split(",")):
        cfg = parse_config(cname)
        t0 = time.perf_counter()
        v = vs.Viewer(Path(a.trunk), a.device, a.tol, a.sub_iters, cache=Path(a.cache) / cname,
                      live=dict(cfg, model=a.model, cfv=a.cfv, check_every=20, keep_best=True))
        _ = v.live  # network, flop strategy, T-value source
        res.setdefault("setup_seconds", {})[cname] = time.perf_counter() - t0
        if first is None:
            first = v
            picks = [(lines[i % len(lines)], int(rng.integers(v.upper.streets[1].num_deals))) for i in range(a.spots + 1)]
        for j, (fa, d) in enumerate(picks):
            ta, tb = (int(x) for x in v.river.turn_cards[d])
            turn = vs.card_str(ta) + vs.card_str(tb)
            q = {"fa": fa, "turn": turn, "live": "1"}
            t1 = time.perf_counter()
            st = v.state(q)
            wall = time.perf_counter() - t1
            if j == 0:  # warm-up (compiles, first network calls)
                res.setdefault("warmup_seconds", {})[cname] = wall
                continue
            src = st["source"]
            row = {"config": cname, "deal": d, "flop": fa, "wall": wall, "gap_pots": src["gap_pots"],
                   "kept": src["kept"], "seconds": src["seconds"]}
            res["runs"].append(row)
            print(f"{cname:8s} deal {d:4d} flop {fa:6s}: {wall:6.2f}s, gaps {src['gap_pots'][0]:.2e} / "
                  f"{src['gap_pots'][1]:.2e} pots, kept {src['kept']}", flush=True)
            if ci == 0 and j <= a.rivers:
                st = v.state(dict(q, ta="0,0"))
                if st.get("need") != "river":
                    continue
                rem = [int(x) for x in v.river.remaining[d]]
                ra, rb = rem[int(rng.integers(len(rem)))], None
                rb = next(x for x in rem[::-1] if x != ra)
                river = vs.card_str(ra) + vs.card_str(rb)
                for mode in ("exact", "quick"):
                    t1 = time.perf_counter()
                    st = v.state(dict(q, ta="0,0", river=river, mode=mode))
                    w = time.perf_counter() - t1
                    r = st["resolve"]
                    res["rivers"].append({"deal": d, "flop": fa, "mode": mode, "wall": w, "gap_pots": r["gap_pots"],
                                          "iterations": r["iterations"], "seconds": r["seconds"]})
                    print(f"   river {mode:5s}: {w:6.2f}s, {r['iterations']} its, gaps {r['gap_pots'][0]:.2e} / "
                          f"{r['gap_pots'][1]:.2e} pots", flush=True)
        res["host_max_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        res["gpu_max_allocated_GB"] = torch.cuda.max_memory_allocated() / 2 ** 30 if a.device == "cuda" else None
        out.write_text(json.dumps(res, indent=1, default=float))
    for cname in a.configs.split(","):
        rs = [r for r in res["runs"] if r["config"] == cname]
        if rs:
            w = [r["wall"] for r in rs]
            g = [x for r in rs for x in r["gap_pots"]]
            res.setdefault("summary", {})[cname] = {"wall_mean": float(np.mean(w)), "wall_max": float(np.max(w)),
                                                    "gap_pots_median": float(np.median(g)), "gap_pots_max": float(np.max(g))}
            print(f"{cname}: {np.mean(w):.2f}s mean, {np.max(w):.2f}s max; gap median {np.median(g):.2e}, max "
                  f"{np.max(g):.2e} pots", flush=True)
    out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
