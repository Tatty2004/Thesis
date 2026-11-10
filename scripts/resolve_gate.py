"""The re-solving gate (PLAN_valuenet.md E) on the small deck: grade a blueprint from
scripts/resolve_blueprints.py exactly, then with every river re-solved (safe or unsafe) through
the evaluator's provider, and write the numbers to results/valuenet/resolve/gate_<name>_<mode>.json.

Usage: python scripts/resolve_gate.py <eq|dcfr20|kmeans_2d> <safe|unsafe> [tol] [iters] [variant]
"""
import json
import sys
import time

import numpy as np

from bombpot.core.abstraction.river import EquityBuckets
from bombpot.core.solvers.range_cfr import load_range_strategy
from bombpot.core.solvers.resolve import Resolver
from bombpot.core.solvers.sampled_cfr import (Lossless, StoredStrategy, exact_exploitability, finish_exploitability,
                                             last_street_sums)
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

sys.path.insert(0, "scripts")
from resolve_blueprints import GAME, OUT  # noqa: E402


def main(name, mode, tol=1e-4, iters=3000, variant="cfr+"):
    upper = holdem_public_tree(GAME, upto=2)
    river = HoldemRiver(GAME, upper)
    D1 = upper.streets[-1].num_deals
    sigma, info = load_range_strategy(OUT / f"{name}.npz")
    if name == "kmeans_2d":
        a = EquityBuckets.load(OUT / "kmeans_2d_buckets.npz")
    else:
        a = Lossless(D1, river.num_children, len(upper.hands))  # the full tree's river deals, in order
    before = exact_exploitability(upper, river, a, sigma, chunk=1024)
    print("blueprint", before, flush=True)
    r = Resolver(upper, river, sigma, StoredStrategy(a, sigma[2], upper.templates[2].players), safe=mode == "safe",
                 variant=variant, iters=iters, tol=tol, batch=D1)
    start = time.time()
    sums = last_street_sums(upper, river, a, sigma, chunk=1024, provider=r,
                            progress=lambda done, total: print(f"chunk {done}/{total}", time.time() - start, flush=True))
    seconds = time.time() - start
    after = finish_exploitability(upper, sigma, sums)
    print("re-solved", after, flush=True)
    gaps = {p: sum(float(g["gap"].sum()) for (d, q), g in r.diagnostics.items() if q == p) for p in (0, 1)}
    worst_excess = max(float((np.maximum(g["violation"], 0).sum(-1) - g["gap"]).max()) for g in r.diagnostics.values())
    norm_violation = max(float((np.maximum(g["violation"], 0).max(-1) / np.where(g["mass"] > 0, g["mass"], 1)).max())
                         for g in r.diagnostics.values())
    raw_violation = max(float(g["violation"].max()) for g in r.diagnostics.values())
    its = [g["iterations"] for g in r.diagnostics.values()]
    out = {"blueprint": name, "blueprint_info": info, "mode": mode, "tol": tol, "iters": iters, "variant": variant,
           "before": before, "after": after, "rise": after["exploitability"] - before["exploitability"],
           "bound": (gaps[0] + gaps[1]) / 2, "sum_gap": gaps,
           "max_root_exploitability": max(g["exploitability"] for g in r.diagnostics.values()),
           "max_violation_over_mass": norm_violation, "max_violation_raw": raw_violation,
           "max_excess_over_gap": worst_excess,
           "iterations": {"min": min(its), "median": float(np.median(its)), "max": max(its)},
           "seconds_total": seconds, "resolves": len(r.diagnostics),
           "seconds_per_resolve": seconds / len(r.diagnostics)}
    path = OUT / f"gate_{name}_{mode}_{variant}_{tol:g}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "blueprint_info"}, default=str), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1], *(float(args[2]),) if len(args) > 2 else (), *(int(args[3]),) if len(args) > 3 else (),
         *(args[4],) if len(args) > 4 else ())
