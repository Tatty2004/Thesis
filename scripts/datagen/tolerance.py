"""How the solve tolerance of a value-net sample moves its labels (real deck by default).

For datagen's samples of a seed, one DCFR trajectory per sample (FusedSubgames, float64) is
checked every 10 iterations as datagen checks it; the values at the first check within each
tolerance are exactly what datagen would store at that tolerance. They are compared with the
values at the tightest tolerance (the reference), in pots:
  max      largest |cfv - ref| over possible hands, both players
  rms      root mean square over possible hands, both players
  wmean    range-weighted mean, sum_i range_p[i] |cfv_p[i] - ref_p[i]| averaged over the players
           (the error on hands the player actually holds)
and the iterations each tolerance took (the cost: seconds are proportional).

    python scripts/datagen/tolerance.py --n 5 --seed 0 --tols 1e-2 3e-3 1e-3 3e-4 --ref 1e-5 --threads 16
"""
import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    ap.add_argument("--ref", type=float, default=1e-5)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--deck", default="13,4")
    ap.add_argument("--template", default="pot")
    ap.add_argument("--out")
    a = ap.parse_args()
    import numba

    numba.set_num_threads(a.threads)
    from bombpot.core.solvers.fused import FusedSubgames
    from bombpot.experiments.valuenet import datagen as dg

    nr, ns = (int(x) for x in a.deck.split(","))
    rng = np.random.default_rng(a.seed)
    tols = sorted(a.tols, reverse=True) + [a.ref]
    rows = []
    for n in range(a.n):
        spot = dg.sample_spot(rng, nr, ns)
        fp = dg.FlopPair(spot["flops"], nr, ns, a.template)
        spot.update(fp.sample_betting(rng, 2.0))
        d = fp.turn_deal(spot["turn"])
        valid = fp.upper.streets[1].valid[d]
        st = fp.river_street(d)
        strength = dg.turn_equity(st).mean(axis=1)
        r = [dg.random_range(rng, valid, strength), dg.random_range(rng, valid, strength)]
        pot = spot["pot"]
        f = FusedSubgames(st, [spot["template"]], [pot / 2], [tuple(r)], "dcfr", dtype=np.dtype(a.dtype))
        got, start = {}, time.perf_counter()
        while len(got) < len(tols) and f.t[0] < a.iters:
            f.iterate(10)
            ev = f.evaluate()
            expl = ev["exploitability"][0] / pot
            for tol in tols:
                if tol not in got and expl <= tol:
                    got[tol] = {"iters": int(f.t[0]), "expl": expl, "seconds": f.seconds,
                                "cfv": [ev["values"][0][p] / pot for p in (0, 1)]}
        if a.ref not in got:
            print(f"sample {n}: reference tolerance {a.ref:g} not reached in {a.iters} iterations "
                  f"(at {expl:.2e}); using the last iterate", flush=True)
            got[a.ref] = {"iters": int(f.t[0]), "expl": expl, "seconds": f.seconds,
                          "cfv": [ev["values"][0][p] / pot for p in (0, 1)]}
        ref = got[a.ref]
        scale = max(np.abs(ref["cfv"][p][valid]).max() for p in (0, 1))
        for tol in tols:
            g = got[tol]
            diff = [np.abs(g["cfv"][p] - ref["cfv"][p]) for p in (0, 1)]
            row = {"sample": n, "pot": pot, "nodes": spot["template"].num_nodes, "tol": tol, "iters": g["iters"],
                   "expl": g["expl"], "kernel_seconds": g["seconds"],
                   "max": float(max(x[valid].max() for x in diff)),
                   "rms": float(np.sqrt(np.mean(np.concatenate([x[valid] for x in diff]) ** 2))),
                   "wmean": float(np.mean([r[p] @ diff[p] for p in (0, 1)])), "ref_abs_max": float(scale),
                   "ref_wmean_abs": float(np.mean([r[p] @ np.abs(ref["cfv"][p]) for p in (0, 1)]))}
            rows.append(row)
            print(f"sample {n} pot {pot:6.2f} nodes {row['nodes']:2d} tol {tol:7.0e}  iters {row['iters']:5d}  expl "
                  f"{row['expl']:.2e}  max {row['max']:.2e}  rms {row['rms']:.2e}  wmean {row['wmean']:.2e}  "
                  f"(|ref| max {scale:.2f}, weighted {row['ref_wmean_abs']:.3f})  {time.perf_counter() - start:.0f}s",
                  flush=True)
    print("\ntol      iters(mean)  cost/1e-3  max(mean / worst)      rms(mean)   wmean(mean / worst)")
    base = np.mean([x["iters"] for x in rows if x["tol"] == 1e-3]) if 1e-3 in tols else np.nan
    for tol in tols:
        sel = [x for x in rows if x["tol"] == tol]
        it = np.mean([x["iters"] for x in sel])
        print(f"{tol:7.0e}  {it:10.1f}  {it / base:9.2f}  {np.mean([x['max'] for x in sel]):.2e} / "
              f"{max(x['max'] for x in sel):.2e}  {np.mean([x['rms'] for x in sel]):.2e}    "
              f"{np.mean([x['wmean'] for x in sel]):.2e} / {max(x['wmean'] for x in sel):.2e}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
