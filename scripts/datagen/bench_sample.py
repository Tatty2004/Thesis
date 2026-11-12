"""Time real-deck value-net samples (experiments/valuenet/datagen.py) for one solver setup.

The samples are datagen's for the seed (same flop pairs, turns, betting, ranges), so runs with
different solvers, precisions, thread counts and tolerances solve the same subgames. Prints
per-sample setup and solve seconds, iterations, exploitability, and the peak RSS; with
--save, writes the values (float64, in pots) to an .npz so that runs can be compared
(scripts/datagen/compare_values.py).

    python scripts/datagen/bench_sample.py --n 5 --seed 0 --solver fused --dtype float32 --threads 4
"""
import argparse
import resource
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--solver", default="fused")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--variant", default="dcfr")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--template", default="pot")
    ap.add_argument("--spread", type=float, default=2.0)
    ap.add_argument("--deck", default="13,4")
    ap.add_argument("--only", type=int, nargs="*", help="solve only these sample indices (the others are drawn, "
                                                        "so the random stream stays datagen's, but not solved)")
    ap.add_argument("--save")
    a = ap.parse_args()
    import numba

    numba.set_num_threads(a.threads)
    from bombpot.experiments.valuenet import datagen as dg

    nr, ns = (int(x) for x in a.deck.split(","))
    rng = np.random.default_rng(a.seed)
    rows, total_setup, total_solve, done = {}, 0.0, 0.0, 0
    start = time.perf_counter()
    for n in range(a.n):
        t0 = time.perf_counter()
        spot = dg.sample_spot(rng, nr, ns)
        fp = dg.FlopPair(spot["flops"], nr, ns, a.template)
        spot.update(fp.sample_betting(rng, a.spread))
        skip = a.only is not None and n not in a.only
        row = dg.solve_sample(fp, spot, rng, a.tol, 0 if skip else a.iters, a.variant, a.solver, a.dtype)
        if skip:
            continue
        wall = time.perf_counter() - t0
        done += 1
        total_setup += wall - row["seconds"]
        total_solve += row["seconds"]
        rows[n] = row
        print(f"sample {n}  pot {spot['pot']:7.2f}  nodes {spot['template'].num_nodes:2d}  iters {row['iters']:5d}  "
              f"expl {row['subgame_exploitability']:.3e}  setup {wall - row['seconds']:6.1f}s  solve "
              f"{row['seconds']:7.1f}s", flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    cpu = time.process_time()
    print(f"RESULT solver {a.solver} dtype {a.dtype} threads {a.threads} tol {a.tol:g} samples {done}  "
          f"per sample: wall {(total_setup + total_solve) / done:.1f}s (setup {total_setup / done:.1f}s, solve "
          f"{total_solve / done:.1f}s)  process CPU {cpu / done:.1f}s  threads x wall "
          f"{a.threads * (total_setup + total_solve) / done:.0f}s  peak RSS {peak:.2f} GB  total "
          f"{time.perf_counter() - start:.0f}s", flush=True)
    if a.save:
        np.savez(a.save, idx=np.array(sorted(rows)), **{f"{k}_{n}": np.asarray(rows[n][k]) for n in rows
                                                        for k in ("cfv0", "cfv1", "range0", "range1", "possible",
                                                                  "subgame_exploitability", "iters")})


if __name__ == "__main__":
    main()
