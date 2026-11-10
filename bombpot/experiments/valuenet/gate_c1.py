"""Gate C1 (PLAN_valuenet.md, section C): a flop+turn DCFR solve with exact river leaves
(ExactRiverLeaf), graded exactly with every river re-solved (unsafe: solve_subgames from the
blueprint's closing ranges at the turn), against a full-width lossless DCFR solve of the same
three-street game at equal iteration counts. Small deck only (5 ranks x 3 suits).

    python -m bombpot.experiments.valuenet.gate_c1 --iters 200 --checks 10,25,50,100,200 \
        --leaf-tol 1e-3 --resolve-tol 1e-3 --out results/valuenet/subgame/gate_c1.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bombpot.core.solvers.range_cfr import RangeCFR, forward
from bombpot.core.solvers.subgame import ExactRiverLeaf, solve_subgames
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

GAME = Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=("2c3c4d", "5c6d4c"), bet_sizes=(2, 4, 4))


def grade(full, upper, leaf, upper_sigma, tol, iters, variant="cfr+"):
    """Exact exploitability of the flop+turn strategy with every river re-solved from its
    closing ranges (unsafe re-solve), on the full three-street tree."""
    _, closing = forward(upper, upper_sigma, closing=True)
    D, L, C, H = closing[0].shape
    sub = solve_subgames(leaf.street, full.templates[2], full.contrib[2], closing[0].reshape(D, L * C, H),
                         closing[1].reshape(D, L * C, H), leaf.group, variant, iters, tol,
                         criterion="total")
    sigma = [list(upper_sigma[0]), list(upper_sigma[1]), list(sub.strategy)]
    out = RangeCFR(full).exploitability(sigma)
    out.update(resolve_iterations=sub.iterations, resolve_max_exploitability=float(sub.exploitability.max()),
               resolve_total_raw=float(sub.exploitability_raw.sum()), resolve_seconds=sub.seconds)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--checks", default="10,25,50,100,200")
    ap.add_argument("--leaf-tol", type=float, default=1e-3)
    ap.add_argument("--resolve-tol", type=float, default=1e-3)
    ap.add_argument("--sub-iters", type=int, default=5000)
    ap.add_argument("--inner", default="cfr+", choices=["cfr+", "dcfr"], help="subgame solver variant")
    ap.add_argument("--check-every", type=int, default=10, help="subgame tolerance checks")
    ap.add_argument("--part", default="both", choices=["both", "lossless", "leaf"])
    ap.add_argument("--out", default="results/valuenet/subgame/gate_c1.json")
    a = ap.parse_args(argv)
    checks = sorted({int(x) for x in a.checks.split(",")} | {a.iters})
    full = holdem_public_tree(GAME)
    upper = holdem_public_tree(GAME, upto=2)
    river = HoldemRiver(GAME, upper)
    result = {"args": vars(a), "lossless": {}, "leaf": {}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(result, indent=1, default=float))

    if a.part in ("both", "lossless"):
        ref = RangeCFR(full, "dcfr")
        while ref.t < a.iters:
            ref.iteration()
            if ref.t in checks:
                result["lossless"][ref.t] = {"wall_time": ref.solve_seconds, **ref.exploitability()}
                print(f"lossless it {ref.t:4d} {ref.solve_seconds:8.1f}s expl {result['lossless'][ref.t]['exploitability']:.5f}",
                      flush=True)
                save()

    if a.part == "lossless":
        return
    leaf = ExactRiverLeaf(upper, river, full.templates[2], full.contrib[2], a.leaf_tol, a.inner, a.sub_iters,
                          a.check_every)
    solver = RangeCFR(upper, "dcfr", leaf=leaf)
    while solver.t < a.iters:
        solver.iteration()
        if solver.t in checks:
            t0 = time.perf_counter()
            flop_turn = solver.exploitability()  # br against the leaf values
            avg = solver.hand_sigma(solver.average_strategy())
            exact = grade(full, upper, leaf, avg, a.resolve_tol, a.sub_iters, a.inner)
            its = [s["iterations"] for s in leaf.stats]
            result["leaf"][solver.t] = {
                "wall_time": solver.solve_seconds, "flop_turn": flop_turn, "exact": exact,
                "leaf_solves": len(its), "leaf_iterations_mean": float(np.mean(its)),
                "leaf_iterations_max": int(max(its)),
                "leaf_max_exploitability": max(s["max_exploitability"] for s in leaf.stats),
                "grade_seconds": time.perf_counter() - t0}
            ref = result["lossless"].get(solver.t, {}).get("exploitability", float("nan"))
            print(f"leaf it {solver.t:4d} {solver.solve_seconds:8.1f}s flop+turn {flop_turn['exploitability']:.5f} "
                  f"exact {exact['exploitability']:.5f} (lossless {ref:.5f}) leaf its mean {np.mean(its):.0f} "
                  f"max {max(its)}", flush=True)
            save()


if __name__ == "__main__":
    main()
