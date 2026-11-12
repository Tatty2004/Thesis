"""Cost of the streamed exact evaluator (sampled_cfr.exact_exploitability) on the no-limit game.

    python scripts/exact_grade_timing.py small [--stack 20]   # gate C1's small deck, measured end to end
    python scripts/exact_grade_timing.py real [--deals 32]    # the real deck: ONE chunk of river deals

small: 5 ranks x 3 suits, flops 2c3c4d / 5c6d4c (gate_c1.make_game), a random stored per-class
strategy with a lossless river, graded in full: seconds for the streamed river (last_street_sums)
and for the streets above (finish_exploitability), next to the limit game on the same deck.

real: Holdem(flops Ah7c2d / KsKd9h, ante 1, 100 behind, sizes 1/2 pot and pot, cap 2). No
evaluation is run. It times the evaluator's river work (street_reach, then street_values for the
three jobs: both best responses and the value) on `--deals` river deals of turn deal 0, every
river line class, with random root reaches and a random river strategy (the cost doesn't depend
on the numbers), and multiplies up to the 2,070 x 1,892 river deals. Not timed: the river
provider (a stored lookup or the re-solves) and the flop and turn passes of finish_exploitability.
"""
import argparse
import json
import resource
import time

import numpy as np

from bombpot.core.solvers.range_cfr import street_reach, street_values
from bombpot.core.solvers.sampled_cfr import _JOBS, Lossless, finish_exploitability, last_street_sums
from bombpot.experiments.valuenet.gate_c1 import make_game
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def describe(classes_t) -> dict:
    return {"classes": len(classes_t), "lines": int(sum(lc.num_lines for lc in classes_t)),
            "line_nodes": int(sum(lc.num_lines * lc.template.num_nodes for lc in classes_t)),
            "line_actions": int(sum(lc.num_lines * sum(len(a) for a in lc.template.actions) for lc in classes_t)),
            "all_in_lines": int(sum(lc.num_lines for lc in classes_t if lc.template.num_nodes == 0))}


def small(args) -> list:
    out = []
    for name in ("limit", "pot"):
        game = make_game(name, args.stack)
        upper = holdem_public_tree(game, upto=2)
        river = HoldemRiver(game, upper)
        D1, n, H = upper.streets[-1].num_deals, river.num_children, len(upper.hands)
        a = Lossless(D1, n, H)
        rng = np.random.default_rng(0)
        sigma = []
        for t in range(3):
            rows = D1 * n if t == 2 else upper.streets[t].num_deals
            sigma.append([[rng.dirichlet(np.ones(len(acts)), size=(rows, lc.num_lines, H))
                           for acts in lc.template.actions] for lc in upper.classes[t]])
        last_street_sums(upper, river, a, sigma, chunk=args.chunk, chunks=range(1))  # compile
        best = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            sums = last_street_sums(upper, river, a, sigma, chunk=args.chunk)
            t1 = time.perf_counter()
            result = finish_exploitability(upper, sigma, sums)
            t2 = time.perf_counter()
            if best is None or t2 - t0 < best[0] + best[1]:
                best = (t1 - t0, t2 - t1)
        out.append({"deck": "small", "game": name, "stack": args.stack if name == "pot" else None, "turn_deals": D1,
                    "rivers_per_turn_deal": n, "river_deals": D1 * n, "hands": H, "chunk": args.chunk,
                    "river": describe(upper.classes[2]), "turn": describe(upper.classes[1]),
                    "river_seconds": best[0], "upper_seconds": best[1], "seconds": best[0] + best[1],
                    "river_seconds_per_deal": best[0] / (D1 * n), "peak_gb": peak_gb(), **result})
    return out


def real(args) -> list:
    game = Holdem(flops=("Ah7c2d", "KsKd9h"), board_cards=(3, 1, 1), betting="pot", stack=args.stack,
                  sizes=(0.5, 1), max_aggressive=2)
    t0 = time.perf_counter()
    upper = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
    setup = time.perf_counter() - t0
    D1, n, H = upper.streets[-1].num_deals, river.num_children, len(upper.hands)
    classes = upper.classes[2]
    rng = np.random.default_rng(0)

    def inputs(m):
        st = river.street(np.zeros(m, dtype=np.int64), np.arange(m))
        roots = [tuple(rng.uniform(0.5, 1.0, (m, lc.num_lines, H)) for _ in (0, 1)) for lc in classes]
        hsig = [[rng.dirichlet(np.ones(len(acts)), size=(m, lc.num_lines, H)) for acts in lc.template.actions]
                for lc in classes]
        return st, roots, hsig

    def work(st, roots, hsig):
        nodes, _ = street_reach(classes, hsig, roots)
        for p, mode in _JOBS:
            street_values(classes, st, hsig, nodes, p, mode, True)

    work(*inputs(2))  # compile
    x = inputs(args.deals)
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        work(*x)
        times.append(time.perf_counter() - t0)
    per_deal = min(times) / args.deals
    return [{"deck": "real", "game": "pot", "stack": args.stack, "turn_deals": D1, "rivers_per_turn_deal": n,
             "river_deals": D1 * n, "hands": H, "river": describe(classes), "turn": describe(upper.classes[1]),
             "deals_timed": args.deals, "seconds": times, "river_seconds_per_deal": per_deal,
             "projected_river_hours": per_deal * D1 * n / 3600, "setup_seconds": setup, "peak_gb": peak_gb()}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck", choices=["small", "real"])
    ap.add_argument("--stack", type=float, default=None, help="chips behind (default 20 small, 100 real)")
    ap.add_argument("--chunk", type=int, default=700)
    ap.add_argument("--deals", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.stack is None:
        args.stack = 20.0 if args.deck == "small" else 100.0
    for row in (small if args.deck == "small" else real)(args):
        print(json.dumps(row, default=float), flush=True)


if __name__ == "__main__":
    main()
