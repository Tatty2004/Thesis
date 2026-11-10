"""Gate C1 (PLAN_valuenet.md, section C): a flop+turn DCFR solve with exact river leaves
(ExactRiverLeaf), graded exactly with every river re-solved (unsafe: solve_subgames from the
blueprint's closing ranges at the turn), against a full-width lossless DCFR solve of the same
three-street game at equal iteration counts. Small deck only (5 ranks x 3 suits); the limit game,
or (--game pot) the no-limit game with short stacks (several line classes, all-in lines).

    python -m bombpot.experiments.valuenet.gate_c1 --iters 200 --checks 10,25,50,100,200 \
        --leaf-tol 1e-3 --resolve-tol 1e-3 --out results/valuenet/subgame/gate_c1.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bombpot.core.solvers.range_cfr import RangeCFR, class_reach, forward
from bombpot.core.solvers.subgame import ExactRiverLeaf
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

FLOPS = ("2c3c4d", "5c6d4c")


def make_game(name: str, stack: float = 20.0):
    if name == "limit":
        return Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=FLOPS, bet_sizes=(2, 4, 4))
    return Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=FLOPS, betting="pot", stack=stack,
                  sizes=(0.5, 1), max_aggressive=2)


def grade(full, upper, river, upper_sigma, tol, iters, variant="cfr+", check_every=10):
    """Exact exploitability of the flop+turn strategy with every river re-solved from its
    closing ranges (unsafe re-solve), on the full three-street tree."""
    resolver = ExactRiverLeaf(upper, river, tol, variant=variant, iters=iters, check_every=check_every)
    _, closing = forward(upper, upper_sigma, closing=True)
    sigma = [upper_sigma[0], upper_sigma[1], resolver.strategy(closing)]
    out = RangeCFR(full).exploitability(sigma)
    st = resolver.stats[-1]
    out.update(resolve_iterations=st["iterations"], resolve_max_exploitability=st["max_exploitability"],
               resolve_total_raw=st["total_exploitability_raw"], resolve_seconds=st["seconds"])
    return out


class CFRDSolver(RangeCFR):
    """RangeCFR with a leaf that also keeps CFR-D's average river strategy (Burch, Johanson and
    Bowling 2014): after each player's pass, that pass's subgame strategies (the leaf's solve
    from the current closing ranges) are added to a running river average at that player's
    river nodes, weighted like the trunk's average (iteration weight x own reach). The trunk's
    average with this river average is CFR-D's output strategy, the composite whose
    exploitability its convergence bound covers; a river re-solved from the trunk average's
    closing ranges is not."""

    def __init__(self, pt, leaf, **kw):
        super().__init__(pt, "dcfr", leaf=leaf, **kw)
        self.river_cum = [[np.zeros((leaf.street.num_deals, lc.num_lines, len(pt.hands), len(a)))
                           for a in lc.template.actions] for lc in leaf.classes]

    def instant_regrets(self, p: int):
        reach, regrets = super().instant_regrets(p)
        closing, subs = self.leaf._memo  # the solve this pass's backward used
        w = float(self.t) ** self.gamma
        for lc, sub, roots, cum in zip(self.leaf.classes, subs, self.leaf.roots(closing), self.river_cum):
            if lc.template.num_nodes == 0:
                continue
            nodes, _ = class_reach(lc.template, sub.strategy, tuple(r[self.leaf.group] for r in roots))
            for k, q in enumerate(lc.template.players):
                if q == p:
                    cum[k] += w * nodes[k][p][..., None] * sub.strategy[k]
        return reach, regrets

    def river_average(self) -> list:
        out = []
        for row in self.river_cum:
            out.append([])
            for c in row:
                total = c.sum(-1, keepdims=True)
                out[-1].append(np.where(total > 0, c / np.where(total > 0, total, 1.0), 1.0 / c.shape[-1]))
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
    ap.add_argument("--game", default="limit", choices=["limit", "pot"])
    ap.add_argument("--stack", type=float, default=20.0, help="chips behind (pot game)")
    ap.add_argument("--out", default="results/valuenet/subgame/gate_c1.json")
    a = ap.parse_args(argv)
    checks = sorted({int(x) for x in a.checks.split(",")} | {a.iters})
    game = make_game(a.game, a.stack)
    full = holdem_public_tree(game)
    upper = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
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
    leaf = ExactRiverLeaf(upper, river, a.leaf_tol, variant=a.inner, iters=a.sub_iters, check_every=a.check_every)
    solver = CFRDSolver(upper, leaf)
    while solver.t < a.iters:
        solver.iteration()
        if solver.t in checks:
            t0 = time.perf_counter()
            flop_turn = solver.exploitability()  # br against the leaf values
            avg = solver.hand_sigma(solver.average_strategy())
            exact = grade(full, upper, river, avg, a.resolve_tol, a.sub_iters, a.inner, a.check_every)
            cfrd = RangeCFR(full).exploitability([avg[0], avg[1], solver.river_average()])
            its = [s["iterations"] for s in leaf.stats]
            result["leaf"][solver.t] = {
                "wall_time": solver.solve_seconds, "flop_turn": flop_turn, "exact": exact, "cfrd": cfrd,
                "leaf_solves": len(its), "leaf_iterations_mean": float(np.mean(its)),
                "leaf_iterations_max": int(max(its)),
                "leaf_max_exploitability": max(s["max_exploitability"] for s in leaf.stats),
                "grade_seconds": time.perf_counter() - t0}
            ref = result["lossless"].get(solver.t, {}).get("exploitability", float("nan"))
            print(f"leaf it {solver.t:4d} {solver.solve_seconds:8.1f}s flop+turn {flop_turn['exploitability']:.5f} "
                  f"exact {exact['exploitability']:.5f} cfr-d {cfrd['exploitability']:.5f} (lossless {ref:.5f}) leaf its mean {np.mean(its):.0f} "
                  f"max {max(its)}", flush=True)
            save()


if __name__ == "__main__":
    main()
