"""Gate C1 (PLAN_valuenet.md, section C): a flop+turn solve with exact river leaves
(ExactRiverLeaf), graded exactly on the full three-street tree, against a full-width lossless
solve of the same game with the same variant, the same update order and equal iteration counts.
Small deck only (5 ranks x 3 suits); the limit game, or (--game pot) the no-limit game with
short stacks (several line classes, all-in lines).

Each check grades the trunk's average strategy with three rivers:
  exact      every river solved once from the trunk average's closing ranges (unsafe re-solve)
  cfr-d      CFR-D's composite (Burch et al. 2014): the river solves averaged over the trunk's
             iterations as realization plans; the strategy CFR-D's bound is about
  deepstack  every river safely re-solved (the gadget), the opponent's constraint values being
             its leaf root values averaged with the trunk's weights
"flop+turn" is the trunk alone against the leaf values (not an exploitability of a strategy).

The trunk updates both players from one leaf solve per iteration (simultaneous, CFR-D's
accounting) or, with --alternating, one after the other with a solve each, as RangeCFR does;
the lossless reference uses the same order.

    python -m bombpot.experiments.valuenet.gate_c1 --iters 200 --checks 10,20,50,100,200 \
        --trunk dcfr --inner dcfr --leaf-tol 1e-3 --resolve-tol 1e-3 --out results/valuenet/subgame/gate_c1.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bombpot.core.solvers.range_cfr import VARIANTS, RangeCFR, _update, class_reach, forward
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


class SimultaneousCFR(RangeCFR):
    """RangeCFR with simultaneous updates: both players' regrets from the same profile (and so,
    with a leaf, from one leaf solve, which `_after_values` can read)."""

    def iteration(self) -> None:
        start = time.perf_counter()
        self.t += 1
        t_ = self.t
        weight = self.average_weight()
        pos = t_ ** self.alpha / (t_ ** self.alpha + 1.0)
        neg = t_ ** self.beta / (t_ ** self.beta + 1.0)
        hsig = self.hand_sigma()
        reach = self.reach(hsig)
        regrets = {}
        for p in (0, 1):
            V, R = self.values(p, hsig, reach)
            self.value[p] = float(V[0][0][0].sum())
            regrets[p] = [[[None if r is None else self._to_ids(t, p, r, self.regret[t][c][k].shape[2])
                            for k, r in enumerate(cls)] for c, cls in enumerate(row)] for t, row in enumerate(R)]
        self._after_values()
        for p in (0, 1):
            for t, c, k in self._nodes(p):
                reg = self.regret[t][c][k]
                A = reg.shape[-1]
                self.sigma[t][c][k] = np.ascontiguousarray(self.sigma[t][c][k])
                own = np.ascontiguousarray(self.own_reach(t, c, k, reach)).reshape(-1)
                _update(reg.size, reg.reshape(-1, A), np.ascontiguousarray(regrets[p][t][c][k]).reshape(-1, A),
                        self.sigma[t][c][k].reshape(-1, A), self.cum[t][c][k].reshape(-1, A), own, weight, pos, neg,
                        VARIANTS[self.variant])
        self.solve_seconds += time.perf_counter() - start

    def _after_values(self) -> None:
        pass

    def average_weight(self) -> float:
        """Iteration t's weight in the average (RangeCFR.iteration's `weight`)."""
        t = float(self.t)
        return {"cfr": 1.0, "cfr+": t, "dcfr": t ** self.gamma}[self.variant]


class CFRDSolver(SimultaneousCFR):
    """RangeCFR with a leaf that also keeps CFR-D's average river strategy (Burch, Johanson and
    Bowling 2014): after each player's pass, that pass's subgame strategies (the leaf's solve
    from the current closing ranges) are added to a running river average at that player's
    river nodes, weighted like the trunk's average (iteration weight x own reach). The trunk's
    average with this river average is CFR-D's output strategy, the composite whose
    exploitability its convergence bound covers; a river re-solved from the trunk average's
    closing ranges is not."""

    def __init__(self, pt, leaf, variant: str = "dcfr", simultaneous: bool = True, **kw):
        """simultaneous: one leaf solve per iteration, from the current profile, gives both
        players' regrets and both players' river averages (CFR-D's accounting: each player's
        regret is measured against the opponent river strategy that enters the average). With
        alternating updates (False) the two players' passes use different solves."""
        super().__init__(pt, variant, leaf=leaf, **kw)
        self.simultaneous = simultaneous
        # DeepStack-style grading: each player's leaf root values, averaged with the trunk's
        # weights. In solver units a root value is linear in the other player's realization
        # plan, so this average is each player's value against the other's CFR-D average.
        self.cfv_cum = [[None, None] for _ in leaf.classes]
        self.weight_sum = [0.0, 0.0]
        self.river_cum = [[np.zeros((leaf.street.num_deals, lc.num_lines, len(pt.hands), len(a)))
                           for a in lc.template.actions] for lc in leaf.classes]

    def _accumulate_river(self, values_of, strategy_of) -> None:
        """Add the leaf's latest solve (its memo) to the averages: the root values of the players
        `values_of`, and the river strategies of the players `strategy_of`."""
        closing, subs = self.leaf._memo
        w = self.average_weight()
        for q in values_of:
            self.weight_sum[q] += w
        for c, sub in enumerate(subs):
            for q in values_of:
                v = w * self.leaf.root_values(sub, q)
                self.cfv_cum[c][q] = v if self.cfv_cum[c][q] is None else self.cfv_cum[c][q] + v
        for lc, sub, roots, cum in zip(self.leaf.classes, subs, self.leaf.roots(closing), self.river_cum):
            if lc.template.num_nodes == 0:
                continue
            nodes, _ = class_reach(lc.template, sub.strategy, tuple(r[self.leaf.group] for r in roots))
            for k, q in enumerate(lc.template.players):
                if q in strategy_of:
                    cum[k] += w * nodes[k][q][..., None] * sub.strategy[k]

    def instant_regrets(self, p: int):  # (alternating updates only)
        # Player p's own river strategy from this pass's solve, weighted as p's trunk average is
        # in this pass (so the composite is one realization plan). p's regrets are measured
        # against the opponent's river in this solve, which enters the average in the opponent's
        # pass from another solve: CFR-D's accounting is exact only with simultaneous updates.
        # (Averaging the opponent's river here instead was tried: 0.0196 against 0.0214 at 100
        # DCFR iterations on the one-board game, but the composite is then not the average plan.)
        reach, regrets = super().instant_regrets(p)
        self._accumulate_river((p,), (p,))
        return reach, regrets

    def iteration(self) -> None:
        if not self.simultaneous:
            return RangeCFR.iteration(self)
        return super().iteration()

    def _after_values(self) -> None:
        self._accumulate_river((0, 1), (0, 1))  # both players' values came from this one solve (memo)

    def cfv_average(self, c: int, q: int) -> np.ndarray:
        """Player q's averaged leaf root values at river class c, (D_turn, L_c, H), solver units."""
        return self.cfv_cum[c][q] / self.weight_sum[q]

    def river_average(self) -> list:
        out = []
        for row in self.river_cum:
            out.append([])
            for c in row:
                total = c.sum(-1, keepdims=True)
                out[-1].append(np.where(total > 0, c / np.where(total > 0, total, 1.0), 1.0 / c.shape[-1]))
        return out


def deepstack_grade(full, upper, solver, tol, iters, variant="dcfr", geom=None):
    """DeepStack-style grading: every river re-solved safely (resolve_batch with the gadget),
    for each player p separately, from the trunk average's closing reach of p, with the
    opponent's T values its averaged leaf root values (CFRDSolver.cfv_average); node k of each
    river class plays the re-solve of its acting player. Exact exploitability on the full tree."""
    from bombpot.core.solvers.resolve import resolve_batch

    leaf = solver.leaf if geom is None else geom  # geom: an ExactRiverLeaf for the river's layout (gate_f)
    avg = solver.hand_sigma(solver.average_strategy())
    _, closing = forward(upper, avg, closing=True)
    D1 = upper.streets[-1].num_deals
    river, gaps, its = [], 0.0, 0
    for c, (lc, roots) in enumerate(zip(leaf.classes, leaf.roots(closing))):
        if lc.template.num_nodes == 0:
            river.append([])
            continue
        res = {}
        for p in (0, 1):
            o = 1 - p
            res[p] = resolve_batch(leaf.street, lc.template, lc.contrib, leaf.group, D1, roots[p], roots[o],
                                   solver.cfv_average(c, o), p, variant, iters, tol, safe=True)
            gaps += float(np.sum(res[p].gap))
            its = max(its, int(np.max(res[p].iterations)))
        river.append([res[q].strategy[k] for k, q in enumerate(lc.template.players)])
    out = RangeCFR(full).exploitability([avg[0], avg[1], river])
    out.update(resolve_gap_sum=gaps, resolve_iterations=its)
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
    ap.add_argument("--trunk", default="dcfr", choices=["cfr", "cfr+", "dcfr"], help="trunk (and lossless) variant")
    ap.add_argument("--alternating", action="store_true",
                    help="alternating updates (two leaf solves per iteration), for the trunk and the reference")
    ap.add_argument("--leaf-values", default="br", choices=["br", "average"], help="ExactRiverLeaf's root values")
    ap.add_argument("--leaf-criterion", default="total", choices=["total", "max", "cfv"])
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
        ref = (RangeCFR if a.alternating else SimultaneousCFR)(full, a.trunk)
        while ref.t < a.iters:
            ref.iteration()
            if ref.t in checks:
                result["lossless"][ref.t] = {"wall_time": ref.solve_seconds, **ref.exploitability()}
                print(f"lossless it {ref.t:4d} {ref.solve_seconds:8.1f}s expl {result['lossless'][ref.t]['exploitability']:.5f}",
                      flush=True)
                save()

    if a.part == "lossless":
        return
    leaf = ExactRiverLeaf(upper, river, a.leaf_tol, variant=a.inner, iters=a.sub_iters, check_every=a.check_every,
                          criterion=a.leaf_criterion, values=a.leaf_values)
    solver = CFRDSolver(upper, leaf, a.trunk, simultaneous=not a.alternating)
    while solver.t < a.iters:
        solver.iteration()
        if solver.t in checks:
            t0 = time.perf_counter()
            flop_turn = solver.exploitability()  # br against the leaf values
            avg = solver.hand_sigma(solver.average_strategy())
            exact = grade(full, upper, river, avg, a.resolve_tol, a.sub_iters, a.inner, a.check_every)
            cfrd = RangeCFR(full).exploitability([avg[0], avg[1], solver.river_average()])
            deep = deepstack_grade(full, upper, solver, a.resolve_tol, a.sub_iters, a.inner)
            its = [s["iterations"] for s in leaf.stats]
            result["leaf"][solver.t] = {
                "wall_time": solver.solve_seconds, "flop_turn": flop_turn, "exact": exact, "cfrd": cfrd, "deepstack": deep,
                "leaf_solves": len(its), "leaf_iterations_mean": float(np.mean(its)),
                "leaf_iterations_max": int(max(its)),
                "leaf_max_exploitability": max(s["max_exploitability"] for s in leaf.stats),
                "grade_seconds": time.perf_counter() - t0}
            ref = result["lossless"].get(solver.t, {}).get("exploitability", float("nan"))
            print(f"leaf it {solver.t:4d} {solver.solve_seconds:8.1f}s flop+turn {flop_turn['exploitability']:.5f} "
                  f"exact {exact['exploitability']:.5f} cfr-d {cfrd['exploitability']:.5f} deepstack {deep['exploitability']:.5f} (lossless {ref:.5f}) leaf its mean {np.mean(its):.0f} "
                  f"max {max(its)}", flush=True)
            save()


if __name__ == "__main__":
    main()
