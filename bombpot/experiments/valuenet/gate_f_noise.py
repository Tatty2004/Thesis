"""Gate F's noise study: gate F's flop+turn solve and grades (gate_f.py) with the exact river leaf
(ExactRiverLeaf, best-response values, as gate C1) corrupted by a fixed random error of a set
size, in place of the network. The curve of grade against leaf error says how accurate a
network must be for the trunk to pass gate F.

The error is fixed: one standard normal z per (player, turn-end class, deal, line, close, hand),
drawn once from --seed, so it is systematic, as a network's is, and the trunk can steer toward
it (fresh noise every iteration would average out). Modes:
  rel    v (1 + s z): s is the relative error of every value
  scale  v + s z mean_h |v|: additive, s times the mean |value| over the hands of that
         (deal, line, close) in the same call; hands whose exact value is 0 (impossible) stay 0
The leaf error reported (gate_f.leaf_error, the noisy leaf against a fresh exact one) is
comparable with the network runs' (op1c: 0.191 weighted / 4.04 all hands, exact 2.06 / 27.2).

    python -m bombpot.experiments.valuenet.gate_f_noise --noise 0.05 --mode rel \
        --iters 100 --checks 50,100 --out results/valuenet/noise/rel_0.05.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bombpot.core.solvers.subgame import ExactRiverLeaf
from bombpot.experiments.valuenet.gate_c1 import FLOPS, deepstack_grade, grade, make_game
from bombpot.experiments.valuenet.gate_f import LeafCFR, leaf_error
from bombpot.holdem.public import HoldemRiver, holdem_public_tree


class NoisyLeaf:
    """A leaf's values with a fixed random error (module docstring)."""

    def __init__(self, leaf, noise: float, mode: str = "rel", seed: int = 0):
        if mode not in ("rel", "scale"):
            raise ValueError(f"unknown mode {mode!r}")
        self.leaf, self.noise, self.mode = leaf, noise, mode
        self.rng = np.random.default_rng(seed)
        self.z = {}

    def _z(self, p: int, c: int, shape) -> np.ndarray:
        if (p, c) not in self.z:
            self.z[p, c] = self.rng.standard_normal(shape)
        return self.z[p, c]

    def values(self, p: int, t: int, closing) -> list:
        out = []
        for c, v in enumerate(self.leaf.values(p, t, closing)):
            z = self._z(p, c, v.shape)
            if self.mode == "rel":
                out.append(v * (1.0 + self.noise * z))
            else:
                scale = np.abs(v).mean(axis=-1, keepdims=True)
                out.append(v + np.where(v != 0, self.noise * z * scale, 0.0))
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--noise", type=float, required=True, help="s: the error's size (module docstring)")
    ap.add_argument("--mode", default="rel", choices=["rel", "scale"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--checks", default="50,100")
    ap.add_argument("--trunk", default="dcfr", choices=["cfr", "cfr+", "dcfr"])
    ap.add_argument("--leaf-tol", type=float, default=1e-2, help="the exact leaf's tolerance (C1: 1e-2 / 1e-3)")
    ap.add_argument("--resolve-tol", type=float, default=1e-3)
    ap.add_argument("--sub-iters", type=int, default=5000)
    ap.add_argument("--inner", default="dcfr", choices=["cfr+", "dcfr"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    checks = sorted({int(x) for x in a.checks.split(",")} | {a.iters})
    game = make_game("limit")
    full = holdem_public_tree(game)
    upper = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
    exact_leaf = ExactRiverLeaf(upper, river, a.leaf_tol, variant=a.inner, iters=a.sub_iters, values="br")
    leaf = NoisyLeaf(exact_leaf, a.noise, a.mode, a.seed)
    geom = ExactRiverLeaf(upper, river, a.resolve_tol)
    result = {"args": vars(a), "flops": FLOPS, "checks": {}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    solver = LeafCFR(upper, leaf, a.trunk)
    while solver.t < a.iters:
        solver.iteration()
        if solver.t in checks:
            t0 = time.perf_counter()
            flop_turn = solver.exploitability()
            avg = solver.hand_sigma(solver.average_strategy())
            exact = grade(full, upper, river, avg, a.resolve_tol, a.sub_iters, a.inner)
            deep = deepstack_grade(full, upper, solver, a.resolve_tol, a.sub_iters, a.inner, geom=geom)
            err = leaf_error(upper, river, leaf, avg, a.resolve_tol, a.sub_iters, a.inner)
            result["checks"][solver.t] = {"wall_time": solver.solve_seconds, "flop_turn": flop_turn, "exact": exact,
                                          "deepstack": deep, **err, "grade_seconds": time.perf_counter() - t0}
            print(f"noise {a.mode} {a.noise:g} it {solver.t:4d} {solver.solve_seconds:7.1f}s flop+turn "
                  f"{flop_turn['exploitability']:.5f} exact {exact['exploitability']:.5f} deepstack "
                  f"{deep['exploitability']:.5f} | leaf error {err['leaf_error_weighted']:.5f} weighted, "
                  f"{err['leaf_error_all']:.4f} all hands (exact values {err['exact_abs_weighted']:.3f}, "
                  f"{err['exact_abs_all']:.2f})", flush=True)
            out.write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
