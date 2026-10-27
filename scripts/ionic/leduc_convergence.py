"""Leduc convergence past tests/test_range_cfr.py::test_converges_to_the_lp_value[leduc].

The test's setup (Leduc, the sequence-form LP's value, default RangeCFR = DCFR), run longer:
prints exploitability at --checkpoints and whether the LP value lies in the best-response
bracket [-br_value_p1, br_value_p0] (the test's second check, 1e-9 slack). The tree solver
(CFR on the compiled tree, also DCFR) runs alongside for comparison.

    .venv/bin/python scripts/ionic/leduc_convergence.py [--checkpoints 1500 2000 3000]
"""
import argparse
import platform
import sys

import numpy as np

from bombpot.core.public import build_public_tree
from bombpot.core.solvers.cfr import CFR
from bombpot.core.solvers.lp import solve_game
from bombpot.core.solvers.range_cfr import RangeCFR
from bombpot.core.tree import compile_game
from bombpot.toygames import Leduc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[1500, 2000, 3000])
    args = ap.parse_args()
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | {platform.machine()}")
    game = Leduc()
    tree, pt = compile_game(game), build_public_tree(game)
    (v, v1), _ = solve_game(tree)
    print(f"LP value {v!r} (player 1's LP: {v1!r})")
    solvers = {"range": RangeCFR(pt), "tree": CFR(tree)}
    for name, s in solvers.items():
        for target in sorted(args.checkpoints):
            while s.t < target:
                s.iteration()
            rep = s.log_row()
            lo, hi = -rep["br_value_p1"], rep["br_value_p0"]
            print(f"{name:5s} iteration {s.t:5d}: exploitability {rep['exploitability']:.6e}  "
                  f"bracket [{lo:.12f}, {hi:.12f}]  LP value bracketed: {lo - 1e-9 <= v <= hi + 1e-9}",
                  flush=True)


if __name__ == "__main__":
    main()
