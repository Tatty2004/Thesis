"""Where do the tree and range solvers part ways on holdem-4x2-1-b2?

Diagnoses tests/test_range_cfr.py::test_iterates_match_tree_solver, which fails on Ionic for
that game. Runs both solvers side by side as the test does (CFR on the compiled tree,
RangeCFR on the public tree, 20 iterations) and, at the first iteration where their current
or average strategies differ by more than the test's 1e-9, prints every differing infoset:
its key, both solvers' cumulative regrets (full precision) and both strategies.

    .venv/bin/python scripts/ionic/diagnose_ties.py [--variants dcfr cfr+ cfr]
"""
import argparse
import platform
import sys

import numba
import numpy as np
import scipy

from bombpot.core.public import build_public_tree, to_tree_sigma, tree_coordinates
from bombpot.core.solvers.cfr import CFR
from bombpot.core.solvers.range_cfr import RangeCFR
from bombpot.core.tree import compile_game
from bombpot.holdem import Holdem

TOL = 1e-9  # the test's atol


def owners(tree, p: int) -> np.ndarray:
    """The infoset each of player p's sequences belongs to (-1 for the empty sequence)."""
    out = np.full(tree.n_seqs[p], -1, dtype=np.int64)
    for n in range(tree.n_infosets[p]):
        f, m = tree.first_seq[p][n], tree.num_actions[p][n]
        out[f:f + m] = n
    return out


def show(label: str, x: np.ndarray) -> str:
    return f"{label} [" + ", ".join(repr(float(v)) for v in x) + "]"


def report(tree, a, b, co, own, p: int, seqs: np.ndarray) -> None:
    for n in sorted({int(own[p][s]) for s in seqs}):
        t, d, l, k, i = co[p][n]
        f, m = tree.first_seq[p][n], tree.num_actions[p][n]
        print(f"  player {p} infoset {tree.keys[p][n]}")
        print("    " + show("tree  regret", a.regret[p][f:f + m]))
        print("    " + show("range regret", b.regret[t][k][d, l, i]))
        print("    " + show("tree  sigma ", a.sigma[p][f:f + m]))
        print("    " + show("range sigma ", b.sigma[t][k][d, l, i]))


def run(tree, pt, co, own, variant: str, iterations: int) -> None:
    a, b = CFR(tree, variant), RangeCFR(pt, variant)
    for it in range(1, iterations + 1):
        a.iteration()
        b.iteration()
        avg = to_tree_sigma(pt, tree, b.hand_sigma(b.average_strategy()))
        cur = to_tree_sigma(pt, tree, b.hand_sigma())
        bad_avg = [np.flatnonzero(np.abs(x - y) > TOL) for x, y in zip(a.average_strategy(), avg)]
        bad_cur = [np.flatnonzero(np.abs(x - y) > TOL) for x, y in zip(a.sigma, cur)]
        if any(len(s) for s in bad_avg + bad_cur):
            print(f"{variant}: first divergence at iteration {it}: "
                  f"current strategy differs on {sum(map(len, bad_cur))} sequences, "
                  f"average on {sum(map(len, bad_avg))}; value diff {b.value[0] - a.value[0]:.3e}")
            for p in (0, 1):
                if len(bad_cur[p]):
                    report(tree, a, b, co, own, p, bad_cur[p])
            return
    print(f"{variant}: no divergence in {iterations} iterations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["dcfr", "cfr+", "cfr"])
    ap.add_argument("--iterations", type=int, default=20)
    args = ap.parse_args()
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | scipy {scipy.__version__} | "
          f"numba {numba.__version__} | {platform.machine()}")
    game = Holdem(num_ranks=4, num_suits=2, board_cards=(1,), num_boards=2)
    tree, pt = compile_game(game), build_public_tree(game)
    co = [tree_coordinates(pt, tree, p) for p in (0, 1)]
    own = [owners(tree, p) for p in (0, 1)]
    for variant in args.variants:
        run(tree, pt, co, own, variant, args.iterations)


if __name__ == "__main__":
    main()
