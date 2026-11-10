"""Blueprints for the re-solving gate (PLAN_valuenet.md E) on the small three-street deck of
tests/holdem/test_holdem_river.py, saved under results/valuenet/resolve/:
  eq.npz        full-width lossless DCFR on holdem_public_tree(game), to exploitability <= 1e-6
                or 2,000 iterations
  dcfr20.npz    the same solver after 20 iterations (a poor blueprint)
  kmeans_2d.npz the sampled solver with kmeans_2d river buckets (k = 3), 200 iterations of every
                turn deal x 4 rivers (a bucketed blueprint), stored per bucket
Usage: python scripts/resolve_blueprints.py {eq,dcfr20,kmeans_2d}
"""
import sys
import time
from pathlib import Path

import numpy as np

from bombpot.core.abstraction.river import fit_equity_buckets
from bombpot.core.solvers.range_cfr import RangeCFR, save_range_strategy
from bombpot.core.solvers.sampled_cfr import SampledRangeCFR
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

GAME = Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=("2c3c4d", "5c6d4c"), bet_sizes=(2, 4, 4))
OUT = Path("results/valuenet/resolve")


def main(which: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if which in ("eq", "dcfr20"):
        solver = RangeCFR(holdem_public_tree(GAME))
        cap, every = (2000, 100) if which == "eq" else (20, 20)
        log = []
        while solver.t < cap:
            for _ in range(every):
                solver.iteration()
            row = solver.log_row()
            log.append(row)
            print({k: (round(v, 10) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            if row["exploitability"] <= 1e-6:
                break
            save_range_strategy(OUT / f"{which}.npz", solver.average_strategy(), **row)
        save_range_strategy(OUT / f"{which}.npz", solver.average_strategy(), **log[-1])
    elif which == "kmeans_2d":
        upper = holdem_public_tree(GAME, upto=2)
        river = HoldemRiver(GAME, upper)
        D1 = upper.streets[-1].num_deals
        a = fit_equity_buckets(river, D1, "kmeans_2d", 3, samples=12, seed=0)
        solver = SampledRangeCFR(upper, river, a, batch=D1, samples=4, seed=0)
        start = time.time()
        for _ in range(200):
            solver.iteration()
        save_range_strategy(OUT / "kmeans_2d.npz", solver.average_strategy(), iterations=200,
                            seconds=time.time() - start)
        a.save(OUT / "kmeans_2d_buckets.npz")
        print("kmeans_2d saved", time.time() - start, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
