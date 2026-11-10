"""Seconds per CFR+ iteration of a real-deck river re-solve (timing only: there is no
blueprint yet, so the ranges are uniform-ish random and the gadget's T values come from a
uniform river strategy). One turn deal of Holdem(flops=("Ah7c2d", "KsKd9h"), board_cards=(3, 1, 1)):
1,892 river deals x 1,035 hands, for `lines` of the 25 entering lines.
Usage: NUMBA_NUM_THREADS=16 python scripts/resolve_timing.py <lines> <iterations>
"""
import json
import resource
import sys
import time

import numpy as np

from bombpot.core.solvers.resolve import blueprint_cfv, resolve, restrict_lines
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

lines, iters = int(sys.argv[1]), int(sys.argv[2])
game = Holdem(flops=("Ah7c2d", "KsKd9h"), board_cards=(3, 1, 1))
t0 = time.time()
upper = holdem_public_tree(game, upto=2)
river = HoldemRiver(game, upper)
n, H, lc = river.num_children, len(upper.hands), upper.classes[2][0]
tp = restrict_lines(lc.template, np.arange(lines))
st = river.street(np.full(n, 0), np.arange(n))
contrib = lc.contrib[:lines]
rng = np.random.default_rng(0)
r0, r1 = rng.uniform(0.5, 1.0, (lines, H)), rng.uniform(0.5, 1.0, (lines, H))
bp = [np.full((n, lines, H, len(a)), 1.0 / len(a)) for a in tp.actions]
setup = time.time() - t0
t0 = time.time()
cfv = blueprint_cfv(st, tp, contrib, bp, r0, r1, 1, "br")
t_cfv = time.time() - t0
del bp
resolve(st, tp, contrib, r0, r1, cfv, 0, iters=2, check_every=1000)  # compile / warm up
t0 = time.time()
res = resolve(st, tp, contrib, r0, r1, cfv, 0, iters=iters, check_every=iters)
total = time.time() - t0
out = {"lines": lines, "river_deals": n, "hands": H, "iterations": res.iterations, "seconds": total,
       "seconds_per_iteration_incl_one_check": total / iters, "blueprint_cfv_seconds": t_cfv, "setup_seconds": setup,
       "exploitability_at_end": res.exploitability, "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
print(json.dumps(out), flush=True)
