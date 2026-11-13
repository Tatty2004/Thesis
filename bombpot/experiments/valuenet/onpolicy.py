"""On-policy training data for the value network (DeepStack's second data source, PLAN_valuenet.md F):
the ranges a flop+turn solve with a network leaf actually asks the leaf about, labelled by exact
river solves. Small-deck limit game of gate F only (one flop pair, every turn deal).

A solve with the given network as the leaf (gate_f.LeafCFR) runs `--iters` iterations. At each
iteration in `--snapshots`, the closing ranges of both players' passes (the leaf's queries) and of
the average strategy are taken; every (turn deal, line, close) of each is one sample: both ranges
masked to the hands possible at that deal and normalized to sum 1 (uniform if a player has no mass
there, as NetLeaf feeds it), the river subgame at that line's pot solved exactly to `--tol` pots
(FusedSubgames, DCFR, river-only chance weights), stored in datagen's shard layout (README.md):
cfv0/cfv1 and br0/br1 in pots, line = the river class line, pot_spread = 1.

    python -m bombpot.experiments.valuenet.onpolicy --model results/valuenet/net/<run>/best.pt \
        --iters 100 --snapshots 2,4,...  --out results/valuenet/data/<dir>/onpolicy_<k>.h5

`--part k/n` takes every n-th sample from the k-th (for parallel processes).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from bombpot.core.solvers.fused import FusedSubgames
from bombpot.core.solvers.range_cfr import forward
from bombpot.core.solvers.subgame import line_template
from bombpot.experiments.valuenet import datagen as dg
from bombpot.experiments.valuenet.gate_c1 import make_game
from bombpot.experiments.valuenet.gate_f import LeafCFR, net_leaf, turn_pots
from bombpot.holdem.public import HoldemRiver, holdem_public_tree


def normalized(r: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.where(valid, r, 0.0)
    m = x.sum()
    return x / m if m > 0 else valid / valid.sum()


def collect(model, iters: int, snapshots, trunk: str = "dcfr"):
    """Closing ranges [(r0, r1) each (D, L, C, H)] of the leaf's queries at the snapshot
    iterations (both passes) and of the average strategy there."""
    game = make_game("limit")
    upper = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
    solver = LeafCFR(upper, net_leaf(model, game, upper, river), trunk)
    out = []
    while solver.t < iters:
        solver.iteration()
        if solver.t in snapshots:
            for p in (0, 1):  # pass p queried with the current ranges of that moment
                out.append(tuple(np.array(x) for x in solver.leaf.closing[p][0]))
            _, closing = forward(upper, solver.hand_sigma(solver.average_strategy()), closing=True)
            out.append(tuple(np.array(x) for x in closing[0]))
    return game, upper, river, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--snapshots", default=",".join(str(x) for x in range(2, 101, 2)))
    ap.add_argument("--tol", type=float, default=1e-4, help="subgame exploitability target, pots")
    ap.add_argument("--sub-iters", type=int, default=5000)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--batch", type=int, default=400, help="subgames per fused batch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from bombpot.valuenet.model import load_model

    k, n_parts = (int(x) for x in a.part.split("/"))
    snaps = {int(x) for x in a.snapshots.split(",")}
    t0 = time.perf_counter()
    game, upper, river, ranges = collect(load_model(a.model), a.iters, snaps)
    print(f"collected {len(ranges)} range sets in {time.perf_counter() - t0:.0f}s", flush=True)
    turn = upper.streets[1]
    D, H = turn.valid.shape
    pots = turn_pots(upper)[0]
    rc = upper.classes[2][0]
    line_of = {(int(rc.parent_line[l]), int(rc.parent_close[l])): l for l in range(rc.num_lines)}
    tp = line_template(rc.template, 0)  # the limit river's betting is the same on every line
    L, C = pots.shape
    items = [(s, d, l, c) for s in range(len(ranges)) for d in range(D) for l in range(L) for c in range(C)
             if (l, c) in line_of]
    items = items[k::n_parts]
    hands = np.array(upper.hands, dtype=np.int64)
    attrs = {"num_ranks": 5, "num_suits": 3, "board_cards": "3,1,1", "num_boards": 2, "ante": 1.0,
             "total_chips": dg.STACK, "template": "limit", "source": "onpolicy", "model": a.model,
             "trunk_iters": a.iters, "snapshots": a.snapshots, "tol_pots": a.tol, "iters_cap": a.sub_iters,
             "variant": "dcfr", "solver": "fused", "git": dg.git_hash(), "units": dg.UNITS}
    shard = dg.Shard(a.out, H, attrs)
    hset = shard.add_hands(hands)
    flops = np.array(game.flops)
    by_deal: dict = {}
    for it in items:
        by_deal.setdefault(it[1], []).append(it)
    done, start = 0, time.perf_counter()
    for d, its in sorted(by_deal.items()):
        valid = turn.valid[d]
        st = river.street(*river.all_children([d]))  # as FlopPair.river_street: river-only chance weights
        st.kappa = np.full(st.num_deals, river.scale)
        equity = dg.turn_equity(st)
        for b in range(0, len(its), a.batch):
            chunk = its[b:b + a.batch]
            roots = [tuple(normalized(ranges[s][q][d, l, c], valid) for q in (0, 1)) for s, _, l, c in chunk]
            pot = [float(pots[l, c]) for _, _, l, c in chunk]
            f = FusedSubgames(st, [tp] * len(chunk), [x / 2 for x in pot], roots, "dcfr")
            ev = f.solve(a.sub_iters, np.array(pot) * a.tol)
            for j, (s, _, l, c) in enumerate(chunk):
                P = pot[j]
                v = [ev["values"][j][q] / P for q in (0, 1)]
                br = [ev["br_values"][j][q] / P for q in (0, 1)]
                r0, r1 = roots[j]
                shard.add({"flops": flops, "turn": np.asarray(river.turn_cards[d]), "hand_set": hset, "pot": P,
                           "stack": dg.STACK - P / 2, "line": int(rc.lines[line_of[(l, c)]]), "pot_spread": 1.0,
                           "range0": r0, "range1": r1, "possible": valid, "equity": equity, "cfv0": v[0],
                           "cfv1": v[1], "br0": br[0], "br1": br[1], "zero_sum_residual": float(r0 @ v[0] + r1 @ v[1]),
                           "subgame_exploitability": ev["exploitability"][j] / P, "iters": ev["iterations"][j],
                           "seconds": 0.0, "mass": ev["mass"][j]})
            done += len(chunk)
        print(f"deal {d}: {done}/{len(items)} samples, {time.perf_counter() - start:.0f}s", flush=True)
    shard.close()


if __name__ == "__main__":
    main()
