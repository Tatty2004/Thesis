"""Gate F (PLAN_valuenet.md, section F): a flop+turn solve of the small deck's limit game with a
trained value network as the turn-end leaf (NetLeaf), graded exactly on the full three-street
tree. The numbers stand next to gate C1's (the same solve with exact river leaves): the gap is
the network's cost.

Each check grades the trunk's average strategy three ways:
  flop+turn  the trunk alone against the network's values (not an exploitability of a strategy;
             what the solver believes)
  exact      every river solved once from the trunk average's closing ranges (unsafe re-solve)
  deepstack  every river safely re-solved (the gadget), the opponent's constraint values being
             the network's values for it, averaged over the trunk's iterations with the trunk's
             weights. This is the strategy the design plays (PLAN_valuenet.md, 3).
and measures the network at the final average strategy's turn-end ranges against exact river
solves there (ExactRiverLeaf, best-response root values), in solver units (chips per hand of
the whole game): `leaf_error_weighted` sums |network - exact| x the player's own reach over
every turn-end state and hand (the error of the values of the hands the strategy holds), and
`leaf_error_all` leaves the own reach out (every hand counts, held or not: what the trunk's
regrets see).

    python -m bombpot.experiments.valuenet.gate_f --model results/valuenet/net/<run>/best.pt \
        --iters 200 --checks 50,100,200 --out results/valuenet/net/<run>/gate_f.json

Needs torch (the scratch environment).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bombpot.core.solvers.range_cfr import RangeCFR, forward
from bombpot.core.solvers.subgame import ExactRiverLeaf
from bombpot.experiments.valuenet import datagen as dg
from bombpot.experiments.valuenet.gate_c1 import FLOPS, deepstack_grade, grade, make_game
from bombpot.holdem.public import HoldemRiver, holdem_public_tree


def turn_pots(upper) -> list:
    """[c] (L_c, closes_c): the pot at each turn-closing line, twice the contrib of the river
    line that continues it."""
    pots = [np.zeros((lc.num_lines, len(lc.template.closes))) for lc in upper.classes[1]]
    for rc in upper.classes[2]:
        for l in range(rc.num_lines):
            pots[rc.parent_class[l]][rc.parent_line[l], rc.parent_close[l]] = 2.0 * rc.contrib[l]
    if any((p <= 0).any() for p in pots):
        raise ValueError("a turn close without a river line")
    return pots


def net_leaf(model, game, upper, river, device="cpu"):
    """NetLeaf for a game's turn end: equities, boards, pots and chance weights from the game.
    Only games whose river lines all have betting (no all-in classes: the limit game)."""
    from bombpot.valuenet.leaf import NetLeaf

    if any(rc.template.num_nodes == 0 for rc in upper.classes[2]):
        raise NotImplementedError("river classes without betting (all-in) need exact showdown values")
    turn = upper.streets[1]
    D = turn.num_deals
    equity = np.stack([dg.turn_equity(river.street(*river.all_children([d]))) for d in range(D)])
    flops = np.array(game.flops)
    boards = np.concatenate([np.broadcast_to(flops, (D, 2, 3)), np.asarray(river.turn_cards)[:, :, None]], axis=-1)
    pots = turn_pots(upper)
    stacks = [dg.STACK - p / 2 for p in pots]  # datagen's limit samples: chips behind = 101 - pot / 2
    return NetLeaf(model, equity, pots, stacks, turn.kappa, device, boards_by_deal=boards, possible_by_deal=turn.valid)


class Recording:
    """A leaf that remembers the last values it gave each player."""

    def __init__(self, leaf):
        self.leaf, self.last, self.closing = leaf, {}, {}

    def values(self, p: int, t: int, closing):
        self.closing[p] = closing  # the ranges the leaf was asked about (onpolicy.py collects them)
        self.last[p] = self.leaf.values(p, t, closing)
        return self.last[p]


class LeafCFR(RangeCFR):
    """RangeCFR with a leaf, keeping each player's leaf values averaged over the iterations
    with the trunk's averaging weights (the opponent's constraint values of a safe re-solve,
    as DeepStack keeps them)."""

    def __init__(self, pt, leaf, variant: str = "dcfr", **kw):
        super().__init__(pt, variant, leaf=Recording(leaf), **kw)
        self.river_classes = pt.classes[pt.num_streets]
        self.leaf_cum, self.weight_sum = [None, None], [0.0, 0.0]

    def instant_regrets(self, p: int):
        out = super().instant_regrets(p)
        t = float(self.t)
        w = {"cfr": 1.0, "cfr+": t, "dcfr": t ** self.gamma}[self.variant]
        v = self.leaf.last[p]
        self.leaf_cum[p] = [w * x for x in v] if self.leaf_cum[p] is None else [a + w * x for a, x in
                                                                                zip(self.leaf_cum[p], v)]
        self.weight_sum[p] += w
        return out

    def cfv_average(self, c: int, q: int) -> np.ndarray:
        """Player q's averaged leaf values at river class c, (D_turn, L_c, H), solver units."""
        rc = self.river_classes[c]
        avg = [x / self.weight_sum[q] for x in self.leaf_cum[q]]
        out = np.empty((avg[0].shape[0], rc.num_lines, avg[0].shape[-1]))
        for pc in np.unique(rc.parent_class):
            m = np.flatnonzero(rc.parent_class == pc)
            out[:, m] = avg[pc][:, rc.parent_line[m], rc.parent_close[m]]
        return out


def leaf_error(upper, river, net, hsig, tol: float, iters: int, variant: str) -> dict:
    """The network against exact river solves at the turn-end ranges of `hsig` (module docstring)."""
    exact = ExactRiverLeaf(upper, river, tol, variant=variant, iters=iters, values="br")
    reach, closing = forward(upper, hsig, closing=True)
    out = {"leaf_error_weighted": 0.0, "leaf_error_all": 0.0, "exact_abs_weighted": 0.0, "exact_abs_all": 0.0}
    for p in (0, 1):
        a, b = net.values(p, 1, closing), exact.values(p, 1, closing)
        for c, (x, y) in enumerate(zip(a, b)):
            own = closing[c][p]
            out["leaf_error_all"] += float(np.abs(x - y).sum())
            out["leaf_error_weighted"] += float((own * np.abs(x - y)).sum())
            out["exact_abs_all"] += float(np.abs(y).sum())
            out["exact_abs_weighted"] += float((own * np.abs(y)).sum())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a saved ValueNet (best.pt)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--checks", default="50,100,200")
    ap.add_argument("--trunk", default="dcfr", choices=["cfr", "cfr+", "dcfr"])
    ap.add_argument("--resolve-tol", type=float, default=1e-3)
    ap.add_argument("--sub-iters", type=int, default=5000)
    ap.add_argument("--inner", default="dcfr", choices=["cfr+", "dcfr"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from bombpot.valuenet.data import flop_pair_hash
    from bombpot.valuenet.model import load_model

    checks = sorted({int(x) for x in a.checks.split(",")} | {a.iters})
    game = make_game("limit")
    full = holdem_public_tree(game)
    upper = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
    model = load_model(a.model, a.device)
    net = net_leaf(model, game, upper, river, a.device)
    geom = ExactRiverLeaf(upper, river, a.resolve_tol)  # the river's layout, for the re-solves
    cfg_path = Path(a.model).with_name("config.yaml")
    result = {"args": vars(a), "flops": FLOPS, "features": model.features.to_dict(), "model_info": model.info, "checks": {}}
    if cfg_path.exists():
        import yaml

        data = yaml.safe_load(cfg_path.read_text())["data"]
        h = float(flop_pair_hash(np.array(game.flops)[None], data["salt"], model.features.num_suits)[0])
        result["flop_pair_held_out"] = bool(h < data["holdout_fraction"]) and data.get("split", "flop_pair") == "flop_pair"
        result["data"] = data
        print(f"flop pair {FLOPS}: {'held out of' if result['flop_pair_held_out'] else 'IN'} the training set", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    solver = LeafCFR(upper, net, a.trunk)
    while solver.t < a.iters:
        solver.iteration()
        if solver.t in checks:
            t0 = time.perf_counter()
            flop_turn = solver.exploitability()
            avg = solver.hand_sigma(solver.average_strategy())
            exact = grade(full, upper, river, avg, a.resolve_tol, a.sub_iters, a.inner)
            deep = deepstack_grade(full, upper, solver, a.resolve_tol, a.sub_iters, a.inner, geom=geom)
            err = leaf_error(upper, river, net, avg, a.resolve_tol, a.sub_iters, a.inner)
            result["checks"][solver.t] = {"wall_time": solver.solve_seconds, "flop_turn": flop_turn, "exact": exact,
                                          "deepstack": deep, **err, "grade_seconds": time.perf_counter() - t0}
            print(f"net it {solver.t:4d} {solver.solve_seconds:7.1f}s flop+turn {flop_turn['exploitability']:.5f} "
                  f"exact {exact['exploitability']:.5f} deepstack {deep['exploitability']:.5f} | leaf error "
                  f"{err['leaf_error_weighted']:.5f} weighted, {err['leaf_error_all']:.4f} all hands "
                  f"(exact values {err['exact_abs_weighted']:.3f}, {err['exact_abs_all']:.2f})", flush=True)
            out.write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
