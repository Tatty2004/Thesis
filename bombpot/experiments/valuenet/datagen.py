"""Training data for the turn-end value network (PLAN_valuenet.md, section F): random river
subgames solved exactly by core/solvers/subgame.py, one HDF5 shard per job.

A sample: a random flop pair (two disjoint 3-card flops), random turn cards (one per board),
the river's betting (below), and random ranges for both
players over the hands possible at that turn deal (DeepStack's recursive random splitting,
plus some near-uniform and some very peaked ranges), normalized to sum 1. The river subgame of
that turn deal is solved from those ranges to `tol` pots with chance weights of the river
alone (each valid river deal of a hand pair weighs 1 / (rest x (rest - 1))), so its values are
"values given normalized ranges". The format is in experiments/valuenet/README.md.

River betting:
- `--template pot` (the real game: no-limit, 1/2-pot and pot sizes, 2 aggressive actions per
  street, 100 behind after an ante of 1): a river line drawn uniformly from the real game's
  river lines that still have betting (pt.classes[2]), its pot P_l = 2 x contrib, then
  P = P_l x exp(U(-ln s, ln s)) (s = `--spread`, default 2; 1 keeps the real pots), capped so
  each player keeps more than 0 behind. The river template for that pot and stack is walked
  from the engine (core.public._walk on a river state with both contributions set to P/2).
- `--template limit` (the limit baseline): pot log-uniform on [2, 200], stacks 101 - P/2,
  the limit river (bets of 4 chips), the pot only setting the chips already in.

    python -m bombpot.experiments.valuenet.datagen --n 100 --seed 0 --out results/valuenet/subgame/shard_0.h5
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from bombpot.core.game import CHANCE
from bombpot.core.limit import CALL
from bombpot.core.public import Template, _class_template, _walk
from bombpot.core.solvers.fused import solve_subgame_fused
from bombpot.core.solvers.subgame import line_template, solve_subgame
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

STACK = 101.0  # chips each player has, ante included (PLAN_valuenet.md: the game)
POT_RANGE = (2.0, 200.0)

UNITS = (
    "Units. range0/range1 (N, H): each player's reach over the hand list hands[hand_set[n]], zero on hands "
    "that share a card with the boards (flops and turn cards; possible[n] marks the others, which can also "
    "have range 0), normalized to sum 1. The river subgame is solved with chance weight kappa = river.scale "
    "= 1 / (rest (rest - 1)) per valid river deal (rest = cards left after both boards and both hands), the "
    "river chance only, so for any hand pair its valid river deals' weights sum to 1. cfv0/cfv1 (N, H) are "
    "the root counterfactual values of the subgame's average strategy profile (one profile for both "
    "players), in pots: cfv_p[i] = sum over opponent hands j compatible with i (no shared card) of "
    "range_o[j] E_river[payoff of p | i, j] / pot, NOT renormalized per hand over the compatible j. Payoffs "
    "count all chips (the pot/2 each has in included), so zero_sum_residual = range0 . cfv0 + range1 . cfv1 "
    "= 0 up to rounding. br0/br1 (N, H): the same with player p best-responding per hand to the opponent's "
    "average strategy (br_p >= cfv_p; range0 . br0 + range1 . br1 = 2 x subgame_exploitability x mass). "
    "subgame_exploitability = (br0 . range0 + br1 . range1) / 2 / mass / pot. stack = "
    "chips behind for each player at the start of the river. equity (N, H, 2): each hand's mean showdown "
    "share per board over the river deals it can see, against a uniform opponent range. Inverse transform "
    "(a turn-end leaf at turn deal d, closing line with pot P, unnormalized closing reaches R0, R1): agg_p[i] "
    "= cfv_p[i] x P x sum over possible j of R_o[j] x turn.kappa[d], with turn.kappa[d] the turn street's "
    "chance weight (holdem_public_tree(game, upto=2).streets[1].kappa[d]) and no other factor: the real "
    "river's weight per deal is turn.kappa[d] x river.scale, and river.scale is already inside cfv. This is "
    "the range solver's units (chance weight x opponent reach x payoff)."
)


# Sampling -------------------------------------------------------------------------------------


def random_range(rng, valid: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """A random range over the hands where `valid`, summing to 1 (zeros elsewhere).

    80%: recursive random splitting (DeepStack): the hands, in order of strength (half the
    time) or in random order, are split at a random point and each part gets a random share of
    the mass, recursively. 10%: near-uniform. 10%: very peaked (a few hands)."""
    idx = np.flatnonzero(valid)
    n = len(idx)
    out = np.zeros(len(valid))
    u = rng.random()
    if u < 0.1:
        x = 1.0 + 0.1 * rng.random(n)
    elif u < 0.2:
        k = int(rng.integers(1, max(2, n // 20) + 1))
        x = np.zeros(n)
        x[rng.choice(n, size=k, replace=False)] = rng.random(k) + 1e-3
    else:
        order = np.argsort(strength[idx], kind="stable") if rng.random() < 0.5 else rng.permutation(n)
        x = np.zeros(n)
        stack = [(0, n, 1.0)]
        while stack:
            lo, hi, m = stack.pop()
            if hi - lo == 1:
                x[order[lo]] = m
                continue
            cut = int(rng.integers(lo + 1, hi))
            p = rng.random()
            stack.append((lo, cut, m * p))
            stack.append((cut, hi, m * (1.0 - p)))
    out[idx] = x
    return out / out.sum()


def sample_spot(rng, num_ranks: int, num_suits: int) -> dict:
    """Random flop pair and turn cards."""
    cards = rng.permutation(num_ranks * num_suits)
    flops = (tuple(sorted(int(c) for c in cards[:3])), tuple(sorted(int(c) for c in cards[3:6])))
    return {"flops": flops, "turn": (int(cards[6]), int(cards[7]))}


def make_game(flops, num_ranks: int, num_suits: int, template: str) -> Holdem:
    if template == "limit":
        return Holdem(num_ranks=num_ranks, num_suits=num_suits, board_cards=(3, 1, 1), flops=flops, bet_sizes=(2, 4, 4))
    if template == "pot":  # the real game (PLAN_valuenet.md)
        return Holdem(num_ranks=num_ranks, num_suits=num_suits, board_cards=(3, 1, 1), flops=flops, betting="pot",
                      stack=STACK - 1.0, sizes=(0.5, 1.0), max_aggressive=2)
    raise ValueError(f"unknown template {template!r}")


def river_start(game):
    """A state where the river's betting begins (the flop and turn checked down)."""
    s = game.initial_state()
    while True:
        if game.current_player(s) == CHANCE:
            s = game.next_state(s, game.chance_outcomes(s)[0][0])
        elif game.street(s) == 2:
            return s
        else:
            s = game.next_state(s, CALL)


def river_template(game, contrib: float) -> Template:
    """The engine's river betting with `contrib` chips in from each player (one line, chips as
    (1,) arrays); no betting (an all-in class's template) once nothing is behind."""
    if contrib >= game.total:
        return Template([], [], [], [], [("", np.zeros(1))])
    s = river_start(game)._replace(contrib=(float(contrib), float(contrib)))
    _, tp, _ = _walk(game, s)
    return _class_template([tp])


class FlopPair:
    """The pieces built once per flop pair: game, flop+turn tree (for the turn deals, the
    hands and the river's line classes), river provider."""

    def __init__(self, flops, num_ranks: int, num_suits: int, template: str = "limit", limit_pots: str = "loguniform"):
        self.template, self.limit_pots = template, limit_pots
        self.game = make_game(flops, num_ranks, num_suits, template)
        self.upper = holdem_public_tree(self.game, upto=2)
        self.river = HoldemRiver(self.game, self.upper)
        self.hands = np.array(self.upper.hands, dtype=np.int64)
        # the real river lines that still have betting: (class, line within it)
        self.lines = [(c, l) for c, lc in enumerate(self.upper.classes[2]) if lc.template.num_nodes > 0
                      for l in range(lc.num_lines)]

    def sample_betting(self, rng, spread: float = 2.0) -> dict:
        """The river's betting for one sample: template, pot P (both players in for P/2),
        stack (chips behind each), the real line it came from (street index, -1 for limit) and
        the spread factor applied to that line's pot."""
        if self.template == "limit" and self.limit_pots == "lines":
            # the limit game's own river pots (a real line's, spread like the pot template's);
            # the limit river's betting is the same on every line
            lc = self.upper.classes[2][0]
            l = int(rng.integers(lc.num_lines))
            factor = float(np.exp(rng.uniform(-np.log(spread), np.log(spread)))) if spread > 1 else 1.0
            pot = max(2 * float(lc.contrib[l]) * factor, 2 * self.game.ante)
            return {"template": line_template(lc.template, 0), "pot": pot, "stack": STACK - pot / 2,
                    "line": int(lc.lines[l]), "pot_spread": pot / (2 * float(lc.contrib[l]))}
        if self.template == "limit":
            pot = float(np.exp(rng.uniform(np.log(POT_RANGE[0]), np.log(POT_RANGE[1]))))
            return {"template": line_template(self.upper.classes[2][0].template, 0), "pot": pot,
                    "stack": STACK - pot / 2, "line": -1, "pot_spread": 1.0}
        c, l = self.lines[int(rng.integers(len(self.lines)))]
        lc = self.upper.classes[2][c]
        factor = float(np.exp(rng.uniform(-np.log(spread), np.log(spread)))) if spread > 1 else 1.0
        contrib = min(float(lc.contrib[l]) * factor, self.game.total - 0.5)  # keep something behind
        contrib = max(contrib, self.game.ante)
        tp = line_template(lc.template, l) if factor == 1.0 else river_template(self.game, contrib)
        return {"template": tp, "pot": 2 * contrib, "stack": self.game.total - contrib,
                "line": int(lc.lines[l]), "pot_spread": contrib / float(lc.contrib[l])}

    def turn_deal(self, turn) -> int:
        hit = np.flatnonzero((self.river.turn_cards == np.array(turn)).all(axis=1))
        assert len(hit) == 1
        return int(hit[0])

    def river_street(self, d: int):
        """The river CardStreet of turn deal d, with chance weights of the river only."""
        st = self.river.street(*self.river.all_children([d]))
        st.kappa = np.full(st.num_deals, self.river.scale)
        return st


def turn_equity(st) -> np.ndarray:
    """(H, 2): each hand's mean showdown share per board over the river deals it can see."""
    eq = st.equity()  # (D, H, B), 0 where the hand can't be held
    seen = st.valid.sum(axis=0)
    return np.divide(eq.sum(axis=0), seen[:, None], out=np.zeros(eq.shape[1:]), where=seen[:, None] > 0)


def solve_sample(fp: FlopPair, spot: dict, rng, tol: float, iters: int, variant: str = "cfr+",
                 solver: str = "fused", dtype: str = "float64", device: str = "cuda", fuse: bool = False) -> dict:
    """One sample: ranges, equity and the solved subgame (values and exploitability in pots).
    spot holds turn, template, pot and stack (sample_spot, then FlopPair.sample_betting).
    solver "fused" (core/solvers/fused.py: one compiled kernel per deal; in float64 the same
    numbers as "subgame", bit for bit) or "subgame" (core/solvers/subgame.py, float64 only)."""
    d = fp.turn_deal(spot["turn"])
    valid = fp.upper.streets[1].valid[d]
    st = fp.river_street(d)
    equity = turn_equity(st)
    strength = equity.mean(axis=1)
    r0, r1 = random_range(rng, valid, strength), random_range(rng, valid, strength)
    pot = spot["pot"]
    if solver == "fused":
        sub = solve_subgame_fused(st, spot["template"], pot / 2, r0, r1, variant, iters, tol * pot, dtype=dtype,
                                  keep_strategy=False)
    elif solver == "gpu":  # the torch solver on a GPU (valuenet/gpu_subgame.py); torch is imported here only
        import torch

        from bombpot.valuenet.gpu_subgame import solve_subgame_gpu

        sub = solve_subgame_gpu(st, spot["template"], pot / 2, r0, r1, variant, iters, tol * pot,
                                dtype=getattr(torch, dtype), device=device, fuse=fuse)
    elif solver == "subgame" and dtype == "float64":
        sub = solve_subgame(st, spot["template"], pot / 2, r0, r1, variant, iters, tol * pot, stack=spot["stack"])
    else:
        raise ValueError(f"solver {solver!r} with dtype {dtype!r}")
    cfv0, cfv1 = sub.values[0] / pot, sub.values[1] / pot  # both from the one average-strategy profile
    # each hand's best-response value against the opponent's average strategy: what a leaf must
    # hand out (ExactRiverLeaf's "br"), right for a hand with little or no range too
    br0, br1 = sub.br_values[0] / pot, sub.br_values[1] / pot
    return {"range0": r0, "range1": r1, "possible": valid, "equity": equity, "cfv0": cfv0, "cfv1": cfv1,
            "br0": br0, "br1": br1,
            "zero_sum_residual": float(r0 @ cfv0 + r1 @ cfv1),
            "subgame_exploitability": sub.exploitability / pot, "iters": sub.iterations, "seconds": sub.seconds,
            "mass": sub.mass}


# Shards ------------------------------------------------------------------------------------


def git_hash() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                           check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "bombpot", "pyproject.toml"], cwd=root,
                               capture_output=True, text=True).stdout.strip()
        return h + (" dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Shard:
    """An HDF5 shard, grown one sample at a time (so a killed job keeps what it solved)."""

    PER_SAMPLE = {"flops": ((2, 3), "i1"), "turn": ((2,), "i1"), "hand_set": ((), "i4"), "pot": ((), "f4"),
                  "stack": ((), "f4"), "subgame_exploitability": ((), "f4"), "iters": ((), "i4"),
                  "seconds": ((), "f4"), "mass": ((), "f4"), "zero_sum_residual": ((), "f4"), "line": ((), "i4"),
                  "pot_spread": ((), "f4")}
    PER_HAND = {"range0": ((), "f4"), "range1": ((), "f4"), "cfv0": ((), "f4"), "cfv1": ((), "f4"),
                "br0": ((), "f4"), "br1": ((), "f4"),
                "equity": ((2,), "f4"), "possible": ((), "bool")}

    def __init__(self, path, H: int, attrs: dict):
        try:
            import h5py
        except ImportError as e:  # h5py lives in the torch environment (CLAUDE.md, Stack)
            raise ImportError("datagen shards need h5py: use /n/fs/scratch/rb5709/envs/torch/bin/python") from e
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, "w")
        self.n, self.H = 0, H
        for k, v in attrs.items():
            self.f.attrs[k] = v
        for name, (shape, dt) in self.PER_SAMPLE.items():
            self.f.create_dataset(name, (0,) + shape, dt, maxshape=(None,) + shape, chunks=(256,) + shape)
        for name, (shape, dt) in self.PER_HAND.items():
            self.f.create_dataset(name, (0, H) + shape, dt, maxshape=(None, H) + shape, chunks=(64, H) + shape,
                                  compression="gzip", compression_opts=4)
        self.f.create_dataset("hands", (0, H, 2), "i1", maxshape=(None, H, 2), chunks=(16, H, 2))
        self.hand_sets = 0

    def add_hands(self, hands: np.ndarray) -> int:
        ds = self.f["hands"]
        ds.resize(self.hand_sets + 1, axis=0)
        ds[self.hand_sets] = hands
        self.hand_sets += 1
        return self.hand_sets - 1

    def add(self, row: dict) -> None:
        for name in list(self.PER_SAMPLE) + list(self.PER_HAND):
            ds = self.f[name]
            ds.resize(self.n + 1, axis=0)
            ds[self.n] = row[name]
        self.n += 1
        self.f.flush()

    def close(self) -> None:
        self.f.attrs["n"] = self.n
        self.f.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, required=True, help="samples")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="the shard (.h5)")
    ap.add_argument("--tol", type=float, default=1e-3, help="subgame exploitability target, in pots")
    ap.add_argument("--iters", type=int, default=5000, help="iteration cap per subgame")
    ap.add_argument("--template", default="pot", choices=["pot", "limit"],
                    help="river betting: the real no-limit game's lines (pot) or the limit baseline")
    ap.add_argument("--spread", type=float, default=2.0, help="pot spread factor around the real lines (pot)")
    ap.add_argument("--limit-pots", default="loguniform", choices=["loguniform", "lines"],
                    help="limit template: pot log-uniform on [2, 200], or a real river line's pot x the spread")
    ap.add_argument("--variant", default="dcfr", choices=["cfr+", "dcfr"],
                    help="subgame solver (DCFR: 2.9x faster than CFR+ to 1e-3 pots on the real deck)")
    ap.add_argument("--device", default="cuda", help="solver gpu: the torch device")
    ap.add_argument("--solver", default="fused", choices=["fused", "subgame", "gpu"],
                    help="fused: one compiled kernel per river deal (core/solvers/fused.py), the same numbers "
                         "as subgame in float64; subgame: the batched range-solver passes; gpu: the same "
                         "algorithm in torch on --device (valuenet/gpu_subgame.py; use --dtype float32)")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                    help="solver precision (float32: fused and gpu; values agree to rounding, not bit for bit)")
    ap.add_argument("--fuse", action="store_true", help="solver gpu: compiled updates (about 3x faster, same numbers "
                                                        "to rounding)")
    ap.add_argument("--threads", type=int, default=None, help="numba threads (default: NUMBA_NUM_THREADS)")
    ap.add_argument("--deck", default="13,4", help="ranks,suits (default the real deck)")
    ap.add_argument("--per-flop", type=int, default=1, help="samples per flop pair (each with its own turn)")
    ap.add_argument("--flops", default=None,
                    help="one fixed flop pair for every sample, e.g. 2c3c4d,5c6d4c (default: random flop pairs)")
    ap.add_argument("--log-every", type=int, default=1)
    a = ap.parse_args(argv)
    num_ranks, num_suits = (int(x) for x in a.deck.split(","))
    if a.threads is not None:
        import numba

        numba.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    betting = ({"bet_sizes": "2,4,4 (limit; river bet 4)", "pot_range": "log-uniform on [2, 200], stack = 101 - pot/2"}
               if a.template == "limit" else
               {"bet_sizes": "no-limit, sizes 0.5 and 1 pot, 2 aggressive actions per street, all-in when a size "
                             "reaches the stack", "pot_range": f"a real river line with betting, uniform, pot x "
                             f"exp(U(-ln {a.spread:g}, ln {a.spread:g})), stack = 101 - pot/2"})
    attrs = {"num_ranks": num_ranks, "num_suits": num_suits, "board_cards": "3,1,1", "num_boards": 2,
             "ante": 1.0, "total_chips": STACK, "template": a.template, "spread": a.spread,
             "limit_pots": a.limit_pots, **betting,
             "tol_pots": a.tol, "iters_cap": a.iters,
             "variant": a.variant, "solver": a.solver, "solver_dtype": a.dtype, "solver_fuse": a.fuse, "seed": a.seed, "flops_fixed": a.flops or "", "git": git_hash(), "per_flop": a.per_flop,
             "card_encoding": "card = rank * num_suits + suit, ranks 23456789TJQKA, suits cdhs", "units": UNITS}
    fixed = None
    if a.flops:
        ranks, suits = "23456789TJQKA"[:num_ranks] if num_ranks < 13 else "23456789TJQKA", "cdhs"
        fixed = tuple(tuple(sorted(ranks.index(f[i]) * num_suits + suits.index(f[i + 1]) for i in (0, 2, 4)))
                      for f in a.flops.split(","))
    shard = None
    start = time.perf_counter()
    fp = None
    for n in range(a.n):
        spot = sample_spot(rng, num_ranks, num_suits)
        if fixed is not None:
            rest = rng.permutation([c for c in range(num_ranks * num_suits) if c not in {x for f in fixed for x in f}])
            spot = {"flops": fixed, "turn": (int(rest[0]), int(rest[1]))}
        if (n % a.per_flop == 0 and fixed is None) or fp is None:
            fp = FlopPair(spot["flops"], num_ranks, num_suits, a.template, a.limit_pots)
            if shard is None:
                shard = Shard(a.out, len(fp.hands), attrs)
            hand_set = shard.add_hands(fp.hands)
        elif fixed is None:  # keep the flop pair, draw new turn cards from what's left
            used = {c for f in fp.game.flops for c in f}
            rest = rng.permutation([c for c in range(num_ranks * num_suits) if c not in used])
            spot["flops"], spot["turn"] = fp.game.flops, (int(rest[0]), int(rest[1]))
        spot.update(fp.sample_betting(rng, a.spread))
        t0 = time.perf_counter()
        row = solve_sample(fp, spot, rng, a.tol, a.iters, a.variant, a.solver, a.dtype, a.device, a.fuse)
        row.update(flops=np.array(spot["flops"]), turn=np.array(spot["turn"]), hand_set=hand_set, pot=spot["pot"],
                   stack=spot["stack"], line=spot["line"], pot_spread=spot["pot_spread"],
                   seconds=time.perf_counter() - t0)
        shard.add(row)
        if (n + 1) % a.log_every == 0 or n + 1 == a.n:
            el = time.perf_counter() - start
            print(f"sample {n + 1}/{a.n}  pot {spot['pot']:7.2f}  nodes {spot['template'].num_nodes:2d}  iters {row['iters']:5d}  expl "
                  f"{row['subgame_exploitability']:.2e} pots  solve {row['seconds']:6.1f}s  elapsed {el:7.0f}s  "
                  f"({el / (n + 1):.1f}s/sample)", flush=True)
    if shard is not None:
        shard.close()


if __name__ == "__main__":
    main()
