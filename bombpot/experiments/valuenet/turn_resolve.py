"""Safe turn re-solving from a real_trunk solve (the turn step of real-time play): given the
trunk's average strategy, a flop line that closes the flop and one turn deal, re-solve that
turn deal's turn betting for one player with the gadget (core/solvers/leaf_resolve.py), the
value network (NetLeaf) at every turn-end state exactly as the trunk values them, and print
the re-solved turn strategy.

    python -m bombpot.experiments.valuenet.turn_resolve --trunk results/valuenet/real/<name> \
        --flop-line r0.5c --turn 5h2c [--player 0] [--iters 300] [--cfv auto|avg|backup|onpolicy|br] [--out spot.npz]

Inputs from the trunk directory (real_trunk's outputs):
  strategy.npz  the average strategy; the flop's gives both players' reach at the turn root
                (the same at every turn deal), and with --cfv br the turn's (one slice, read
                without loading the file) gives the opponent's T values
  turncfv.npy   (real_trunk --turn-cfv) both players' turn-root values averaged over the trunk's
                iterations with its weights, (2, turn deals, turn lines, hands), solver units:
                the T values of --cfv avg (the river gadget's construction, real_grade's
                leafcfv.npy one street up)
  trunk.json    the game, the network and --amp
--cfv br: the opponent's best-response value at the turn root against the trunk's average turn
strategy, the network valuing the turn's end (one leaf call per player; what the safety bound
of resolve.py is stated for). Without turncfv.npy (trunks run without --turn-cfv), two cheap
stand-ins for avg from what every trunk stores:
  backup    the trunk's averaged turn-end values (leafcfv.npy, or viewer_pack's leafcfv32.npy)
            backed up through its average turn strategy (no network call): avg's numbers except
            that avg averages each iteration's own turn strategy with its own leaf values
  onpolicy  the values of the average turn strategy, the network valuing the turn's end at the
            average's closing reach (one network call)
auto: avg when turncfv.npy exists, else backup when leafcfv does, else onpolicy.

Units are the trunk's throughout (the turn street's chance weights, unnormalized reaches), so
the T values and the re-solve's values compare directly; `exploitability` is the gadget's
NashConv / mass, chips per unit of reach, and also given in pots of the turn root.

--flop-line is the flop's closing betting string ('cc', 'r0.5c', 'cr1r0.5c', ...); --turn is
board A's then board B's turn card ('5h2c'). Player 0 acts first, so the strategy at the turn
root is player 0's; --player 1 re-solves for player 1 (its strategy at its turn nodes).
Needs torch (the scratch environment).
"""
from __future__ import annotations

import argparse
import json
import struct
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bombpot.core.solvers.leaf_resolve import LeafGadget
from bombpot.core.solvers.range_cfr import class_reach, class_values, stack_closing
from bombpot.core.solvers.subgame import line_template
from bombpot.experiments.valuenet import datagen as dg
from bombpot.experiments.valuenet import real_grade as rg

RANKS, SUITS = "23456789TJQKA", "cdhs"


# The turn's layout -------------------------------------------------------------------------


def turn_lines(upper) -> list:
    """Every turn line as (class, line), in class then line order: the L axis of turncfv.npy."""
    return [(c, l) for c, lc in enumerate(upper.classes[1]) for l in range(lc.num_lines)]


def restrict(lc, lines):
    """A turn class restricted to some of its lines (template chips and contrib)."""
    lines = np.atleast_1d(np.asarray(lines, dtype=np.int64))
    return SimpleNamespace(template=line_template(lc.template, lines), contrib=np.asarray(lc.contrib, float)[lines],
                           lines=lines)


def turn_root_reach(upper, flop_sigma) -> list:
    """[c] (2, L_c, H): both players' reach at every turn root under the flop strategy
    flop_sigma[c][k] (1, 1, H, A) (the same at every turn deal; forward's next_root)."""
    H = len(upper.hands)
    closing = [class_reach(lc.template, s, (np.ones((1, 1, H)), np.ones((1, 1, H))))[1]
               for lc, s in zip(upper.classes[0], flop_sigma)]
    out = []
    for lc in upper.classes[1]:
        x = np.empty((2, lc.num_lines, H))
        for q in (0, 1):
            for i, (pc, pl, pj) in enumerate(zip(lc.parent_class, lc.parent_line, lc.parent_close)):
                x[q, i] = closing[pc][pj][q][0, pl]
        out.append(x)
    return out


def cards_of(s: str, num_suits: int) -> list:
    return [RANKS.index(s[i]) * num_suits + SUITS.index(s[i + 1]) for i in range(0, len(s), 2)]


def card_name(c: int, num_suits: int) -> str:
    return RANKS[c // num_suits] + SUITS[c % num_suits]


def find_line(upper, flop_line: str) -> tuple:
    """(turn class, line) of the turn root after a flop closing string."""
    hist = upper.histories(1)
    for c, row in enumerate(hist):
        for l, h in enumerate(row):
            if h == (flop_line,):
                return c, l
    raise SystemExit(f"no flop line {flop_line!r}; the flop's closing lines are "
                     f"{[h[0] for row in hist for h in row]}")


def find_deal(river, turn) -> int:
    hit = np.flatnonzero((river.turn_cards == np.asarray(turn)[None]).all(axis=1))
    if len(hit) != 1:
        raise SystemExit(f"turn cards {turn} are not a turn deal of this flop pair")
    return int(hit[0])


# Reading the trunk --------------------------------------------------------------------------


def npz_member(path, key):
    """One array of an .npz as a read-only memmap when it is stored uncompressed (np.savez), so
    a slice reads only its bytes; otherwise the array itself."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        with np.load(path) as f:
            return f[key]
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        head = f.read(30)
        n, m = struct.unpack("<HH", head[26:30])
        f.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(f)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def flop_strategy(path, upper) -> list:
    """The stored flop strategy [c][k] (1, 1, H, A), as real_grade.load_strategy rounds it."""
    with np.load(path) as f:
        raw = [[f[f"t0_c{c}_k{k}"] for k in range(len(lc.template.actions))] for c, lc in enumerate(upper.classes[0])]
    return rg.rounded([raw])[0]


def turn_strategy_slice(path, upper, d: int, c: int, lines) -> list:
    """The stored turn strategy at deal d for lines of class c: [k] (1, len(lines), H, A)."""
    lc = upper.classes[1][c]
    raw = [np.asarray(npz_member(path, f"t1_c{c}_k{k}")[d][np.asarray(lines)])[None] for k in range(len(lc.template.actions))]
    return rg.rounded([[raw]])[0][0]


def deal_leaf(model, equity_d, pot, kappa_d: float, board_d, valid_d, hands, device="cpu", amp: bool = False):
    """NetLeaf for one turn deal (a street of one deal and one turn class's chosen lines):
    equity_d (H, 2), pot (L, closes) at those lines' closes, board_d (2, 4) card ids, valid_d
    (H,). Its values are the trunk's NetLeaf's at that deal and those lines."""
    from bombpot.valuenet.leaf import NetLeaf

    pot = np.asarray(pot, dtype=float)
    return NetLeaf(model, np.asarray(equity_d, dtype=np.float64)[None], [pot], [dg.STACK - pot / 2],
                   np.array([float(kappa_d)]), device, boards_by_deal=np.asarray(board_d)[None],
                   possible_by_deal=np.asarray(valid_d, bool)[None], hands=hands, amp=amp)


# Values under the blueprint ------------------------------------------------------------------


def blueprint_values(street, classes, sigma, roots, leaf) -> dict:
    """Both players' on-policy and best-response values at the roots of `classes` under the
    strategy sigma[c][k] from root reaches roots[c] = (reach0, reach1), the leaf valuing the
    closes (one call per player): {"values": [q][c], "br": [q][c]} each (D, L_c, H)."""
    nodes, closing = [], []
    for lc, s, r in zip(classes, sigma, roots):
        n, cl = class_reach(lc.template, s, r)
        nodes.append(n)
        closing.append(cl)
    out = {"values": [None, None], "br": [None, None]}
    for q in (0, 1):
        agg = leaf.values(q, 1, stack_closing(closing))
        for mode, key in (("cfr", "values"), ("br", "br")):
            out[key][q] = [class_values(lc.template, street, lc.contrib, sigma[c], nodes[c], q, mode, False, agg[c])[0][0]
                           for c, lc in enumerate(classes)]
    return out


def backup_values(street, classes, sigma, roots, agg) -> list:
    """[q][c] (D, L_c, H): both players' on-policy values at the roots of `classes` under the
    strategy sigma[c][k] from root reaches roots[c], each close taking a stored value instead of
    a leaf call: agg[q][c] (D, L_c, closes_c, H), solver units (the trunk's averaged leaf values,
    real_trunk's leafcfv.npy). The T values of --cfv backup: the trunk's averaged turn-end values
    backed up through its average turn strategy (no network call)."""
    out = [None, None]
    nodes = [class_reach(lc.template, s, r)[0] for lc, s, r in zip(classes, sigma, roots)]
    for q in (0, 1):
        out[q] = [class_values(lc.template, street, lc.contrib, sigma[c], nodes[c], q, "cfr", False, agg[q][c])[0][0]
                  for c, lc in enumerate(classes)]
    return out


def close_index(upper) -> dict:
    """(turn class, turn line, close) -> index among all river lines (real_grade.river_lines)."""
    out, i = {}, 0
    for rc in upper.classes[upper.num_streets]:
        for l in range(rc.num_lines):
            out[(int(rc.parent_class[l]), int(rc.parent_line[l]), int(rc.parent_close[l]))] = i
            i += 1
    return out


# One real-deck spot ---------------------------------------------------------------------------


class Spot:
    """A trunk directory loaded for turn re-solves: the game, the network leaf's pieces, the
    flop strategy and the T values' source."""

    def __init__(self, trunk, device="cuda", model_path=None, amp=None):
        import torch

        from bombpot.holdem.public import HoldemRiver, holdem_public_tree
        from bombpot.valuenet.model import load_model

        t0 = time.perf_counter()
        self.trunk = Path(trunk)
        self.info = json.loads((self.trunk / "trunk.json").read_text())
        args = self.info["args"]
        self.nr, self.ns = (int(x) for x in args["deck"].split(","))
        flops = tuple(tuple(sorted(cards_of(f, self.ns))) for f in args["flops"].split(","))
        self.game = dg.make_game(flops, self.nr, self.ns, "pot")
        self.upper = holdem_public_tree(self.game, upto=2)
        self.river = HoldemRiver(self.game, self.upper)
        self.turn = self.upper.streets[1]
        self.D, self.H = self.turn.valid.shape
        from bombpot.experiments.valuenet.gate_f import turn_pots

        self.pots = turn_pots(self.upper)
        self.amp = bool(args.get("amp", False)) if amp is None else amp
        self.device = device
        self.model = load_model(model_path or args["model"], device)
        name = f"equity_{args['flops'].replace(',', '_')}_{self.nr}x{self.ns}.npy"
        eq = next((x for x in (self.trunk / name, self.trunk.parent / name) if x.exists()), None)
        self.equity = np.load(eq, mmap_mode="r") if eq is not None else None
        self.boards = np.concatenate([np.broadcast_to(np.array(self.game.flops), (self.D, 2, 3)),
                                      np.asarray(self.river.turn_cards)[:, :, None]], axis=-1)
        self.hands = np.array(self.upper.hands, dtype=np.int64)
        self.flop_sigma = flop_strategy(self.trunk / "strategy.npz", self.upper)
        self.roots = turn_root_reach(self.upper, self.flop_sigma)
        cfv = self.trunk / "turncfv.npy"
        self.turncfv = np.load(cfv, mmap_mode="r") if cfv.exists() else None
        leaf = next((x for x in (self.trunk / "leafcfv32.npy", self.trunk / "leafcfv.npy") if x.exists()), None)
        self.leafcfv = np.load(leaf, mmap_mode="r") if leaf is not None else None  # (2, D, river lines, H)
        self.close_of = close_index(self.upper)
        self.lines = turn_lines(self.upper)
        self.torch = torch
        self.setup_seconds = time.perf_counter() - t0

    def deal_equity(self, d: int) -> np.ndarray:
        if self.equity is not None:
            return np.asarray(self.equity[d], dtype=np.float64)
        from bombpot.experiments.valuenet.real_trunk import river_street

        return dg.turn_equity(river_street(self.river, d))

    def leaf(self, d: int, c: int, lines):
        """NetLeaf for turn deal d and lines of turn class c: one row per close and line."""
        return deal_leaf(self.model, self.deal_equity(d), self.pots[c][np.atleast_1d(lines)], self.turn.kappa[d],
                         self.boards[d], self.turn.valid[d], self.hands, self.device, self.amp)

    def setup(self, d: int, c: int, l: int, cfv_source: str = "auto") -> dict:
        """The pieces of the re-solve at (deal d, turn class c, line l): street, class, roots,
        leaf, both players' T values (and the blueprint's values when computed)."""
        street = self.turn.subset([d])
        cls = restrict(self.upper.classes[1][c], [l])
        r = self.roots[c][:, l][:, None, None]  # (2, 1, 1, H)
        leaf = self.leaf(d, c, [l])
        src = self.cfv_source(cfv_source)
        out = {"street": street, "cls": cls, "reach": r, "leaf": leaf, "cfv_source": src, "deal": d, "class": c,
               "line": l, "pot": 2.0 * float(cls.contrib[0])}
        if src not in ("avg", "backup", "onpolicy", "br"):
            raise ValueError(f"unknown T-value source {src!r}")
        if src == "avg":
            if self.turncfv is None:
                raise SystemExit(f"{self.trunk}/turncfv.npy is missing (run real_trunk with --turn-cfv), use --cfv br")
            i = self.lines.index((c, l))
            out["cfv"] = np.asarray(self.turncfv[:, d, i], dtype=np.float64)[:, None, None]  # (2, 1, 1, H)
        elif src == "backup":
            if self.leafcfv is None:
                raise SystemExit(f"{self.trunk} has no leafcfv.npy (or leafcfv32.npy) for --cfv backup")
            sig = turn_strategy_slice(self.trunk / "strategy.npz", self.upper, d, c, [l])
            closes = len(cls.template.closes)
            idx = [self.close_of[(c, l, j)] for j in range(closes)]
            agg = [[np.asarray(self.leafcfv[q, d, idx], dtype=np.float64)[None, None]] for q in (0, 1)]
            bv = backup_values(street, [cls], [sig], [(r[0], r[1])], agg)
            out["cfv"] = np.stack([bv[q][0] for q in (0, 1)])
        else:
            sig = turn_strategy_slice(self.trunk / "strategy.npz", self.upper, d, c, [l])
            bp = blueprint_values(street, [cls], [sig], [(r[0], r[1])], leaf)
            out["blueprint"] = bp
            out["cfv"] = np.stack([bp["br" if src == "br" else "values"][q][0] for q in (0, 1)])
        return out

    def cfv_source(self, src: str = "auto") -> str:
        """auto: avg when turncfv.npy exists, else backup when the averaged leaf values do,
        else onpolicy (needs only strategy.npz and the network)."""
        if src != "auto":
            return src
        return "avg" if self.turncfv is not None else "backup" if self.leafcfv is not None else "onpolicy"

    def resolve(self, s: dict, player: int, iters: int, tol=None, check_every: int = 10,
                simultaneous: bool = False, keep_best: bool = False, progress=None) -> tuple:
        """simultaneous: both players updated from one profile, one network pass per iteration
        (half the calls of alternating updates; LeafGadget). keep_best, progress: LeafGadget.solve's."""
        o = 1 - player
        g = LeafGadget(s["street"], [s["cls"]], [s["reach"][player]], [s["cfv"][o]], player, s["leaf"],
                       simultaneous=simultaneous)
        ev = g.solve(iters, tol, check_every, keep_best=keep_best, progress=progress)
        return g, ev


def strategy_table(spot: Spot, s: dict, g: LeafGadget, player: int, node: int = 0) -> tuple:
    """(actions, rows): player's re-solved strategy at turn node `node` per hand it can hold
    at this deal, with its reach, sorted by reach."""
    tp = s["cls"].template
    avg = g.average_strategy()[0][node][0, 0]  # (H, A)
    reach = s["reach"][player][0, 0]
    valid = spot.turn.valid[s["deal"]]
    rows = []
    for h in np.flatnonzero(valid):
        a, b = spot.hands[h]
        rows.append((card_name(a, spot.ns) + card_name(b, spot.ns), float(reach[h]), avg[h]))
    rows.sort(key=lambda r: -r[1])
    return tp.actions[node], rows


def report(spot: Spot, s: dict, g: LeafGadget, ev: dict, player: int) -> dict:
    o = 1 - player
    mass = float(ev["mass"][0][0, 0])
    gap = float(ev["gap"][0][0, 0])
    rp = s["reach"][player][0, 0]
    out = {"deal": s["deal"], "turn": [card_name(int(x), spot.ns) for x in spot.river.turn_cards[s["deal"]]],
           "flop_line": spot.upper.histories(1)[s["class"]][s["line"]][0], "player": player, "iterations": g.t,
           "cfv_source": s["cfv_source"], "pot": s["pot"], "gap": gap, "mass": mass,
           "exploitability_chips": gap / mass if mass > 0 else 0.0,
           "exploitability_pots": (gap / mass / s["pot"]) if mass > 0 else 0.0,
           "seconds": g.seconds, "leaf_seconds": g.leaf_seconds, "leaf_calls": g.leaf_calls,
           "history": ev["history"]}
    # against the trunk: o's best response to the re-solve vs its T values, p's value vs the trunk's
    cfv_o, br_o = s["cfv"][o][0, 0], ev["br"][o][0][0, 0]
    out["violation_sum"] = float(np.maximum(br_o - cfv_o, 0.0).sum())
    out["opponent_br_minus_cfv"] = float((br_o - cfv_o).sum())
    vp = ev["values"][player][0][0, 0]
    out["resolver_value"] = float((rp * vp).sum())
    trunk_v = s["blueprint"]["values"][player][0][0, 0] if "blueprint" in s else s["cfv"][player][0, 0]
    out["trunk_resolver_value"] = float((rp * trunk_v).sum())  # on-policy (br source) or averaged (avg)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--flop-line", required=True, help="the flop's closing betting string, e.g. cc or r0.5c")
    ap.add_argument("--turn", required=True, help="board A's and board B's turn cards, e.g. 5h2c")
    ap.add_argument("--player", type=int, default=0, choices=[0, 1])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tol", type=float, help="stop once gap / mass <= tol (chips), checked every --check-every")
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--cfv", default="auto", choices=["auto", "avg", "backup", "onpolicy", "br"])
    ap.add_argument("--simultaneous", action="store_true", help="simultaneous updates: one network pass per iteration")
    ap.add_argument("--keep-best", action="store_true", help="end on the check (every --check-every) with the "
                                                             "smallest gadget exploitability")
    ap.add_argument("--node", type=int, default=None, help="turn template node to print (default: the root for "
                                                            "player 0, player 1's first node otherwise)")
    ap.add_argument("--top", type=int, default=30, help="hands to print (by reach)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", help="override trunk.json's network")
    ap.add_argument("--out", help="write the strategy and the report to this .npz")
    a = ap.parse_args(argv)
    spot = Spot(a.trunk, a.device, a.model)
    c, l = find_line(spot.upper, a.flop_line)
    d = find_deal(spot.river, cards_of(a.turn, spot.ns))
    t0 = time.perf_counter()
    s = spot.setup(d, c, l, a.cfv)
    setup = time.perf_counter() - t0
    g, ev = spot.resolve(s, a.player, a.iters, a.tol, a.check_every, a.simultaneous, a.keep_best)
    rep = report(spot, s, g, ev, a.player)
    rep.update(setup_seconds=spot.setup_seconds, spot_setup_seconds=setup, simultaneous=a.simultaneous)
    tp = s["cls"].template
    node = a.node if a.node is not None else tp.players.index(a.player)
    acts, rows = strategy_table(spot, s, g, a.player, node)
    mass = sum(r[1] for r in rows)
    freq = sum(r[1] * r[2] for r in rows) / mass if mass > 0 else np.zeros(len(acts))
    print(f"turn re-solve: flop line {rep['flop_line']!r}, turn {''.join(rep['turn'])} (deal {d}), player "
          f"{a.player}, pot {s['pot']:.0f}, T values '{s['cfv_source']}'")
    print(f"{g.t} iterations in {g.seconds:.2f}s ({g.leaf_calls} network calls, {g.leaf_seconds:.2f}s); gadget "
          f"exploitability {rep['exploitability_chips']:.3g} chips = {rep['exploitability_pots']:.3g} pots; "
          f"setup {spot.setup_seconds:.1f}s + {setup:.2f}s")
    print(f"node {node} (betting so far {tp.hist[node]!r}), range-weighted: "
          + ", ".join(f"{x} {f:.3f}" for x, f in zip(acts, freq)))
    print(f"{'hand':6s} {'reach':>7s} " + " ".join(f"{x:>6s}" for x in acts))
    for name, r, p in rows[:a.top]:
        print(f"{name:6s} {r:7.4f} " + " ".join(f"{x:6.3f}" for x in p))
    if a.out:
        np.savez(a.out, strategy=g.average_strategy()[0][node][0, 0], actions=np.array(acts),
                 hands=spot.hands, reach=s["reach"][a.player][0, 0], valid=spot.turn.valid[d],
                 report=json.dumps(rep, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "history"}, default=float))


if __name__ == "__main__":
    main()
