"""Safe re-solving of a street with a leaf (core/solvers/leaf_resolve.py) and the turn re-solve
built on it (experiments/valuenet/turn_resolve.py), small deck (5 ranks x 3 suits, the real
no-limit betting of datagen.make_game "pot", one turn class of 13 lines).

- The gadget, on a last street without a leaf, is resolve._solve's: equal strategies, gadget
  averages and gaps, safe and unsafe, when every deal is its own subgame.
Tests that replay the whole river run on the limit game (one class per street), the others on
the pot game.
- Blueprint values per turn root (turn_resolve.blueprint_values, any subset of deals and lines)
  equal range_cfr.backward's on the flop+turn tree with the same leaf; turn_root_reach equals
  forward's turn roots.
- Safety, exactly: with a leaf that replays a fixed river strategy (the flop+turn game is then
  an ordinary game), re-solving every turn root for p with T values = o's best response to the
  trunk raises o's best response in that game by at most the summed gaps, and per root and
  hand o's best response exceeds its T value by at most the gap in total.
- LeafCFR(turn_cfv=True) keeps the turn-root values averaged with the trunk's weights and
  leaves the trunk's iterates unchanged.
- The one-deal NetLeaf of turn_resolve gives the trunk's NetLeaf values at that deal and line.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bombpot.core.solvers.leaf_resolve import LeafGadget
from bombpot.core.solvers.range_cfr import (RangeCFR, backward, close_values, forward, next_root, street_reach,
                                            street_values)
from bombpot.core.solvers.resolve import resolve_batch
from bombpot.experiments.valuenet import datagen as dg
from bombpot.experiments.valuenet import turn_resolve as tr
from bombpot.experiments.valuenet.gate_f import LeafCFR
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

FLOPS = ((0, 3, 7), (9, 11, 6))  # 2c3c4d / 5c6d4c
_built = {}


def built(name="pot", full=True):
    """(game, full tree or None, flop+turn tree, river). The tests that replay a whole river use
    the limit game (one class per street): the pot game's random river strategy alone is ~6 GB."""
    key = (name, full)
    if key not in _built:
        game = dg.make_game(FLOPS, 5, 3, name)
        upper = holdem_public_tree(game, upto=2)
        _built[key] = (game, holdem_public_tree(game) if full else None, upper, HoldemRiver(game, upper))
    return _built[key]


class ReplayLeaf:
    """The river under a fixed strategy for both players, as a leaf (tests/holdem/test_leaf.py)."""

    def __init__(self, full, river, hsig_river):
        D1 = full.streets[1].num_deals
        self.st = river.street(*river.all_children(np.arange(D1)))
        self.full, self.D1, self.hsig = full, D1, hsig_river

    def values(self, p, t, closing):
        cl = [[(r0[:, :, j], r1[:, :, j]) for j in range(r0.shape[2])] for r0, r1 in closing]
        f = self.full
        roots = next_root(cl, f.classes[1], f.classes[2], self.st.parent)
        nodes, _ = street_reach(f.classes[2], self.hsig, roots)
        V, _ = street_values(f.classes[2], self.st, self.hsig, nodes, p, "cfr", True)
        return close_values([v[0] for v in V], f.classes[1], f.classes[2], self.st.parent, self.D1)


class SubsetLeaf:
    """A full-tree leaf restricted to some turn deals and lines of class 0: closing arrays
    of the subset are scattered into zeros of the full shape (the other states get no reach)."""

    def __init__(self, leaf, upper, deals, lines):
        self.leaf, self.upper, self.deals, self.lines = leaf, upper, np.asarray(deals), np.asarray(lines)

    def values(self, p, t, closing):
        r0, r1 = closing[0]
        D = self.upper.streets[1].num_deals
        full = []
        for r in (r0, r1):
            x = np.zeros((D, self.upper.classes[1][0].num_lines) + r.shape[2:])
            x[np.ix_(self.deals, self.lines)] = r
            full.append(x)
        agg = self.leaf.values(p, t, [tuple(full)])
        return [agg[0][np.ix_(self.deals, self.lines)]]


def random_hsig(pt, seed, streets=None):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(pt.num_streets if streets is None else streets):
        out.append([])
        for c, lc in enumerate(pt.classes[t]):
            row = []
            for a in lc.template.actions:
                x = rng.random(pt.shape(t, c) + (len(a),)) ** 3
                row.append(x / x.sum(-1, keepdims=True))
            out[-1].append(row)
    return out


# The gadget itself, on a last street ------------------------------------------------------------


@pytest.mark.parametrize("safe", [True, False])
@pytest.mark.parametrize("p", [0, 1])
def test_gadget_equals_resolve_on_a_last_street(safe, p):
    game, _, upper, river = built("pot", full=False)
    st = river.street(*river.all_children([3]))
    n = st.num_deals
    rc = upper.classes[2][2]
    tp, contrib = rc.template, rc.contrib
    L, H = rc.num_lines, len(upper.hands)
    rng = np.random.default_rng(p + 2 * safe)
    rp = rng.random((n, L, H)) ** 2 * (rng.random((n, L, H)) > 0.2)
    ro = rng.random((n, L, H)) ** 2
    scale = st.fold(np.ones((n, L, H)), 1 - p).mean() * contrib.max()
    cfv = rng.uniform(-1, 1, (n, L, H)) * scale
    iters = 30
    group = np.arange(n)
    want = resolve_batch(st, tp, contrib, group, n, rp, ro, cfv if safe else None, p, "dcfr", iters, tol=0.0,
                         safe=safe, check_every=iters)
    lc = SimpleNamespace(template=tp, contrib=contrib)
    g = LeafGadget(st, [lc], [rp], [cfv] if safe else None, p, None, reach_o=None if safe else [ro])
    ev = g.solve(iters)
    assert int(np.max(want.iterations)) == iters == g.t
    avg = g.average_strategy()[0]
    held = rp > 0
    for k, q in enumerate(tp.players):
        if q == p or not safe:
            mask = held if q == p else np.ones_like(held)
            np.testing.assert_allclose(avg[k][mask], want.strategy[k][mask], rtol=1e-10, atol=1e-12)
        else:  # o's average is weighted by its F probability in both
            np.testing.assert_allclose(avg[k], want.strategy[k], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ev["gap"][0], want.gap, rtol=1e-9, atol=1e-12 * np.abs(want.gap).max())
    np.testing.assert_allclose(ev["mass"][0], want.mass, rtol=1e-12)
    np.testing.assert_allclose(ev["br"][1 - p][0], want.br_cfv[1 - p], rtol=1e-9, atol=1e-12 * scale)
    if safe:
        np.testing.assert_allclose(g.gadget_average()[0], want.gadget, rtol=1e-10, atol=1e-12)


# Blueprint values and the turn roots ------------------------------------------------------------


def test_blueprint_values_equal_backward():
    game, full, upper, river = built("limit")
    hsig = random_hsig(full, 5)
    leaf = ReplayLeaf(full, river, hsig[2])
    reach = forward(upper, hsig[:2])
    for c, lc in enumerate(upper.classes[1]):
        roots = tr.turn_root_reach(upper, hsig[0])[c]
        for q in (0, 1):  # the turn root reach is the flop's closing reach at every deal
            np.testing.assert_array_equal(np.broadcast_to(roots[q], reach[1][c][0][q].shape), reach[1][c][0][q])
    bp = tr.blueprint_values(upper.streets[1], upper.classes[1], hsig[1], [r[0] for r in reach[1]], leaf)
    for q in (0, 1):
        for mode, key in (("cfr", "values"), ("br", "br")):
            V, _ = backward(upper, hsig[:2], reach, q, mode, leaf=leaf)
            np.testing.assert_allclose(bp[key][q][0], V[1][0][0], rtol=1e-12, atol=1e-15)
    # one deal and one line, as the real-deck re-solve sets it up
    d, l = 7, 3
    st = upper.streets[1].subset([d])
    cls = tr.restrict(upper.classes[1][0], [l])
    sub = SubsetLeaf(leaf, upper, [d], [l])
    r = tuple(x[d:d + 1, l:l + 1] for x in reach[1][0][0])
    one = tr.blueprint_values(st, [cls], [[s[d:d + 1, l:l + 1] for s in hsig[1][0]]], [r], sub)
    for q in (0, 1):
        np.testing.assert_allclose(one["br"][q][0][0, 0], bp["br"][q][0][d, l], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(one["values"][q][0][0, 0], bp["values"][q][0][d, l], rtol=1e-12, atol=1e-15)


# Safety in an ordinary game (the river held fixed) ----------------------------------------------


def _replace(hsig_turn, new, player, tp):
    out = [list(row) for row in hsig_turn]
    for k, q in enumerate(tp.players):
        if q == player:
            out[0][k] = new[0][k]
    return out


@pytest.mark.parametrize("p", [0, 1])
def test_safe_turn_resolve_bound(p):
    game, full, upper, river = built("limit")
    o = 1 - p
    hsig_river = random_hsig(full, 11)[2]
    leaf = ReplayLeaf(full, river, hsig_river)
    trunk = LeafCFR(upper, leaf, "dcfr", turn_cfv=True)
    for _ in range(15):
        trunk.iteration()
    avg = trunk.hand_sigma(trunk.average_strategy())
    reach = forward(upper, avg)
    br_trunk = float(backward(upper, avg, reach, o, "br", leaf=leaf)[0][0][0][0].sum())
    roots = [r[0] for r in reach[1]]
    bp = tr.blueprint_values(upper.streets[1], upper.classes[1], avg[1], roots, leaf)
    tp = upper.classes[1][0].template
    results = {}
    for source in ("br", "avg"):
        cfv = bp["br"][o] if source == "br" else trunk.turn_cfv_average(o)
        g = LeafGadget(upper.streets[1], upper.classes[1], [roots[0][p]], cfv, p, leaf)
        ev = g.solve(40)
        comp = [avg[0], _replace(avg[1], g.average_strategy(), p, tp)]
        br = float(backward(upper, comp, forward(upper, comp), o, "br", leaf=leaf)[0][0][0][0].sum())
        results[source] = (br, float(ev["gap"][0].sum()), ev)
    br, gaps, ev = results["br"]
    assert br <= br_trunk + gaps + 1e-12, (br, br_trunk, gaps)
    viol = np.maximum(ev["br"][o][0] - bp["br"][o][0], 0.0).sum(-1)  # (D, L)
    assert (viol <= ev["gap"][0] + 1e-12 * np.abs(bp["br"][o][0]).sum(-1).max()).all()
    # the gadget at 40 iterations is tight compared with the trunk's room (a non-trivial check)
    assert gaps < 0.5 * abs(br_trunk)
    # the averaged values (the trunk's own) are within the same order here too
    assert results["avg"][0] <= br_trunk + results["avg"][1] + 0.05 * abs(br_trunk)


# The trunk's averaged turn-root values -----------------------------------------------------------


def test_trunk_turn_cfv_average():
    game, full, upper, river = built("limit")
    leaf = ReplayLeaf(full, river, random_hsig(full, 3)[2])
    a, b = LeafCFR(upper, leaf, "dcfr"), LeafCFR(upper, leaf, "dcfr", turn_cfv=True)
    sig0 = [[[s.copy() for s in cls] for cls in row] for row in b.hand_sigma()]
    for s in (a, b):
        s.iteration()
    # iteration 1 has weight 1: player 0's values come from the uniform start, player 1's from
    # player 0's updated strategy with its own still uniform
    v0 = backward(upper, sig0, forward(upper, sig0), 0, leaf=leaf)[0][1][0][0]
    np.testing.assert_allclose(b.turn_cfv_average(0)[0], v0, rtol=1e-12, atol=1e-15)
    sig1 = b.hand_sigma()
    for t in range(2):
        for k, q in enumerate(upper.classes[t][0].template.players):
            if q == 1:
                sig1[t][0][k] = sig0[t][0][k]
    v1 = backward(upper, sig1, forward(upper, sig1), 1, leaf=leaf)[0][1][0][0]
    np.testing.assert_allclose(b.turn_cfv_average(1)[0], v1, rtol=1e-12, atol=1e-15)
    for s in (a, b):
        s.iteration()
    for x, y in zip(a.regret[1][0], b.regret[1][0]):
        np.testing.assert_array_equal(x, y)
    assert a.leaf_cum[0][0].tobytes() == b.leaf_cum[0][0].tobytes()


# The one-deal network leaf ----------------------------------------------------------------------


def test_one_deal_net_leaf_equals_the_trunk_leaf():
    torch = pytest.importorskip("torch")
    from bombpot.experiments.valuenet.gate_f import turn_pots
    from bombpot.valuenet.features import FeatureConfig
    from bombpot.valuenet.leaf import NetLeaf
    from bombpot.valuenet.model import HandSetNet, ModelConfig

    game, _, upper, river = built("pot", full=False)
    turn = upper.streets[1]
    D, H = turn.valid.shape
    torch.manual_seed(0)
    cfg = FeatureConfig("hand_set", H, num_ranks=5, num_suits=3)
    model = HandSetNet(cfg, ModelConfig(kind="set", depth=2, width=16, heads=2)).double().eval()
    rng = np.random.default_rng(0)
    equity = rng.random((D, H, 2)) * turn.valid[..., None]
    boards = np.concatenate([np.broadcast_to(np.array(game.flops), (D, 2, 3)),
                             np.asarray(river.turn_cards)[:, :, None]], axis=-1)
    hands = np.array(upper.hands, dtype=np.int64)
    pots = turn_pots(upper)
    full_leaf = NetLeaf(model, equity, pots, [dg.STACK - x / 2 for x in pots], turn.kappa, boards_by_deal=boards,
                        possible_by_deal=turn.valid, hands=hands)
    lc = upper.classes[1][0]
    closing = [tuple(rng.random((D, lc.num_lines, lc.template.num_closes, H)) for _ in (0, 1))]
    d, lines = 11, [2, 9]
    one = tr.deal_leaf(model, equity[d], pots[0][lines], turn.kappa[d], boards[d], turn.valid[d], hands)
    sub = [tuple(x[d:d + 1, lines] for x in closing[0])]
    both = one.values_both(1, sub)
    for p in (0, 1):
        want = full_leaf.values(p, 1, closing)[0][d, lines]
        got = one.values(p, 1, sub)[0][0]
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(both[p][0][0], got)  # one network pass for both players


class _Both:
    """A leaf with values_both built from values (to count calls)."""

    def __init__(self, leaf):
        self.leaf, self.calls = leaf, 0

    def values(self, p, t, closing):
        return self.leaf.values(p, t, closing)

    def values_both(self, t, closing):
        self.calls += 1
        return self.leaf.values(0, t, closing), self.leaf.values(1, t, closing)


def test_simultaneous_gadget():
    # simultaneous updates: one values_both call per iteration (and one per evaluation), the
    # same iterates as two values calls, and the gadget gap falls
    game, full, upper, river = built("limit")
    leaf = ReplayLeaf(full, river, random_hsig(full, 4)[2])
    hsig = random_hsig(upper, 6)
    reach = forward(upper, hsig)
    roots = [r[0] for r in reach[1]]
    d = [5, 40]
    st = upper.streets[1].subset(d)
    sub = SubsetLeaf(leaf, upper, d, np.arange(upper.classes[1][0].num_lines))
    rp = [roots[0][0][d]]
    bp = tr.blueprint_values(st, upper.classes[1], [[s[d] for s in hsig[1][0]]], [tuple(r[d] for r in roots[0])], sub)
    a = LeafGadget(st, upper.classes[1], rp, bp["br"][1], 0, sub, simultaneous=True)
    counted = _Both(sub)
    b = LeafGadget(st, upper.classes[1], rp, bp["br"][1], 0, counted, simultaneous=True)
    early = a.solve(10)["gap"][0].sum()
    b.solve(10)
    assert counted.calls == 11 and b.leaf_calls == 11
    for x, y in zip(a.cum[0], b.cum[0]):
        np.testing.assert_array_equal(x, y)
    late = a.solve(80)["gap"][0].sum()
    assert late < 0.5 * early
