"""C: turn-end leaf values in the range solver (range_cfr.backward's `leaf`), small deck.

A leaf that replays the real river (every river deal of every turn deal, a fixed river
strategy, street_values and close_values) must reproduce backward on the full three-street
tree, in both modes: this pins the provider's units. ExactRiverLeaf's values equal separate
subgame solves from the closing ranges. Each check runs on the limit game (one class per
street) and on a no-limit game with short stacks (several classes per street, all-in lines
with no river betting).
"""
import numpy as np
import pytest

from bombpot.core.solvers.range_cfr import (RangeCFR, backward, close_values, forward, next_root, street_reach,
                                            street_values)
from bombpot.core.solvers.subgame import ExactRiverLeaf, line_template, solve_subgame
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

FLOPS = ("2c3c4d", "5c6d4c")
GAMES = {"limit": lambda: Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=FLOPS, bet_sizes=(2, 4, 4)),
         "pot": lambda: Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=FLOPS, betting="pot", stack=20,
                               sizes=(0.5, 1), max_aggressive=2)}
_built = {}


def built(name):
    if name not in _built:
        game = GAMES[name]()
        upper = holdem_public_tree(game, upto=2)
        _built[name] = (holdem_public_tree(game), upper, HoldemRiver(game, upper))
    return _built[name]


def random_hsig(pt, seed):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(pt.num_streets):
        out.append([])
        for c, lc in enumerate(pt.classes[t]):
            row = []
            for a in lc.template.actions:
                x = rng.random(pt.shape(t, c) + (len(a),)) ** 3
                row.append(x / x.sum(-1, keepdims=True))
            out[-1].append(row)
    return out


class ReplayLeaf:
    """The real river under a fixed strategy, as a leaf: what backward computes below the turn."""

    def __init__(self, full, river, hsig_river, mode="cfr"):
        D1 = full.streets[1].num_deals
        self.st = river.street(*river.all_children(np.arange(D1)))
        self.full, self.D1, self.hsig, self.mode = full, D1, hsig_river, mode

    def values(self, p, t, closing):
        cl = [[(r0[:, :, j], r1[:, :, j]) for j in range(r0.shape[2])] for r0, r1 in closing]
        f = self.full
        roots = next_root(cl, f.classes[1], f.classes[2], self.st.parent)
        nodes, _ = street_reach(f.classes[2], self.hsig, roots)
        V, _ = street_values(f.classes[2], self.st, self.hsig, nodes, p, self.mode, True)
        return close_values([v[0] for v in V], f.classes[1], f.classes[2], self.st.parent, self.D1)


def assert_close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * max(np.abs(b).max(), 1e-300))


@pytest.mark.parametrize("name", GAMES)
def test_forward_exposes_the_closing_reach(name):
    _, upper, _ = built(name)
    hsig = random_hsig(upper, 0)
    plain = forward(upper, hsig)
    reach, closing = forward(upper, hsig, closing=True)
    for a, b in zip(plain, reach):
        for x, y in zip(a, b):
            for u, v in zip(x, y):
                assert all(np.array_equal(i, j) for i, j in zip(u, v))
    assert len(closing) == len(upper.classes[1])
    for (r0, r1), lc in zip(closing, upper.classes[1]):
        assert r0.shape == (upper.streets[1].num_deals, lc.num_lines, lc.template.num_closes, len(upper.hands))


@pytest.mark.parametrize("name", GAMES)
@pytest.mark.parametrize("mode", ["cfr", "br"])
def test_replayed_river_reproduces_the_full_tree(name, mode):
    full, upper, river = built(name)
    hsig = random_hsig(full, 1)
    reach_full = forward(full, hsig)
    reach_up = forward(upper, hsig[:2])
    for p in (0, 1):
        leaf = ReplayLeaf(full, river, hsig[2], mode)
        Vf, Rf = backward(full, hsig, reach_full, p, mode)
        Vl, Rl = backward(upper, hsig[:2], reach_up, p, mode, leaf=leaf)
        for t in (0, 1):
            for c, lc in enumerate(upper.classes[t]):
                for k in range(max(lc.template.num_nodes, 1)):
                    assert_close(Vl[t][c][k], Vf[t][c][k])
                    if Rf[t][c] and Rf[t][c][k] is not None:
                        assert_close(Rl[t][c][k], Rf[t][c][k])
        assert Vl[0][0][0].sum() == pytest.approx(Vf[0][0][0].sum(), rel=1e-12)


@pytest.mark.parametrize("name", GAMES)
def test_range_cfr_with_a_replayed_river(name):
    # RangeCFR(upper, leaf=...) iterates the flop and turn exactly as RangeCFR on the full tree
    # does when the river strategy is held fixed: compare instant regrets and values.
    full, upper, river = built(name)
    hsig = random_hsig(full, 2)
    a = RangeCFR(full)
    b = RangeCFR(upper, leaf=ReplayLeaf(full, river, hsig[2]))
    assert len(b.regret) == 2
    a.sigma = [[list(c) for c in row] for row in hsig]
    b.sigma = [[list(c) for c in row] for row in hsig[:2]]
    for p in (0, 1):
        _, ra = a.instant_regrets(p)
        _, rb = b.instant_regrets(p)
        for t, c, k in b._nodes(p):
            assert_close(rb[t][c][k], ra[t][c][k])
        assert b.value[p] == pytest.approx(a.value[p], rel=1e-12)


@pytest.mark.parametrize("name", GAMES)
def test_exact_leaf_equals_separate_subgames(name):
    full, upper, river = built(name)
    leaf = ExactRiverLeaf(upper, river, tol=None, iters=30)
    hsig = random_hsig(upper, 3)
    _, closing = forward(upper, hsig, closing=True)
    got = [leaf.values(p, 1, closing) for p in (0, 1)]
    assert len(leaf.stats) == 1  # the second player's call reuses the same solve
    rng = np.random.default_rng(4)
    for rc in full.classes[2]:  # a few river lines of every river class
        for m in rng.choice(rc.num_lines, size=min(2, rc.num_lines), replace=False):
            pc, pl, pj = rc.parent_class[m], rc.parent_line[m], rc.parent_close[m]
            d = int(rng.integers(upper.streets[1].num_deals))
            st = river.street(*river.all_children([d]))
            sub = solve_subgame(st, line_template(rc.template, m), rc.contrib[m], closing[pc][0][d, pl, pj],
                                closing[pc][1][d, pl, pj], iters=30)
            for p in (0, 1):
                np.testing.assert_allclose(got[p][pc][d, pl, pj], sub.values[p], rtol=1e-12, atol=1e-20)
    # every closing line got a value: the river classes cover the turn's closes
    covered = [np.zeros(c[0].shape[1:3], dtype=bool) for c in closing]
    for rc in full.classes[2]:
        for pc, pl, pj in zip(rc.parent_class, rc.parent_line, rc.parent_close):
            covered[pc][pl, pj] = True
    assert all(c.all() for c in covered)
