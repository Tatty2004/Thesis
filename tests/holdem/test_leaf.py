"""C: turn-end leaf values in the range solver (range_cfr.backward's `leaf`), small deck.

A leaf that replays the real river (every river deal of every turn deal, a fixed river
strategy, street_values and close_values) must reproduce backward on the full three-street
tree, in both modes: this pins the provider's units. ExactRiverLeaf's values equal separate
subgame solves from the closing ranges.
"""
import numpy as np
import pytest

from bombpot.core.solvers.range_cfr import (RangeCFR, backward, close_values, forward, next_root, stack_closing,
                                            street_reach, street_values)
from bombpot.core.solvers.subgame import ExactRiverLeaf, solve_subgame
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

GAME = Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=("2c3c4d", "5c6d4c"), bet_sizes=(2, 4, 4))


@pytest.fixture(scope="module")
def setup():
    full = holdem_public_tree(GAME)
    upper = holdem_public_tree(GAME, upto=2)
    river = HoldemRiver(GAME, upper)
    return full, upper, river


def random_hsig(pt, seed):
    rng = np.random.default_rng(seed)
    out = []
    for t, tp in enumerate(pt.templates[:pt.num_streets]):
        out.append([])
        for a in tp.actions:
            x = rng.random(pt.shape(t) + (len(a),)) ** 3
            out[-1].append(x / x.sum(-1, keepdims=True))
    return out


class ReplayLeaf:
    """The real river under a fixed strategy, as a leaf: what backward computes below the turn."""

    def __init__(self, upper, river, template, contrib, hsig_river, mode="cfr"):
        D1 = upper.streets[-1].num_deals
        self.st = river.street(*river.all_children(np.arange(D1)))
        self.D1, self.tp, self.contrib, self.hsig, self.mode = D1, template, contrib, hsig_river, mode

    def values(self, p, t, closing):
        C = closing[0].shape[2]
        root = next_root([(closing[0][:, :, c], closing[1][:, :, c]) for c in range(C)], self.st.parent)
        nodes, _ = street_reach(self.tp, self.hsig, root)
        V, _ = street_values(self.tp, self.st, self.contrib, self.hsig, nodes, p, self.mode, True)
        return close_values(V[0], self.st.parent, self.D1, C)


def test_forward_exposes_the_closing_reach(setup):
    _, upper, _ = setup
    hsig = random_hsig(upper, 0)
    plain = forward(upper, hsig)
    reach, closing = forward(upper, hsig, closing=True)
    for a, b in zip(plain, reach):
        for x, y in zip(a, b):
            assert all(np.array_equal(u, v) for u, v in zip(x, y))
    _, close = street_reach(upper.templates[1], hsig[1], plain[1][0])
    want = stack_closing(close)
    D, L, H = upper.shape(1)
    assert closing[0].shape == (D, L, upper.templates[1].num_closes, H)
    assert all(np.array_equal(a, b) for a, b in zip(closing, want))


@pytest.mark.parametrize("mode", ["cfr", "br"])
def test_replayed_river_reproduces_the_full_tree(setup, mode):
    full, upper, river = setup
    hsig = random_hsig(full, 1)
    reach_full = forward(full, hsig)
    reach_up = forward(upper, hsig[:2])
    for p in (0, 1):
        leaf = ReplayLeaf(upper, river, full.templates[2], full.contrib[2], hsig[2], mode)
        Vf, Rf = backward(full, hsig, reach_full, p, mode)
        Vl, Rl = backward(upper, hsig[:2], reach_up, p, mode, leaf=leaf)
        for t in (0, 1):
            for k in range(upper.templates[t].num_nodes):
                np.testing.assert_allclose(Vl[t][k], Vf[t][k], rtol=1e-12, atol=1e-12 * np.abs(Vf[t][k]).max())
                if Rf[t][k] is not None:
                    np.testing.assert_allclose(Rl[t][k], Rf[t][k], rtol=1e-12, atol=1e-12 * np.abs(Rf[t][k]).max())
        assert Vl[0][0].sum() == pytest.approx(Vf[0][0].sum(), rel=1e-12)


def test_range_cfr_with_a_replayed_river(setup):
    # RangeCFR(upper, leaf=...) iterates the flop and turn exactly as RangeCFR on the full tree
    # does when the river strategy is held fixed: compare one iteration's instant regrets.
    full, upper, river = setup
    hsig = random_hsig(full, 2)
    a = RangeCFR(full)
    leaf = ReplayLeaf(upper, river, full.templates[2], full.contrib[2], hsig[2])
    b = RangeCFR(upper, leaf=leaf)
    assert len(b.pt.templates) == 2 and len(b.regret) == 2
    a.sigma = [list(row) for row in hsig]
    b.sigma = [list(row) for row in hsig[:2]]
    for p in (0, 1):
        _, ra = a.instant_regrets(p)
        _, rb = b.instant_regrets(p)
        for t in (0, 1):
            for k, q in enumerate(upper.templates[t].players):
                if q == p:
                    np.testing.assert_allclose(rb[t][k], ra[t][k], rtol=1e-12, atol=1e-12 * np.abs(ra[t][k]).max())
        assert b.value[p] == pytest.approx(a.value[p], rel=1e-12)


def test_exact_leaf_equals_separate_subgames(setup):
    full, upper, river = setup
    leaf = ExactRiverLeaf(upper, river, full.templates[2], full.contrib[2], tol=None, iters=30)
    hsig = random_hsig(upper, 3)
    _, closing = forward(upper, hsig, closing=True)
    got = [leaf.values(p, 1, closing) for p in (0, 1)]
    assert len(leaf.stats) == 1  # the second player's call reuses the same solve
    C = upper.templates[1].num_closes
    for d, l, c in [(0, 0, 0), (11, 2, 4), (71, 4, 1)]:
        st = river.street(*river.all_children([d]))
        sub = solve_subgame(st, full.templates[2], full.contrib[2][l * C + c], closing[0][d, l, c],
                            closing[1][d, l, c], iters=30)
        for p in (0, 1):
            np.testing.assert_allclose(got[p][d, l, c], sub.values[p], rtol=1e-12, atol=1e-20)
