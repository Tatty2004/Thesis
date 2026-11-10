"""D: the last-street subgame solver (core/solvers/subgame.py) on the small three-street deck.

A subgame entered with root ranges (r0, r1) is the same game as the one-street public tree
whose chance matrix carries the ranges, W'[d, i, j] = W[d, i, j] r0[i] r1[j] (and S' likewise),
entered with reach 1. On that dense tree plain RangeCFR applies; its counterfactual values
and regrets are the subgame's times the acting player's own root reach, and its own reach
the subgame's divided by it, so regret matching, CFR+'s floor and the average strategy are
the same. Checked in lockstep and free-running, plus best responses, and the units of the
returned values against street_values directly.
"""
import numpy as np
import pytest

from bombpot.core.public import DenseStreet, PublicTree
from bombpot.core.solvers.range_cfr import RangeCFR, street_reach, street_values
from bombpot.core.solvers.subgame import SubgameCFR, solve_subgame, solve_subgames
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

GAME = Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=("2c3c4d", "5c6d4c"), bet_sizes=(2, 4, 4))


@pytest.fixture(scope="module")
def setup():
    upper = holdem_public_tree(GAME, upto=2)
    river = HoldemRiver(GAME, upper)
    return upper, river


def ranges(H, seed, valid=None):
    rng = np.random.default_rng(seed)
    r = [rng.random(H) ** 2 + 0.01 for _ in (0, 1)]
    if valid is not None:
        r = [x * valid for x in r]
    return [x / x.sum() for x in r]


def dense_tree(st, template, contrib, r0, r1):
    """The subgame as a one-street dense public tree with the root ranges in its chance matrix."""
    D, H = st.num_deals, len(st.hands)
    eye = np.broadcast_to(np.eye(H), (D, H, H)).copy()  # reach[d, l = j, :] = e_j
    W = np.swapaxes(st.fold(eye, 0), 1, 2)   # W[d, i, j]
    S = np.swapaxes(st.share(eye, 0), 1, 2)  # S[d, i, j]
    outer = r0[None, :, None] * r1[None, None, :]
    dense = DenseStreet(list(range(D)), np.full(D, -1, dtype=np.int64), W * outer, S * outer)
    return PublicTree("dense", list(range(H)), [dense], [template], [np.array([float(contrib)])])


@pytest.mark.parametrize("d, line", [(0, 0), (17, 3), (50, 12)])
def test_subgame_matches_range_cfr_on_the_dense_game(setup, d, line):
    upper, river = setup
    st = river.street(*river.all_children([d]))
    template, contrib = upper.templates[2], upper.contrib[2][line]
    H = len(upper.hands)
    r = ranges(H, d)
    a = SubgameCFR(st, template, [contrib], (r[0], r[1]))
    b = RangeCFR(dense_tree(st, template, contrib, r[0], r[1]), "cfr+")
    nodes = [(k, q) for k, q in enumerate(template.players)]
    for it in range(30):
        # Lockstep: the same strategies give the same regrets, scaled by own root reach.
        for p in (0, 1):
            b.sigma = [[s.copy() for s in a.sigma[0]]]
            _, ra = a.instant_regrets(p)
            _, rb = b.instant_regrets(p)
            for k, q in nodes:
                if q == p:
                    want = ra[0][k] * r[p][None, None, :, None]
                    np.testing.assert_allclose(rb[0][k], want, rtol=0, atol=1e-13 * max(1.0, np.abs(want).max()))
        a.iteration()
    # Whole updates in lockstep: from the same state (regrets x own root reach, averages /
    # own root reach, the same strategy), one iteration gives the same state again. Free-running
    # runs part ways at exact ties (a regret of 0 in one solver and 1e-22 in the other, see
    # tests/test_range_cfr.py), so they're compared this way.
    for it in range(20):
        b.t = a.t
        scale = [r[q][None, None, :, None] for q in template.players]
        b.sigma = [[s.copy() for s in a.sigma[0]]]
        b.regret = [[x * c for x, c in zip(a.regret[0], scale)]]
        b.cum = [[x / c for x, c in zip(a.cum[0], scale)]]
        a.iteration()
        b.iteration()
        for k, c in enumerate(scale):
            np.testing.assert_allclose(b.regret[0][k], a.regret[0][k] * c, rtol=1e-9,
                                       atol=1e-13 * np.abs(b.regret[0][k]).max())
            np.testing.assert_allclose(b.cum[0][k] * c, a.cum[0][k], rtol=1e-11, atol=1e-15)
    # The same best responses and values of the average strategy.
    ev = a.evaluate()
    hsig = b.hand_sigma(a.average_strategy())
    reach = b.reach(hsig)
    for p in (0, 1):
        for mode, key in (("cfr", "values"), ("br", "br_values")):
            vb = b.values(p, hsig, reach, mode)[0][0][0].sum(0)[0]
            np.testing.assert_allclose(vb, ev[key][p][0, 0] * r[p], rtol=1e-12, atol=1e-18)


def test_values_are_in_street_values_units(setup):
    upper, river = setup
    d, line = 33, 7
    st = river.street(*river.all_children([d]))
    template, contrib = upper.templates[2], upper.contrib[2][line]
    r = ranges(len(upper.hands), 4, upper.streets[1].valid[d])
    sub = solve_subgame(st, template, contrib, r[0], r[1], iters=1000, tol=1e-3)
    assert sub.exploitability <= 1e-3 and sub.iterations < 1000
    D, H = st.num_deals, len(upper.hands)
    hsig = [s[:, None] for s in sub.strategy]  # (D, 1, H, A)
    root = tuple(np.broadcast_to(x, (D, 1, H)).copy() for x in r)
    nodes, _ = street_reach(template, hsig, root)
    for p in (0, 1):
        V, _ = street_values(template, st, np.array([contrib]), hsig, nodes, p, "cfr", True)
        np.testing.assert_allclose(sub.values[p], V[0][:, 0].sum(0), rtol=1e-12, atol=1e-18)
        Vb, _ = street_values(template, st, np.array([contrib]), hsig, nodes, p, "br", True)
        np.testing.assert_allclose(sub.br_values[p], Vb[0][:, 0].sum(0), rtol=1e-12, atol=1e-18)
    # Exploitability in chips: (br0 + br1) / 2 over the root reach mass; zero-sum value.
    mass = (st.fold(root[1], 0)[:, 0] * r[0]).sum()
    # (counterfactual values leave out own reach, so totals weight each hand by its own range)
    br = [sub.br_values[p] @ r[p] for p in (0, 1)]
    assert sub.exploitability == pytest.approx((br[0] + br[1]) / 2 / mass, rel=1e-12)
    assert sub.values[0] @ r[0] + sub.values[1] @ r[1] == pytest.approx(0, abs=1e-15)  # zero-sum
    assert (sub.br_values[0] >= sub.values[0] - 1e-18).all() and (sub.br_values[1] >= sub.values[1] - 1e-18).all()


def test_a_batch_is_independent_subgames(setup):
    # Two turn deals x two lines at once equal four separate solves (fixed iteration count).
    upper, river = setup
    deals, lines = np.array([5, 40]), np.array([0, 9])
    parents, children = river.all_children(deals)
    st = river.street(parents, children)
    group = np.repeat(np.arange(2), river.num_children)
    H = len(upper.hands)
    r = [np.zeros((2, 2, H)), np.zeros((2, 2, H))]
    for g in range(2):
        for j in range(2):
            x = ranges(H, 10 * g + j)
            r[0][g, j], r[1][g, j] = x
    batch = solve_subgames(st, upper.templates[2], upper.contrib[2][lines], r[0], r[1], group, iters=40)
    for g, d in enumerate(deals):
        one = river.street(*river.all_children([d]))
        for j, line in enumerate(lines):
            sub = solve_subgame(one, upper.templates[2], upper.contrib[2][line], r[0][g, j], r[1][g, j], iters=40)
            for p in (0, 1):
                np.testing.assert_allclose(batch.values[p][g, j], sub.values[p], rtol=1e-12, atol=1e-18)
            assert batch.exploitability[g, j] == pytest.approx(sub.exploitability, rel=1e-10)
            for k in range(len(sub.strategy)):
                np.testing.assert_allclose(batch.strategy[k][group == g, j], sub.strategy[k], rtol=0, atol=1e-12)
