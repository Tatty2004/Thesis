"""R3: the Hold'em public tree with card-removal streets.

On small decks it must equal the generic dense public tree (core/public.py, built by
walking every deal): the same deals, parents, possible hands, betting, and fold and
showdown sums for any reach; and the range solver must produce the same iterates on
both. On the 52-card deck, fold and showdown sums are checked against brute force:
an explicit loop over opponent hands with the hand evaluator.
"""
import itertools
import random

import numpy as np
import pytest

from bombpot.core.public import build_public_tree
from bombpot.core.solvers.range_cfr import RangeCFR
from bombpot.holdem import Holdem
from bombpot.holdem.public import holdem_public_tree

SMALL = {
    "4x2-1-b2": dict(num_ranks=4, num_suits=2, board_cards=(1,), num_boards=2),
    "4x2-1+1-b1": dict(num_ranks=4, num_suits=2, board_cards=(1, 1), num_boards=1),
    "3x2-1+1-identical": dict(num_ranks=3, num_suits=2, board_cards=(1, 1), deck_mode="identical"),
    "5x2-3-b1": dict(num_ranks=5, num_suits=2, board_cards=(3,), num_boards=1),       # straights, flushes
    "5x3-3+1-flops": dict(num_ranks=5, num_suits=3, board_cards=(3, 1), flops=("2c3c4d", "5c6d4c")),
    "6x2-2+1-flops": dict(num_ranks=6, num_suits=2, board_cards=(2, 1), flops=("2c7d", "3c3d")),
}
_built = {}


def built(name):
    if name not in _built:
        g = Holdem(**SMALL[name])
        _built[name] = (g, build_public_tree(g), holdem_public_tree(g))
    return _built[name]


@pytest.mark.parametrize("name", SMALL)
def test_same_public_tree_as_the_dense_one(name):
    _, dense, fast = built(name)
    assert fast.hands == dense.hands
    assert [t.hist for t in fast.templates] == [t.hist for t in dense.templates]
    for a, b in zip(fast.contrib, dense.contrib):
        np.testing.assert_array_equal(a, b)
    rng = np.random.default_rng(0)
    for t, (f, d) in enumerate(zip(fast.streets, dense.streets)):
        assert f.deals == d.deals
        np.testing.assert_array_equal(f.parent, d.parent)
        for p in (0, 1):
            np.testing.assert_array_equal(f.possible(p), d.possible(p))
        reach = rng.random((len(d.deals), 3, len(dense.hands)))
        scale = d.fold(np.ones_like(reach), 0).max()
        for p in (0, 1):
            np.testing.assert_allclose(f.fold(reach, p), d.fold(reach, p), rtol=0, atol=1e-13 * scale)
            if t == len(dense.streets) - 1:
                np.testing.assert_allclose(f.share(reach, p), d.share(reach, p), rtol=0, atol=1e-13 * scale)


@pytest.mark.parametrize("name", ["4x2-1-b2", "5x3-3+1-flops", "6x2-2+1-flops"])
def test_range_solver_sees_the_same_game(name):
    # Lockstep along one run: the same strategies in, the same regrets out. (Free-running
    # runs part ways at regrets that are exactly zero by symmetry, which the two trees
    # round differently: see tests/test_range_cfr.py.)
    _, dense, fast = built(name)
    a, b = RangeCFR(dense), RangeCFR(fast)
    for _ in range(15):
        b.sigma = [[s.copy() for s in row] for row in a.sigma]
        for p in (0, 1):
            _, ra = a.instant_regrets(p)
            _, rb = b.instant_regrets(p)
            for x, y in zip(ra, rb):
                for u, v in zip(x, y):
                    if u is not None:
                        np.testing.assert_allclose(u, v, rtol=0, atol=1e-13)
        a.iteration()
    ea, eb = a.exploitability(), b.exploitability(a.average_strategy())
    for key in ea:
        assert eb[key] == pytest.approx(ea[key], abs=1e-12)


# 52 cards: brute force ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def real():
    game = Holdem(flops=("Ah7c2d", "KsKd9h"))
    return game, holdem_public_tree(game)


def test_real_deck_sizes(real):
    game, pt = real
    assert len(pt.hands) == 46 * 45 // 2
    assert [len(s.deals) for s in pt.streets] == [1, 46 * 45]
    # Every street's chance weights add up to one.
    for st in pt.streets:
        total = st.fold(np.ones((len(st.deals), 1, len(pt.hands))), 0).sum()
        assert total == pytest.approx(1.0, abs=1e-12)


def test_real_deck_sums_match_brute_force(real):
    game, pt = real
    st = pt.streets[-1]
    rng = random.Random(1)
    hands = pt.hands
    for _ in range(4):
        d = rng.randrange(len(st.deals))
        reach = np.random.default_rng(d).random((len(st.deals), 1, len(hands)))
        fold = st.fold(reach, 0)[d, 0]
        share0, share1 = st.share(reach, 0)[d, 0], st.share(reach, 1)[d, 0]
        key = st.deals[d]
        boards = (key[0::2], key[1::2])
        public = set(key)
        for i in rng.sample(range(len(hands)), 25):
            hi = hands[i]
            f = s0 = s1 = 0.0
            if not public & set(hi):
                for j, hj in enumerate(hands):
                    if public & set(hj) or set(hi) & set(hj):
                        continue
                    f += reach[d, 0, j]
                    sh = np.mean([np.sign(game.hand_value(hi + b) - game.hand_value(hj + b)) / 2 + 0.5 for b in boards])
                    s0 += reach[d, 0, j] * sh          # hand i as player 0 against j
                    s1 += reach[d, 0, j] * (1 - sh)    # hand j as player 0 against i (player 1)
            k = st.kappa[d]
            assert fold[i] == pytest.approx(k * f, rel=1e-12, abs=1e-18)
            assert share0[i] == pytest.approx(k * s0, rel=1e-12, abs=1e-18)
            assert share1[i] == pytest.approx(k * s1, rel=1e-12, abs=1e-18)
