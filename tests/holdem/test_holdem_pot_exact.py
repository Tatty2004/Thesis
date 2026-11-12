"""The streamed exact evaluator (sampled_cfr.exact_exploitability) on the pot game, whose
streets have several line classes (all-in lines among them: zero-node classes, a plain
showdown on the river).

Small deck of test_holdem_river.py (5 ranks x 3 suits, flops 2c3c4d / 5c6d4c, 3,024 river
deals) with pot-limit/no-limit betting:
  - stack 4: river classes (lines, nodes) (1, 12), (4, 8), (12, 4), (24, 0);
  - stack 20: seven river classes, 137 lines.
Against the full-width range solver on holdem_public_tree(game) (tied to the tree solver by
tests/holdem/test_holdem_pot_range.py):
  - the streamed grade of a stored per-class strategy (lossless river) equals RangeCFR's best
    responses and value of the same strategy, for a random strategy and a few DCFR iterations;
  - shards on turn-deal boundaries add up to the unsplit grade exactly, with the stored
    strategy and with the Resolver provider;
  - safe re-solving through the Resolver provider grades as RangeCFR grades the re-solved
    profile and stays within its bound.
"""
import numpy as np
import pytest

from bombpot.core.solvers.range_cfr import RangeCFR
from bombpot.core.solvers.resolve import Resolver
from bombpot.core.solvers.sampled_cfr import (Lossless, StoredStrategy, exact_exploitability, finish_exploitability,
                                             last_street_sums, split_chunks, zero_sums)
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree


def pot_game(stack):
    return Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=("2c3c4d", "5c6d4c"), betting="pot",
                  stack=stack)


@pytest.fixture(scope="module", params=[4, 20])
def setup(request):
    game = pot_game(request.param)
    full, upper = holdem_public_tree(game), holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
    D1, n, H = upper.streets[-1].num_deals, river.num_children, len(upper.hands)
    assert len(upper.classes[2]) > 2 and any(lc.template.num_nodes == 0 for lc in upper.classes[2])
    return request.param, full, upper, river, Lossless(D1, n, H)


def random_profile(full, seed):
    rng = np.random.default_rng(seed)
    return [[[rng.dirichlet(np.ones(len(a)), size=full.shape(t, c)) for a in lc.template.actions]
             for c, lc in enumerate(full.classes[t])] for t in range(3)]


def dcfr_profile(full, iterations):
    solver = RangeCFR(full)
    for _ in range(iterations):
        solver.iteration()
    return solver.average_strategy()  # no buckets: per hand already


@pytest.mark.parametrize("which", ["random", "dcfr"])
def test_streamed_grade_equals_full_width(setup, which):
    # The lossless river's storage is the full tree's river strategy itself: row = turn deal x
    # rivers + river child, the full tree's deal order (test_holdem_river.py checks it).
    stack, full, upper, river, a = setup
    sigma = random_profile(full, 7) if which == "random" else dcfr_profile(full, 3)
    want = RangeCFR(full).exploitability(sigma)
    got = exact_exploitability(upper, river, a, sigma, chunk=700)
    print(stack, which, got)
    for key in want:
        assert got[key] == pytest.approx(want[key], rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("provider", ["stored", "resolver"])
def test_split_evaluation_adds_up_exactly(setup, provider):
    # Chunks of 28 rivers: a turn deal spans parts of two or three chunks (the re-solver's cache
    # must serve them), and the two shards start on turn-deal boundaries.
    stack, full, upper, river, a = setup
    if stack != 4:
        pytest.skip("one game is enough for the bookkeeping")
    sigma = random_profile(full, 8)
    D1, n = upper.streets[-1].num_deals, river.num_children

    def make():
        if provider == "stored":
            return None
        blueprint = StoredStrategy(a, sigma[2], [lc.template.players for lc in upper.classes[2]])
        return Resolver(upper, river, sigma, blueprint, iters=20, deals=[1, 2, 40, 41], batch=2)

    sums = zero_sums(upper)
    assert all(isinstance(s, list) and len(s) == len(upper.classes[1]) for s in sums)
    for r in split_chunks(D1, n, 28, 2):
        lo, hi = r.start * 28 // n, r.stop * 28 // n
        part = last_street_sums(upper, river, a, sigma, chunk=28, chunks=r, provider=make())
        for s, x in zip(sums, part):
            for sc, xc in zip(s, x):
                assert not xc[:lo].any() and not xc[hi:].any()
                sc[lo:hi] += xc[lo:hi]
    whole = last_street_sums(upper, river, a, sigma, chunk=28, provider=make())
    assert all(np.array_equal(sc, wc) for s, w in zip(sums, whole) for sc, wc in zip(s, w))
    assert finish_exploitability(upper, sigma, sums) == exact_exploitability(upper, river, a, sigma, chunk=28,
                                                                             provider=make())


def test_one_class_storage_is_refused(setup):
    # sigma[t][k] (the sampled solver's storage) has no meaning when a street has several classes.
    stack, full, upper, river, a = setup
    flat = [row[0] for row in random_profile(full, 10)]
    with pytest.raises(ValueError):
        last_street_sums(upper, river, a, flat, chunk=700)


def test_safe_resolving_through_the_evaluator(setup):
    stack, full, upper, river, a = setup
    if stack != 4:
        pytest.skip("the re-solve itself is tested per class in test_holdem_resolve.py")
    sigma = random_profile(full, 9)
    n = river.num_children
    blueprint = StoredStrategy(a, sigma[2], [lc.template.players for lc in upper.classes[2]])
    deals = [0, 33]
    r = Resolver(upper, river, sigma, blueprint, iters=150, tol=1e-2, deals=deals, batch=2)
    before = exact_exploitability(upper, river, a, sigma, chunk=700)
    after = exact_exploitability(upper, river, a, sigma, chunk=700, provider=r)
    # The same profile, built by hand for the full-width solver.
    combined = [sigma[0], sigma[1], [[s.copy() for s in row] for row in sigma[2]]]
    for d in deals:
        parents, children = np.full(n, d), np.arange(n)
        for c, row in enumerate(r(river.street(parents, children), parents, children)):
            for k, s in enumerate(row):
                combined[2][c][k][d * n:(d + 1) * n] = s
    want = RangeCFR(full).exploitability(combined)
    for key in want:
        assert after[key] == pytest.approx(want[key], rel=1e-12, abs=1e-14)
    total_gap = sum(float(g["gap"].sum()) for g in r.diagnostics.values())
    print("pot game, 2 turn deals re-solved:", before["exploitability"], "->", after["exploitability"],
          "bound", total_gap / 2)
    assert after["exploitability"] <= before["exploitability"] + total_gap / 2 + 1e-12
