"""R5: the river, solved by public chance sampling over an abstraction of river infosets.

A small three-street game (5 ranks x 3 suits, fixed flops: 36 hands, 72 turn deals, 3,024
river deals) is small enough for the full-width range solver, which tests/test_range_cfr.py
ties to the tree solver. Against it:
  - the exact evaluator, which streams the river in chunks, grades any stored strategy
    exactly as the full-width solver grades the same strategy spread over every river deal;
  - the sampled solver's instant regrets are unbiased: averaged over every possible sample
    they equal the full-width regrets summed into the same infosets, for the lossless
    abstraction and for equity buckets (imperfect recall);
  - CardStreet.equity() equals a brute-force count;
  - the sampled solver converges (slow).
"""
import numpy as np
import pytest

from bombpot.core.abstraction.river import EquityBuckets, fit_equity_buckets
from bombpot.core.solvers.range_cfr import RangeCFR
from bombpot.core.solvers.sampled_cfr import Lossless, SampledRangeCFR, exact_exploitability
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

GAME = Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=("2c3c4d", "5c6d4c"), bet_sizes=(2, 4, 4))


@pytest.fixture(scope="module")
def setup():
    full = holdem_public_tree(GAME)
    upper = holdem_public_tree(GAME, upto=2)
    river = HoldemRiver(GAME, upper)
    D1 = upper.streets[-1].num_deals
    parents, children = river.all_children(np.arange(D1))
    assert np.array_equal(river.street(parents, children).parent, full.streets[2].parent)  # same deal order
    abstractions = {"lossless": Lossless(D1, river.num_children, len(upper.hands)),
                    "avg_1d": fit_equity_buckets(river, D1, "avg_1d", 3, samples=12, seed=0),
                    "kmeans_2d": fit_equity_buckets(river, D1, "kmeans_2d", 3, samples=12, seed=0)}
    return full, upper, river, abstractions


def random_storage(solver, seed):
    rng = np.random.default_rng(seed)
    out = []
    for row in solver.sigma:
        out.append([])
        for s in row:
            x = rng.random(s.shape) ** 3
            out[-1].append(x / x.sum(-1, keepdims=True))
    return out


def lift(full, river, abstraction, sigma):
    """A stored strategy spread over every river deal of the full tree (per hand)."""
    D1 = full.streets[1].num_deals
    parents, children = river.all_children(np.arange(D1))
    st = river.street(parents, children)
    rows = abstraction.rows(parents, children)
    ids = [np.asarray(abstraction.ids(st, parents, children, p)) for p in (0, 1)]
    last = [s[rows[:, None], :, ids[q]].transpose(0, 2, 1, 3) for s, q in zip(sigma[2], full.templates[2].players)]
    return [list(sigma[0]), list(sigma[1]), last], st, rows, ids


@pytest.mark.parametrize("name", ["lossless", "avg_1d", "kmeans_2d"])
def test_exact_evaluation_streams_like_the_full_tree(setup, name):
    full, upper, river, abstractions = setup
    solver = SampledRangeCFR(upper, river, abstractions[name], batch=4)
    sigma = random_storage(solver, 1)
    streamed = exact_exploitability(upper, river, abstractions[name], sigma, chunk=700)
    hsig, *_ = lift(full, river, abstractions[name], sigma)
    whole = RangeCFR(full).exploitability(hsig)
    for key in whole:
        assert streamed[key] == pytest.approx(whole[key], rel=1e-12, abs=1e-14)


def full_width_regrets(full, river, abstraction, solver, p):
    """Player p's full-width instant regrets, summed into the stored infosets."""
    hsig, st, rows, ids = lift(full, river, abstraction, solver.sigma)
    rc = RangeCFR(full)
    _, R = rc.values(p, hsig, rc.reach(hsig))
    out = [[np.zeros_like(r) for r in row] for row in solver.regret]
    for t, tp in enumerate(full.templates):
        for k, q in enumerate(tp.players):
            if q != p:
                continue
            if t < 2:
                out[t][k] = R[t][k]
            else:
                touched, sums = solver._scatter(R[t][k], rows, ids[p], st.possible(p))
                out[t][k][touched] = sums
    return out


def assert_close(a, b):
    for row_a, row_b in zip(a, b):
        for x, y in zip(row_a, row_b):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-13 * max(1.0, np.abs(y).max()))


@pytest.mark.parametrize("name", ["lossless", "avg_1d", "kmeans_2d"])
def test_sampled_regrets_are_unbiased(setup, name):
    full, upper, river, abstractions = setup
    D1, n = upper.streets[-1].num_deals, river.num_children
    solver = SampledRangeCFR(upper, river, abstractions[name], batch=1)
    solver.sigma = random_storage(solver, 2)
    for p in (0, 1):
        want = full_width_regrets(full, river, abstractions[name], solver, p)
        # Every turn deal, one river child each: averaging over the child gives every river deal.
        mean = [[np.zeros_like(r) for r in row] for row in solver.regret]
        S = np.arange(D1)
        for c in range(n):
            got = solver.instant_regrets(p, S, np.arange(D1), np.full(D1, c), 1.0, 1.0 / n)
            mean = [[m + g / n for m, g in zip(rm, rg)] for rm, rg in zip(mean, got)]
        assert_close(mean, want)
        # One turn deal at a time, all its river children: averaging over the turn deal.
        mean = [[np.zeros_like(r) for r in row] for row in solver.regret]
        for d in range(D1):
            got = solver.instant_regrets(p, np.array([d]), np.zeros(n, dtype=np.int64), np.arange(n), 1.0 / D1, 1.0)
            mean = [[m + g / D1 for m, g in zip(rm, rg)] for rm, rg in zip(mean, got)]
        assert_close(mean, want)


def test_equity_matches_brute_force(setup):
    _, upper, river, _ = setup
    rng = np.random.default_rng(3)
    parents = rng.integers(0, upper.streets[-1].num_deals, size=5)
    children = rng.integers(0, river.num_children, size=5)
    st = river.street(parents, children)
    eq = st.equity()
    hands = upper.hands
    for d, (pa, ch) in enumerate(zip(parents, children)):
        rcards = river.river_cards(np.array([pa]), np.array([ch]))[0]
        tcards = river.turn_cards[pa]
        boards = [tuple(GAME.flops[b]) + (tcards[b], rcards[b]) for b in range(2)]
        public = set(boards[0]) | set(boards[1])
        for i, hi in enumerate(hands):
            if public & set(hi):
                assert not st.possible(0)[d, i] and (eq[d, i] == 0).all()
                continue
            opp = [hj for hj in hands if not (public | set(hi)) & set(hj)]
            for b in range(2):
                mine = GAME.hand_value(hi + boards[b])
                share = np.mean([1.0 if mine > GAME.hand_value(hj + boards[b]) else
                                 0.5 if mine == GAME.hand_value(hj + boards[b]) else 0.0 for hj in opp])
                assert eq[d, i, b] == pytest.approx(share, abs=1e-12)


def test_equity_buckets_are_well_formed(setup):
    _, upper, river, abstractions = setup
    for name in ("avg_1d", "kmeans_2d"):
        a = abstractions[name]
        assert isinstance(a, EquityBuckets) and a.num_ids == a.k + 1
        assert (np.diff(a.centers.mean(axis=2), axis=1) >= 0).all()  # weakest first
        parents = np.arange(upper.streets[-1].num_deals)
        children = np.zeros_like(parents)
        st = river.street(parents, children)
        for p in (0, 1):
            ids = a.ids(st, parents, children, p)
            assert ((ids == a.k) == ~st.possible(p)).all()  # the spare id is for hands that can't be held
            x = st.equity().mean(axis=2, keepdims=True) if name == "avg_1d" else st.equity()
            d2 = ((x[:, :, None, :] - a.centers[parents][:, None, :, :]) ** 2).sum(-1)
            nearest = d2.min(-1)
            got = np.take_along_axis(d2, np.minimum(ids, a.k - 1)[..., None], axis=2)[..., 0]
            assert np.allclose(got[st.possible(p)], nearest[st.possible(p)])


@pytest.mark.slow
def test_sampled_solver_converges(setup):
    # Lossless river, every turn deal and 4 of its 42 river deals per iteration.
    full, upper, river, abstractions = setup
    a = abstractions["lossless"]
    solver = SampledRangeCFR(upper, river, a, batch=upper.streets[-1].num_deals, samples=4, seed=0)
    history = {}
    for it in range(1, 201):
        solver.iteration()
        if it in (25, 200):
            history[it] = exact_exploitability(upper, river, a, solver.average_strategy())
    print({it: round(h["exploitability"], 4) for it, h in history.items()})
    assert history[200]["exploitability"] < 0.5 * history[25]["exploitability"]
    # Every profile's best responses bracket the game value; check against the full-width
    # value (DCFR, known to within twice its own exploitability).
    ref = RangeCFR(full)
    ref.run(iterations=150, log_every=150)
    v = ref.exploitability()
    lo, hi = v["game_value"] - 2 * v["exploitability"], v["game_value"] + 2 * v["exploitability"]
    for h in history.values():
        assert -h["br_value_p1"] - 1e-9 <= hi and lo <= h["br_value_p0"] + 1e-9


@pytest.mark.parametrize("name", ["lossless", "kmeans_2d"])
def test_an_iteration_applies_its_sample(setup, name):
    # The update bookkeeping (which rows, which weights) against the public pieces:
    # regrets grow by t x the sample's instant regrets, the average by t x own reach x the
    # strategy just played (own reach summed over each infoset's hands), and the new
    # strategy is regret matching. Player 0 updates first, so check player 0.
    from bombpot.core.solvers.range_cfr import forward

    _, upper, river, abstractions = setup
    solver = SampledRangeCFR(upper, river, abstractions[name], batch=8, samples=2, seed=5)
    for _ in range(3):  # get away from the all-zero start
        solver.iteration()
    sample = solver.sample()
    solver.sample = lambda: sample
    t = solver.t + 1
    regret = [[r.copy() for r in row] for row in solver.regret]
    cum = [[c.copy() for c in row] for row in solver.cum]
    played = [[s.copy() for s in row] for row in solver.sigma]
    inst = solver.instant_regrets(0, *sample)
    view, rows, ids = solver.view(*sample)
    reach = forward(view, solver.view_sigma(played, sample[0], rows, ids))
    solver.iteration()
    S = sample[0]
    for tt, tp in enumerate(upper.templates):
        for k, q in enumerate(tp.players):
            if q != 0:
                continue
            np.testing.assert_allclose(solver.regret[tt][k], regret[tt][k] + t * inst[tt][k], rtol=1e-12, atol=1e-15)
            own = reach[tt][k][0][..., None]
            if tt == 2:
                touched, own = solver._scatter(reach[tt][k][0], rows, ids[0], view.streets[2].possible(0))
                want = cum[tt][k].copy()
                want[touched] += t * own[..., None] * played[tt][k][touched]
            elif tt == 1:
                want = cum[tt][k].copy()
                want[S] += t * own * played[tt][k][S]
            else:
                want = cum[tt][k] + t * own * played[tt][k]
            np.testing.assert_allclose(solver.cum[tt][k], want, rtol=1e-12, atol=1e-15)
            pos = np.maximum(solver.regret[tt][k], 0)
            total = pos.sum(-1, keepdims=True)
            rm = np.where(total > 0, pos / np.where(total > 0, total, 1), 1 / pos.shape[-1])
            np.testing.assert_allclose(solver.sigma[tt][k], rm, rtol=1e-12, atol=1e-15)


def test_real_deck_river_sums_match_brute_force():
    # The river's strength tables and card bookkeeping on 52 cards, against an explicit loop
    # over opponent hands with the hand evaluator (as tests/holdem/test_holdem_public.py
    # does for the turn).
    from math import comb

    game = Holdem(flops=("Ah7c2d", "KsKd9h"), board_cards=(3, 1, 1), bet_sizes=(2, 4, 4))
    upper = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
    assert river.num_children == 44 * 43
    rng = np.random.default_rng(7)
    parents = rng.integers(0, upper.streets[-1].num_deals, size=3)
    children = rng.integers(0, river.num_children, size=3)
    st = river.street(parents, children)
    hands = upper.hands
    reach = rng.random((3, 1, len(hands)))
    fold, share0, share1 = st.fold(reach, 0)[:, 0], st.share(reach, 0)[:, 0], st.share(reach, 1)[:, 0]
    kappa = 1 / (comb(46, 2) * comb(44, 2) * 42 * 41 * 40 * 39)  # hands, then turn A, B, river A, B
    for d in range(3):
        assert st.kappa[d] == pytest.approx(kappa, rel=1e-12)
        rcards = river.river_cards(parents[d:d + 1], children[d:d + 1])[0]
        tcards = river.turn_cards[parents[d]]
        boards = [tuple(game.flops[b]) + (tcards[b], rcards[b]) for b in range(2)]
        public = set(boards[0]) | set(boards[1])
        assert len(public) == 10  # river cards are new and distinct
        for i in rng.choice(len(hands), size=20, replace=False):
            hi = hands[i]
            f = s0 = s1 = 0.0
            if not public & set(hi):
                for j, hj in enumerate(hands):
                    if (public | set(hi)) & set(hj):
                        continue
                    sh = np.mean([np.sign(game.hand_value(hi + b) - game.hand_value(hj + b)) / 2 + 0.5 for b in boards])
                    f += reach[d, 0, j]
                    s0 += reach[d, 0, j] * sh
                    s1 += reach[d, 0, j] * (1 - sh)
            assert fold[d, i] == pytest.approx(kappa * f, rel=1e-12, abs=1e-30)
            assert share0[d, i] == pytest.approx(kappa * s0, rel=1e-12, abs=1e-30)
            assert share1[d, i] == pytest.approx(kappa * s1, rel=1e-12, abs=1e-30)
