"""E (PLAN_valuenet.md): river re-solving and the exact evaluator's river provider.

On the small three-street game of test_holdem_river.py (5 ranks x 3 suits, fixed flops
2c3c4d / 5c6d4c: 36 hands, 72 turn deals of 42 rivers each, 3,024 river deals):
  - the provider path with the stored strategy equals today's evaluator bit for bit (a frozen
    copy of it below), for the lossless river and fitted buckets, whole and split into shards;
    a re-solving provider's shards also add up to its unsplit evaluation exactly;
  - the subgame loop is RangeCFR's CFR+ on the subgame as its own public tree, bit for bit, and
    a batch of subgames equals each solved alone;
  - the gadget's guarantee: at every re-solved root, the opponent's best-response values exceed
    its blueprint values by at most the gadget's NashConv in total; those root values are the
    ones the exact evaluator sees; and safe re-solving raises the exact exploitability by at
    most half the sum of the gaps (unsafe re-solving is reported next to it);
  - slow: the same with every river re-solved, from a 20-iteration solve, and from the exact
    equilibrium of scripts/resolve_blueprints.py when its file exists.
"""
from pathlib import Path

import numpy as np
import pytest

from bombpot.core.abstraction.river import EquityBuckets, fit_equity_buckets
from bombpot.core.public import PublicTree
from bombpot.core.solvers.range_cfr import (RangeCFR, close_values, forward, load_range_strategy, next_root,
                                            street_reach, street_values)
from bombpot.core.solvers.resolve import Resolver, blueprint_cfv, resolve, resolve_batch, solve_subgame
from bombpot.core.solvers.sampled_cfr import (_JOBS, Lossless, SampledRangeCFR, StoredStrategy, exact_exploitability,
                                             finish_exploitability, last_street_sums, split_chunks, zero_sums)
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

GAME = Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=("2c3c4d", "5c6d4c"), bet_sizes=(2, 4, 4))
RESULTS = Path(__file__).resolve().parents[2] / "results" / "valuenet" / "resolve"


@pytest.fixture(scope="module")
def setup():
    upper = holdem_public_tree(GAME, upto=2)
    river = HoldemRiver(GAME, upper)
    D1 = upper.streets[-1].num_deals
    abstractions = {"lossless": Lossless(D1, river.num_children, len(upper.hands)),
                    "avg_1d": fit_equity_buckets(river, D1, "avg_1d", 3, samples=12, seed=0),
                    "kmeans_2d": fit_equity_buckets(river, D1, "kmeans_2d", 3, samples=12, seed=0)}
    return upper, river, abstractions


def random_storage(upper, river, abstraction, seed):
    rng = np.random.default_rng(seed)
    out = []
    for row in SampledRangeCFR(upper, river, abstraction, batch=1).sigma:
        out.append([])
        for s in row:
            x = rng.random(s.shape) ** 3
            out[-1].append(x / x.sum(-1, keepdims=True))
    return out


def today_last_street_sums(pt, last, abstraction, sigma, chunk=256, chunks=None):
    """last_street_sums as it was before the provider (commit 704bbeb), frozen as the reference."""
    T = len(pt.templates)
    D1 = pt.streets[-1].num_deals
    hsig_upper = [list(row) for row in sigma[:T - 1]]
    reach = forward(pt, hsig_upper)
    _, closing = street_reach(pt.templates[T - 2], hsig_upper[T - 2], reach[T - 2][0])
    tp_last = pt.templates[T - 1]
    agg = zero_sums(pt)
    total = D1 * last.num_children
    if chunks is None:
        chunks = range(-(-total // chunk))
    for c in chunks:
        flat = np.arange(c * chunk, min(total, (c + 1) * chunk))
        parents, children = np.divmod(flat, last.num_children)
        st = last.street(parents, children)
        root = next_root(closing, st.parent)
        rows = abstraction.rows(parents, children)
        ids = [np.asarray(abstraction.ids(st, parents, children, p)) for p in (0, 1)]
        hsig = [s[rows[:, None], :, ids[q]].transpose(0, 2, 1, 3) for s, q in zip(sigma[T - 1], tp_last.players)]
        nodes, _ = street_reach(tp_last, hsig, root)
        for j, (p, mode) in enumerate(_JOBS):
            V, _ = street_values(tp_last, st, pt.contrib[T - 1], hsig, nodes, p, mode, True)
            agg[j] += close_values(V[0], st.parent, D1, pt.templates[T - 2].num_closes)
    return agg


def stored(upper, abstraction, sigma):
    return StoredStrategy(abstraction, sigma[2], upper.templates[2].players)


# The provider path ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["lossless", "avg_1d", "kmeans_2d"])
@pytest.mark.parametrize("chunk", [700, 28])
def test_stored_provider_equals_today_exactly(setup, name, chunk):
    upper, river, abstractions = setup
    a = abstractions[name]
    sigma = random_storage(upper, river, a, 1)
    want = today_last_street_sums(upper, river, a, sigma, chunk)
    for provider in (None, stored(upper, a, sigma)):
        got = last_street_sums(upper, river, a, sigma, chunk, provider=provider)
        assert all(np.array_equal(x, y) for x, y in zip(got, want))
        assert exact_exploitability(upper, river, a, sigma, chunk, provider=provider) == \
            finish_exploitability(upper, sigma, want)


@pytest.mark.parametrize("name", ["lossless", "kmeans_2d"])
def test_split_evaluation_adds_up_with_both_providers(setup, name):
    # Chunks of 28 rivers, so a turn deal spans parts of two or three chunks (the re-solver's
    # cache must serve them), and two shards on turn-deal boundaries.
    upper, river, abstractions = setup
    a = abstractions[name]
    sigma = random_storage(upper, river, a, 2)
    D1, n = upper.streets[-1].num_deals, river.num_children
    for make in (lambda: stored(upper, a, sigma),
                 lambda: Resolver(upper, river, sigma, stored(upper, a, sigma), iters=20, deals=[1, 2, 40, 41],
                                  batch=2)):
        sums = zero_sums(upper)
        for r in split_chunks(D1, n, 28, 2):
            lo, hi = r.start * 28 // n, r.stop * 28 // n
            part = last_street_sums(upper, river, a, sigma, chunk=28, chunks=r, provider=make())
            for s, x in zip(sums, part):
                s[lo:hi] += x[lo:hi]
        whole = last_street_sums(upper, river, a, sigma, chunk=28, provider=make())
        assert all(np.array_equal(x, y) for x, y in zip(sums, whole))
        assert finish_exploitability(upper, sigma, sums) == exact_exploitability(upper, river, a, sigma, chunk=28,
                                                                                 provider=make())


# The subgame solver -----------------------------------------------------------------------------


def test_subgame_solver_is_range_cfr_on_the_subgame(setup):
    upper, river, _ = setup
    n, H, tp = river.num_children, len(upper.hands), upper.templates[2]
    st = river.street(np.full(n, 5), np.arange(n))
    contrib = upper.contrib[2][7:8]
    ref = RangeCFR(PublicTree("subgame", upper.hands, [st], [tp], [contrib]), "cfr+")
    for _ in range(30):
        ref.iteration()
    got = solve_subgame(st, tp, contrib, np.ones(H), np.ones(H), "cfr+", iters=30, check_every=30)
    assert all(np.array_equal(x, y) for x, y in zip(ref.average_strategy()[0], got.strategy))
    e = ref.exploitability()
    assert got.gap[0] == pytest.approx(2 * e["exploitability"], rel=1e-12)
    assert got.br_cfv[0].sum() == pytest.approx(e["br_value_p0"], rel=1e-12)


@pytest.mark.parametrize("safe", [True, False])
def test_a_batch_equals_each_subgame_alone(setup, safe):
    upper, river, _ = setup
    n, H, tp, contrib = river.num_children, len(upper.hands), upper.templates[2], upper.contrib[2]
    rng = np.random.default_rng(1)
    deals = [3, 10, 40]
    L = len(contrib)
    r0, r1 = rng.random((3, L, H)), rng.random((3, L, H))
    st = river.street(np.repeat(deals, n), np.tile(np.arange(n), 3))
    group = np.repeat(np.arange(3), n)
    bp = [rng.dirichlet(np.ones(len(a)), size=(3 * n, L, H)) for a in tp.actions]
    cfv = blueprint_cfv(st, tp, contrib, bp, r0, r1, 1, "br", group, 3)
    batch = resolve_batch(st, tp, contrib, group, 3, r0, r1, cfv, 0, iters=200, tol=2e-3, safe=safe)
    for g, d in enumerate(deals):
        one = river.street(np.full(n, d), np.arange(n))
        cf1 = blueprint_cfv(one, tp, contrib, [x[g * n:(g + 1) * n] for x in bp], r0[g], r1[g], 1, "br")
        alone = resolve(one, tp, contrib, r0[g], r1[g], cf1, 0, iters=200, tol=2e-3, safe=safe)
        assert np.array_equal(cf1, cfv[g])
        assert alone.iterations == batch.iterations[g] and np.array_equal(alone.gap, batch.gap[g])
        assert all(np.array_equal(x, y[g * n:(g + 1) * n]) for x, y in zip(alone.strategy, batch.strategy))


# Safety -----------------------------------------------------------------------------------------


def check_roots(resolver, sums, tol):
    """Per hand at every re-solved root: o's best-response value is at most its blueprint value
    plus the gadget's gap (summed over hands), and equals what the exact evaluator saw there.
    Returns the largest violation per root over its mass, and the sum of the gaps."""
    worst, total_gap = 0.0, 0.0
    for (d, p), g in resolver.diagnostics.items():
        o = 1 - p
        excess = np.maximum(g["violation"], 0.0).sum(-1)  # (L,)
        assert (excess <= g["gap"] + 1e-15 * np.abs(g["gap"]).max() + 1e-300).all()
        assert g["iterations"] == resolver.iters or g["exploitability"] <= tol
        seen = sums[o].reshape(sums[o].shape[0], -1, sums[o].shape[-1])[d]  # job o: o's best response
        np.testing.assert_allclose(g["br"], seen, rtol=1e-12, atol=1e-15 * np.abs(seen).max())
        m = np.where(g["mass"] > 0, g["mass"], 1.0)
        worst = max(worst, float((np.maximum(g["violation"], 0.0).max(-1) / m).max()))
        total_gap += float(g["gap"].sum())
    return worst, total_gap


def grade(upper, river, a, sigma, safe, deals, tol, iters, batch):
    r = Resolver(upper, river, sigma, stored(upper, a, sigma), safe=safe, iters=iters, tol=tol, deals=deals,
                 batch=batch, variant="cfr+")
    sums = last_street_sums(upper, river, a, sigma, chunk=1024, provider=r)
    return r, sums, finish_exploitability(upper, sigma, sums)


def test_safe_resolving_some_rivers_is_safe(setup):
    upper, river, abstractions = setup
    a = abstractions["lossless"]
    sigma = random_storage(upper, river, a, 3)  # a poor blueprint
    before = exact_exploitability(upper, river, a, sigma, chunk=1024)
    tol, deals = 1e-2, [0, 33]
    r, sums, after = grade(upper, river, a, sigma, True, deals, tol, 150, 2)
    _, total_gap = check_roots(r, sums, tol)
    print("safe", before["exploitability"], "->", after["exploitability"], "bound", total_gap / 2)
    assert after["exploitability"] <= before["exploitability"] + total_gap / 2 + 1e-12


def full_width_storage(upper, sigma_full):
    """A full-width three-street strategy as stored storage for the lossless river: the full
    tree's river deals are the river provider's (turn deal, child) pairs in order."""
    return [list(sigma_full[0]), list(sigma_full[1]), list(sigma_full[2])]


@pytest.mark.slow
def test_resolving_every_river_of_a_poor_blueprint_is_safe(setup):
    upper, river, abstractions = setup
    a = abstractions["lossless"]
    solver = RangeCFR(holdem_public_tree(GAME))
    for _ in range(20):
        solver.iteration()
    sigma = full_width_storage(upper, solver.average_strategy())
    before = exact_exploitability(upper, river, a, sigma, chunk=1024)
    assert before["exploitability"] == pytest.approx(solver.exploitability()["exploitability"], rel=1e-12)
    tol = 1e-3
    r, sums, after = grade(upper, river, a, sigma, True, None, tol, 2000, upper.streets[-1].num_deals)
    worst, total_gap = check_roots(r, sums, tol)
    _, _, unsafe = grade(upper, river, a, sigma, False, None, tol, 2000, upper.streets[-1].num_deals)
    print("20 iterations:", before["exploitability"], "safe", after["exploitability"], "bound", total_gap / 2,
          "unsafe", unsafe["exploitability"], "worst violation / mass", worst)
    assert after["exploitability"] <= before["exploitability"] + total_gap / 2 + 1e-12
    assert worst <= tol


@pytest.mark.slow
@pytest.mark.skipif(not (RESULTS / "eq.npz").exists(), reason="run scripts/resolve_blueprints.py eq first")
def test_resolving_every_river_of_the_equilibrium(setup):
    upper, river, abstractions = setup
    a = abstractions["lossless"]
    sigma, info = load_range_strategy(RESULTS / "eq.npz")
    before = exact_exploitability(upper, river, a, sigma, chunk=1024)
    assert before["exploitability"] == pytest.approx(info["exploitability"], rel=1e-9)
    tol = 1e-4
    r, sums, after = grade(upper, river, a, sigma, True, None, tol, 3000, upper.streets[-1].num_deals)
    _, total_gap = check_roots(r, sums, tol)
    print("equilibrium:", before, "re-solved:", after, "bound", total_gap / 2)
    assert after["exploitability"] <= before["exploitability"] + total_gap / 2 + 1e-12
    # Each value lies in [-br_value_p1, br_value_p0], an interval of width twice the exploitability
    # that holds the game value.
    assert abs(after["game_value"] - before["game_value"]) <= 2 * (before["exploitability"] + after["exploitability"]) + 1e-12
