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
  - on the pot game (several river line classes, one all-in), the re-solver works per class and
    its roots agree with the full-width range solver;
  - slow: the same with every river re-solved, from a 20-iteration solve, and from the exact
    equilibrium of scripts/resolve_blueprints.py when its file exists.
"""
from pathlib import Path

import numpy as np
import pytest

from bombpot.core.abstraction.river import EquityBuckets, fit_equity_buckets
from bombpot.core.public import LineClass, PublicTree
from bombpot.core.solvers.range_cfr import (RangeCFR, class_reach, class_values, close_values, forward,
                                            load_range_strategy, next_root)
from bombpot.core.solvers.resolve import (Resolver, blueprint_cfv, resolve, resolve_batch, restrict_lines,
                                          solve_subgame)
from bombpot.core.solvers.sampled_cfr import (_JOBS, Lossless, SampledRangeCFR, StoredStrategy, _classed, _unclassed,
                                             exact_exploitability, finish_exploitability, last_street_sums,
                                             one_class, split_chunks, zero_sums)
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
    """last_street_sums as it was before the provider (valuenet at 786abef, line classes),
    frozen as the reference."""
    cls = one_class(pt)
    T = len(cls)
    D1 = pt.streets[-1].num_deals
    hsig_upper = [list(row) for row in sigma[:T - 1]]
    reach = _unclassed(forward(pt, _classed(hsig_upper)))
    _, closing = class_reach(cls[T - 2].template, hsig_upper[T - 2], reach[T - 2][0])
    tp_last = cls[T - 1].template
    agg = zero_sums(pt)
    total = D1 * last.num_children
    if chunks is None:
        chunks = range(-(-total // chunk))
    for c in chunks:
        flat = np.arange(c * chunk, min(total, (c + 1) * chunk))
        parents, children = np.divmod(flat, last.num_children)
        st = last.street(parents, children)
        root = next_root([closing], cls[T - 2:T - 1], cls[T - 1:], st.parent)[0]
        rows = abstraction.rows(parents, children)
        ids = [np.asarray(abstraction.ids(st, parents, children, p)) for p in (0, 1)]
        hsig = [s[rows[:, None], :, ids[q]].transpose(0, 2, 1, 3) for s, q in zip(sigma[T - 1], tp_last.players)]
        nodes, _ = class_reach(tp_last, hsig, root)
        for j, (p, mode) in enumerate(_JOBS):
            V, _ = class_values(tp_last, st, cls[T - 1].contrib, hsig, nodes, p, mode, True)
            agg[j] += close_values([V[0]], cls[T - 2:T - 1], cls[T - 1:], st.parent, D1)[0]
    return agg


def stored(upper, abstraction, sigma):
    return StoredStrategy.one_class(abstraction, sigma[2], upper.classes[2][0].template.players)



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
    n, H, lc = river.num_children, len(upper.hands), upper.classes[2][0]
    st = river.street(np.full(n, 5), np.arange(n))
    tp, contrib = restrict_lines(lc.template, [7]), lc.contrib[7:8]
    one = np.array([0])
    sub = LineClass(tp, one, contrib, np.array([np.inf]), one - 1, one - 1, one - 1)
    ref = RangeCFR(PublicTree("subgame", upper.hands, [st], [[sub]]), "cfr+")
    for _ in range(30):
        ref.iteration()
    got = solve_subgame(st, tp, contrib, np.ones(H), np.ones(H), "cfr+", iters=30, check_every=30)
    assert all(np.array_equal(x, y) for x, y in zip(ref.average_strategy()[0][0], got.strategy))
    e = ref.exploitability()
    assert got.gap[0] == pytest.approx(2 * e["exploitability"], rel=1e-12)
    assert got.br_cfv[0].sum() == pytest.approx(e["br_value_p0"], rel=1e-12)


@pytest.mark.parametrize("safe", [True, False])
def test_a_batch_equals_each_subgame_alone(setup, safe):
    upper, river, _ = setup
    n, H = river.num_children, len(upper.hands)
    tp, contrib = upper.classes[2][0].template, upper.classes[2][0].contrib
    rng = np.random.default_rng(1)
    deals = [3, 10, 40]
    L = len(contrib)
    r0, r1 = rng.random((3, L, H)), rng.random((3, L, H))
    r0[1] *= rng.random((L, H)) < 0.05  # a sparse range converges at another speed: the batch shrinks
    st = river.street(np.repeat(deals, n), np.tile(np.arange(n), 3))
    group = np.repeat(np.arange(3), n)
    bp = [rng.dirichlet(np.ones(len(a)), size=(3 * n, L, H)) for a in tp.actions]
    cfv = blueprint_cfv(st, tp, contrib, bp, r0, r1, 1, "br", group, 3)
    tol = 2e-3 if safe else 5e-3
    batch = resolve_batch(st, tp, contrib, group, 3, r0, r1, cfv, 0, iters=200, tol=tol, safe=safe, check_every=5)
    assert len(set(batch.iterations.tolist())) > 1
    for g, d in enumerate(deals):
        one = river.street(np.full(n, d), np.arange(n))
        cf1 = blueprint_cfv(one, tp, contrib, [x[g * n:(g + 1) * n] for x in bp], r0[g], r1[g], 1, "br")
        alone = resolve(one, tp, contrib, r0[g], r1[g], cf1, 0, iters=200, tol=tol, safe=safe, check_every=5)
        assert np.array_equal(cf1, cfv[g])
        assert alone.iterations == batch.iterations[g] and np.array_equal(alone.gap, batch.gap[g])
        assert all(np.array_equal(x, y[g * n:(g + 1) * n]) for x, y in zip(alone.strategy, batch.strategy))


# Line classes (the pot game) --------------------------------------------------------------------

POT = Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=("2c3c4d", "5c6d4c"), betting="pot", stack=4)


def test_resolver_per_class_on_the_pot_game():
    # The pot game's river has several line classes, one of them all-in (no betting: the
    # gadget alone). Re-solve two turn deals of a random blueprint, and check every class's
    # re-solved root against the full-width range solver: the opponent's best-response root
    # values the re-solver reports are what RangeCFR computes for the combined profile, they
    # exceed the blueprint's by at most the gap, and exploitability rises by at most half the
    # gaps' sum.
    full, upper = holdem_public_tree(POT), holdem_public_tree(POT, upto=2)
    river = HoldemRiver(POT, upper)
    assert len(upper.classes[2]) > 2 and any(lc.template.num_nodes == 0 for lc in upper.classes[2])
    D1, n, H = upper.streets[-1].num_deals, river.num_children, len(upper.hands)
    rng = np.random.default_rng(4)
    hsig = [[[rng.dirichlet(np.ones(len(a)), size=full.shape(t, c)) for a in lc.template.actions]
             for c, lc in enumerate(full.classes[t])] for t in range(3)]
    blueprint = StoredStrategy(Lossless(D1, n, H), hsig[2], [lc.template.players for lc in upper.classes[2]])
    deals = [0, 33]
    r = Resolver(upper, river, hsig[:2], blueprint, iters=150, tol=1e-2, deals=deals, batch=2)
    combined = [hsig[0], hsig[1], [[s.copy() for s in row] for row in hsig[2]]]
    for d in deals:
        parents, children = np.full(n, d), np.arange(n)
        for c, row in enumerate(r(river.street(parents, children), parents, children)):
            for k, s in enumerate(row):
                combined[2][c][k][d * n:(d + 1) * n] = s
    assert {key[1] for key in r.diagnostics} == set(range(len(upper.classes[2])))
    rc = RangeCFR(full)
    reach = rc.reach(combined)
    br = [rc.values(o, combined, reach, "br")[0][2] for o in (0, 1)]
    total_gap = 0.0
    for (d, c, p), g in r.diagnostics.items():
        o = 1 - p
        want = br[o][c][0][d * n:(d + 1) * n].sum(axis=0)
        np.testing.assert_allclose(g["br"], want, rtol=1e-10, atol=1e-13 * np.abs(want).max())
        assert (np.maximum(g["violation"], 0).sum(-1) <= g["gap"] + 1e-12 * np.abs(g["br"]).max() + 1e-300).all()
        total_gap += float(g["gap"].sum())
    before, after = rc.exploitability(hsig), rc.exploitability(combined)
    print("pot game, 2 turn deals:", before["exploitability"], "->", after["exploitability"], "bound", total_gap / 2)
    assert after["exploitability"] <= before["exploitability"] + total_gap / 2 + 1e-12


# Safety -----------------------------------------------------------------------------------------


def check_roots(resolver, sums, tol):
    """Per hand at every re-solved root: o's best-response value is at most its blueprint value
    plus the gadget's gap (summed over hands), and equals what the exact evaluator saw there.
    Returns the largest violation per root over its mass, and the sum of the gaps."""
    worst, total_gap = 0.0, 0.0
    for (d, c, p), g in resolver.diagnostics.items():
        o = 1 - p
        excess = np.maximum(g["violation"], 0.0).sum(-1)  # (L,)
        assert (excess <= g["gap"] + 1e-12 * np.abs(g["br"]).max() + 1e-300).all()
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
    return [list(row[0]) for row in sigma_full]


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
    sigma = [row[0] for row in sigma]  # one class per street
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
