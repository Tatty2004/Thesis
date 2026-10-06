"""R2: the range solver (core/solvers/range_cfr.py) against the tree solver.

Both run the same algorithm on the same game, one organized by terminal and one by
public node, so they must agree to rounding error:
  - free-running iterates for the first 20 iterations (after that, CFR's dynamics
    amplify rounding differences, as they do between any two implementations);
  - in lockstep along the tree solver's whole trajectory: given the same strategies,
    the same instant regrets and the same averaging weights (own reach);
  - the same best responses, exploitability and value for any profile.
Bucketed: the range solver with infoset ids from a toy-layer Bucketing against the tree
solver on the bucketed (abstract) tree. Bucketing often makes a regret exactly zero by
symmetry, and regret matching jumps there (uniform at 0, pure at +1e-18), so bucketed
runs are compared in lockstep only.
"""
import numpy as np
import pytest

from bombpot.core.abstraction.abstract_game import build_abstract_game, range_buckets
from bombpot.core.abstraction.bucketing import make_bucketing
from bombpot.core.abstraction.features import compute_features
from bombpot.core.eval.exploitability import exploitability
from bombpot.core.public import build_public_tree, from_tree_sigma, to_tree_sigma, tree_coordinates
from bombpot.core.solvers.cfr import CFR
from bombpot.core.solvers.lp import solve_game
from bombpot.core.solvers.range_cfr import RangeCFR
from bombpot.core.tree import compile_game, realization_plan
from bombpot.experiments.certify import random_profile
from bombpot.holdem import Holdem
from bombpot.toygames import BoardLeduc, Kuhn, Leduc

GAMES = {
    "kuhn": lambda: Kuhn(),
    "leduc": lambda: Leduc(),
    "board_leduc-3-1street": lambda: BoardLeduc(num_ranks=3),
    "board_leduc-4-shared": lambda: BoardLeduc(num_ranks=4, streets=2),
    "board_leduc-4-independent": lambda: BoardLeduc(num_ranks=4, streets=2, deck_mode="independent"),
    "board_leduc-4-identical": lambda: BoardLeduc(num_ranks=4, streets=2, deck_mode="identical"),
    "holdem-4x2-1-b2": lambda: Holdem(num_ranks=4, num_suits=2, board_cards=(1,), num_boards=2),
    "holdem-4x2-2-b1": lambda: Holdem(num_ranks=4, num_suits=2, board_cards=(2,), num_boards=1),
    "holdem-3x2-1+1-identical": lambda: Holdem(num_ranks=3, num_suits=2, board_cards=(1, 1), deck_mode="identical"),
}
_built = {}


def built(name):
    if name not in _built:
        g = GAMES[name]()
        _built[name] = (g, compile_game(g), build_public_tree(g))
    return _built[name]


# Games where some regret is exactly zero by symmetry from the first iteration (e.g. a
# hand that can neither win nor lose): there one solver computes 0 and the other 1e-19,
# regret matching plays uniform in one and pure in the other, and the runs part ways.
# Both are right; these games are checked in lockstep only.
EXACT_TIES = {"board_leduc-3-1street", "board_leduc-4-independent", "holdem-4x2-2-b1"}


@pytest.mark.parametrize("name", [n for n in GAMES if n not in EXACT_TIES])
@pytest.mark.parametrize("variant", ["dcfr", "cfr+", "cfr"])
def test_iterates_match_tree_solver(name, variant):
    _, tree, pt = built(name)
    a, b = CFR(tree, variant), RangeCFR(pt, variant)
    for _ in range(20):
        a.iteration()
        b.iteration()
        for x, y in zip(a.average_strategy(), to_tree_sigma(pt, tree, b.hand_sigma(b.average_strategy()))):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-9)
        for x, y in zip(a.sigma, to_tree_sigma(pt, tree, b.hand_sigma())):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-9)
        assert b.value == pytest.approx(a.value, abs=1e-12)


@pytest.mark.parametrize("name", GAMES)
def test_regrets_and_reach_match_in_lockstep(name):
    _, tree, pt = built(name)
    a, b = CFR(tree), RangeCFR(pt)
    co = [tree_coordinates(pt, tree, p) for p in (0, 1)]
    for _ in range(40):
        for p in (0, 1):
            b.sigma = from_tree_sigma(pt, tree, a.sigma)
            reach, regrets = b.instant_regrets(p)
            ra = a._instant_regret(p)
            xa = realization_plan(tree, p, a.sigma[p])
            for n, (t, d, l, k, i) in enumerate(co[p]):
                f, m = tree.first_seq[p][n], tree.num_actions[p][n]
                assert np.abs(ra[f:f + m] - regrets[t][k][d, l, i]).max() < 1e-12
                xb = b.own_reach(t, k, reach)[d, l, i] * b.sigma[t][k][d, l, i]
                assert np.abs(xa[f:f + m] - xb).max() < 1e-12
        a.iteration()


@pytest.mark.parametrize("name", GAMES)
def test_best_response_and_value_match(name):
    _, tree, pt = built(name)
    solver = CFR(tree)
    solver.run(iterations=50, log_every=50)
    b = RangeCFR(pt)
    for sigma in [solver.average_strategy(), random_profile(tree, 3), tree.uniform()]:
        ours = b.exploitability(from_tree_sigma(pt, tree, sigma))
        ref = exploitability(tree, sigma)
        assert ours["br_value_p0"] == pytest.approx(ref.br_value[0], abs=1e-12)
        assert ours["br_value_p1"] == pytest.approx(ref.br_value[1], abs=1e-12)
        assert ours["game_value"] == pytest.approx(ref.game_value, abs=1e-12)


@pytest.mark.parametrize("name", ["leduc", "board_leduc-4-shared", "holdem-4x2-1-b2"])
def test_converges_to_the_lp_value(name):
    _, tree, pt = built(name)
    (v, _), _ = solve_game(tree)
    b = RangeCFR(pt)
    for _ in range(1500):
        b.iteration()
    rep = b.exploitability()
    assert rep["exploitability"] < 1e-4
    assert -rep["br_value_p1"] - 1e-9 <= v <= rep["br_value_p0"] + 1e-9


# Bucketed ------------------------------------------------------------------------------

BUCKETED = [("board_leduc-4-shared", "kmeans_2d", 2), ("board_leduc-4-independent", "emd_2d", 3),
            ("board_leduc-4-shared", "avg_1d", 3), ("holdem-4x2-1-b2", "avg_1d", 3),
            ("holdem-3x2-1+1-identical", "kmeans_2d", 4)]


@pytest.mark.parametrize("name, method, k", BUCKETED, ids=[f"{n}-{m}-k{k}" for n, m, k in BUCKETED])
def test_bucketed_lockstep_and_total_error(name, method, k):
    game, full, pt = built(name)
    bucketing = make_bucketing(compute_features(game), method, k, seed=0, restarts=5)
    ag = build_abstract_game(full, bucketing)
    ids = range_buckets(pt, bucketing)
    a, b = CFR(ag.tree), RangeCFR(pt, buckets=ids)
    # Each abstract infoset is one (street, deal, line, node, id) of the range solver.
    where = []
    for p in (0, 1):
        co = tree_coordinates(pt, full, p)
        c = np.full((ag.tree.n_infosets[p], 5), -1, dtype=np.int64)
        flat = np.column_stack([co[:, :4], [ids[t][p][d, i] for t, d, _, _, i in co]])
        c[ag.infoset_map[p]] = flat
        assert (c[ag.infoset_map[p]] == flat).all() and (c >= 0).all()
        where.append(c)
    for _ in range(25):
        for p in (0, 1):
            for q in (0, 1):
                for n, (t, d, l, kk, i) in enumerate(where[q]):
                    f, m = ag.tree.first_seq[q][n], ag.tree.num_actions[q][n]
                    b.sigma[t][kk][d, l, i] = a.sigma[q][f:f + m]
            reach, regrets = b.instant_regrets(p)
            ra = a._instant_regret(p)
            xa = realization_plan(ag.tree, p, a.sigma[p])
            own = {}
            for n, (t, d, l, kk, i) in enumerate(where[p]):
                f, m = ag.tree.first_seq[p][n], ag.tree.num_actions[p][n]
                assert np.abs(ra[f:f + m] - regrets[t][kk][d, l, i]).max() < 1e-12
                if (t, kk) not in own:
                    own[(t, kk)] = b.own_reach(t, kk, reach)
                assert np.abs(xa[f:f + m] - own[(t, kk)][d, l, i] * b.sigma[t][kk][d, l, i]).max() < 1e-12
        a.iteration()
    # Total error: the range solver's full-game best response to the lifted strategy.
    sigma = [[s.copy() for s in row] for row in b.sigma]
    avg = a.average_strategy()
    for q in (0, 1):
        for n, (t, d, l, kk, i) in enumerate(where[q]):
            f, m = ag.tree.first_seq[q][n], ag.tree.num_actions[q][n]
            sigma[t][kk][d, l, i] = avg[q][f:f + m]
    ours = b.exploitability(sigma)
    ref = exploitability(full, ag.lift(avg))
    assert ours["exploitability"] == pytest.approx(ref.exploitability, abs=1e-12)
    assert ours["game_value"] == pytest.approx(ref.game_value, abs=1e-12)


@pytest.mark.slow
def test_three_streets_match_the_tree_solver():
    # The full-width reference for the river (tests/holdem/test_holdem_river.py): one
    # board, a card on each of three streets, 2.4 million terminals.
    game = Holdem(num_ranks=4, num_suits=2, board_cards=(1, 1, 1), num_boards=1, bet_sizes=(2, 4, 4))
    tree, pt = compile_game(game), build_public_tree(game)
    a, b = CFR(tree), RangeCFR(pt)
    co = [tree_coordinates(pt, tree, p) for p in (0, 1)]
    for _ in range(4):
        for p in (0, 1):
            b.sigma = from_tree_sigma(pt, tree, a.sigma)
            _, regrets = b.instant_regrets(p)
            ra = a._instant_regret(p)
            f, m = tree.first_seq[p], tree.num_actions[p]
            for n, (t, d, l, k, i) in enumerate(co[p]):
                assert np.abs(ra[f[n]:f[n] + m[n]] - regrets[t][k][d, l, i]).max() < 1e-12
        a.iteration()
    for sigma in [a.average_strategy(), random_profile(tree, 5)]:
        ours = RangeCFR(pt).exploitability(from_tree_sigma(pt, tree, sigma))
        ref = exploitability(tree, sigma)
        assert ours["br_value_p0"] == pytest.approx(ref.br_value[0], abs=1e-12)
        assert ours["br_value_p1"] == pytest.approx(ref.br_value[1], abs=1e-12)
        assert ours["game_value"] == pytest.approx(ref.game_value, abs=1e-12)
