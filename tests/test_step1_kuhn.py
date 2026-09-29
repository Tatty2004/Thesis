"""Correctness ladder, step 1: Kuhn poker."""
import numpy as np
import pytest

from helpers import brute_force_best_response, from_openspiel_policy, lockstep, naive_value, random_table
from toygames.eval.best_response import best_response
from toygames.eval.exploitability import exploitability
from toygames.games import Kuhn
from toygames.solvers.cfr import CFR, dcfr_discount
from toygames.solvers.lp import solve_game
from toygames.tree import compile_game, profile_value, strategy_from_table, strategy_table

KUHN_VALUE = -1 / 18


@pytest.fixture(scope="module")
def game():
    return Kuhn()


@pytest.fixture(scope="module")
def tree(game):
    return compile_game(game)


@pytest.fixture(scope="module")
def lp(tree):
    return solve_game(tree)


def test_sizes(tree):
    assert tree.n_infosets == [6, 6]
    assert tree.n_terminals == 30  # 6 deals x 5 betting sequences


def test_lp_value(lp):
    (v0, v1), _ = lp
    assert v0 == pytest.approx(KUHN_VALUE, abs=1e-9)
    assert v1 == pytest.approx(KUHN_VALUE, abs=1e-9)


def test_lp_player1_matches_known_strategy(tree, lp):
    # Player 1's equilibrium strategy in Kuhn is unique.
    _, sigma = lp
    table = strategy_table(tree, sigma)
    known = {
        "1:J::c": {"c": 2 / 3, "r": 1 / 3},  # bluff a third of the time
        "1:J::r": {"f": 1.0, "c": 0.0},
        "1:Q::c": {"c": 1.0, "r": 0.0},
        "1:Q::r": {"f": 2 / 3, "c": 1 / 3},  # call a third of the time
        "1:K::c": {"c": 0.0, "r": 1.0},
        "1:K::r": {"f": 0.0, "c": 1.0},
    }
    for key, probs in known.items():
        for a, prob in probs.items():
            assert table[key][a] == pytest.approx(prob, abs=1e-7), key


def test_lp_player0_in_known_family(tree, lp):
    # Player 0: bet J with alpha in [0, 1/3], bet K with 3 alpha, call with Q at alpha + 1/3.
    _, sigma = lp
    t = strategy_table(tree, sigma)
    alpha = t["0:J::"]["r"]
    assert -1e-7 <= alpha <= 1 / 3 + 1e-7
    assert t["0:K::"]["r"] == pytest.approx(3 * alpha, abs=1e-6)
    assert t["0:Q::"]["r"] == pytest.approx(0.0, abs=1e-7)
    assert t["0:Q::cr"]["c"] == pytest.approx(alpha + 1 / 3, abs=1e-6)
    assert t["0:J::cr"]["c"] == pytest.approx(0.0, abs=1e-7)
    assert t["0:K::cr"]["c"] == pytest.approx(1.0, abs=1e-7)


def test_lp_profile_is_equilibrium(tree, lp):
    _, sigma = lp
    rep = exploitability(tree, sigma)
    assert rep.exploitability < 1e-9
    assert rep.game_value == pytest.approx(KUHN_VALUE, abs=1e-9)


def test_dcfr_exploitability_goes_to_zero(tree):
    # Exploitability of the average strategy falls roughly like 1/T, with small wiggles.
    solver = CFR(tree, "dcfr")
    rows = solver.run(iterations=10_000, log_every=1000)
    expl = [r["exploitability"] for r in rows]
    assert max(expl[5:]) < min(expl[:3])
    assert expl[-1] < 3e-5
    assert rows[-1]["game_value"] == pytest.approx(KUHN_VALUE, abs=1e-6)


@pytest.mark.parametrize("variant", ["cfr", "cfr+", "dcfr"])
def test_log_uses_average_strategy(tree, variant):
    # The current strategy is the wrong thing to measure (it oscillates under CFR, and
    # looks misleadingly good under CFR+); the log must use the average.
    solver = CFR(tree, variant)
    for _ in range(25):
        solver.iteration()
    row = solver.log_row()
    avg = exploitability(tree, solver.average_strategy())
    cur = exploitability(tree, solver.current_strategy())
    assert row["exploitability"] == avg.exploitability
    assert (row["br_value_p0"], row["br_value_p1"]) == avg.br_value
    assert row["game_value"] == avg.game_value
    assert cur.exploitability != avg.exploitability


@pytest.mark.parametrize("variant", ["cfr", "cfr+"])
def test_other_variants_converge(tree, variant):
    rows = CFR(tree, variant).run(iterations=2000, log_every=2000)
    assert rows[-1]["exploitability"] < 1e-3
    assert rows[-1]["game_value"] == pytest.approx(KUHN_VALUE, abs=1e-3)


@pytest.mark.parametrize("seed", range(5))
def test_value_matches_naive_recursion(game, tree, seed):
    table = random_table(tree, seed)
    sigma = strategy_from_table(tree, table)
    assert profile_value(tree, sigma) == pytest.approx(naive_value(game, table), abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_best_response_matches_brute_force(game, tree, seed):
    table = random_table(tree, seed)
    sigma = strategy_from_table(tree, table)
    for p in (0, 1):
        value, br = best_response(tree, p, sigma[1 - p], return_strategy=True)
        assert value == pytest.approx(brute_force_best_response(game, tree, p, table), abs=1e-12)
        # The returned pure strategy achieves that value.
        prof = [sigma[0], sigma[1]]
        prof[p] = br
        achieved = profile_value(tree, prof)
        assert (achieved if p == 0 else -achieved) == pytest.approx(value, abs=1e-12)


def test_dcfr_discount_signs():
    r = np.array([2.0, -2.0, 0.0])
    t, alpha, beta = 3, 1.5, 0.0
    dcfr_discount(r, t, alpha, beta)
    assert r[0] == pytest.approx(2.0 * t**alpha / (t**alpha + 1))  # positive: t^a/(t^a+1)
    assert r[1] == pytest.approx(-2.0 * 0.5)  # negative with beta = 0: halved
    assert r[2] == 0.0


def test_uniform_exploitability_matches_openspiel(tree):
    pyspiel = pytest.importorskip("pyspiel")
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import exploitability as os_expl

    os_game = pyspiel.load_game("kuhn_poker")
    expected = os_expl.exploitability(os_game, os_policy.UniformRandomPolicy(os_game))
    assert exploitability(tree, tree.uniform()).exploitability == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("variant", ["cfr", "cfr+", "dcfr"])
def test_iterates_match_openspiel(game, tree, variant):
    """Same average strategy as OpenSpiel's solver after every one of 30 iterations."""
    pyspiel = pytest.importorskip("pyspiel")
    from open_spiel.python.algorithms import cfr as os_cfr
    from open_spiel.python.algorithms import discounted_cfr

    os_game = pyspiel.load_game("kuhn_poker")
    mapping = lockstep(os_game, game)
    os_solver = {"cfr": os_cfr.CFRSolver, "cfr+": os_cfr.CFRPlusSolver,
                 "dcfr": discounted_cfr.DCFRSolver}[variant](os_game)
    ours = CFR(tree, variant)
    for _ in range(30):
        os_solver.evaluate_and_update_policy()
        ours.iteration()
        theirs = from_openspiel_policy(os_solver.average_policy(), mapping, tree)
        mine = strategy_table(tree, ours.average_strategy())
        for key, probs in theirs.items():
            for a, prob in probs.items():
                assert mine[key][a] == pytest.approx(prob, abs=1e-10), (key, ours.t)
