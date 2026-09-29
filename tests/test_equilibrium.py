"""Check B: our two-board strategies are equilibria, graded by code that isn't ours.

- OpenSpiel grades them through a thin wrapper (openspiel_game.py). Its own tree walk,
  best response and expected-value code must reproduce our best responses, value and
  exploitability, and its DCFR must reproduce our iterates on two-board games.
- A linear program (HiGHS) computes every best response a second way.
- Every solver's value bracket must contain the LP value.
- The dominated-action audit and the board-swap test look at the strategies themselves.
- Saved strategies load back exactly.
toygames/experiments/certify.py runs the non-OpenSpiel checks for N = 4 to 7.
"""
import numpy as np
import pytest

from helpers import from_openspiel_policy, lockstep, random_table
from toygames.eval.audit import audit, dominated_actions
from toygames.eval.exploitability import exploitability
from toygames.experiments.certify import random_profile, swap_boards, swap_key
from toygames.games import BoardLeduc, Kuhn, Leduc
from toygames.solvers.cfr import CFR
from toygames.solvers.lp import best_response_lp, solve_game
from toygames.tree import compile_game, load_strategy, save_strategy, strategy_from_table, strategy_table

pyspiel = pytest.importorskip("pyspiel")
from open_spiel.python.algorithms import discounted_cfr  # noqa: E402
from open_spiel.python.algorithms import exploitability as os_expl  # noqa: E402
from openspiel_game import KeyPolicy, OpenSpielGame  # noqa: E402

MODES = ("shared", "independent", "identical")
TOL = 5e-5  # DCFR target exploitability for the strategies graded here
_solved = {}


def solved(n: int, mode: str):
    """(game, tree, DCFR average strategy with exploitability <= TOL, its exploitability report)."""
    if (n, mode) not in _solved:
        game = BoardLeduc(num_ranks=n, streets=2, deck_mode=mode)
        tree = compile_game(game)
        solver = CFR(tree)
        solver.run(tol=TOL, log_every=10)
        sigma = solver.average_strategy()
        _solved[(n, mode)] = (game, tree, sigma, exploitability(tree, sigma))
    return _solved[(n, mode)]


def openspiel_grades(game, tree, sigma):
    """OpenSpiel's best-response value for each player and its on-policy values."""
    os_game = OpenSpielGame(game)
    pol = KeyPolicy(os_game, strategy_table(tree, sigma))
    brs = [os_expl.best_response(os_game, pol, p) for p in (0, 1)]
    return [b["best_response_value"] for b in brs], brs[0]["on_policy_values"]


def assert_same_grades(br, values, rep, tol=1e-9):
    assert br[0] == pytest.approx(rep.br_value[0], abs=tol)
    assert br[1] == pytest.approx(rep.br_value[1], abs=tol)
    assert values[0] == pytest.approx(rep.game_value, abs=tol)
    assert values[1] == pytest.approx(-rep.game_value, abs=tol)
    assert (br[0] + br[1]) / 2 == pytest.approx(rep.exploitability, abs=tol)


# OpenSpiel ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_openspiel_wrapper_walks_like_our_game(mode):
    game = BoardLeduc(num_ranks=3, streets=2, deck_mode=mode)
    mapping = lockstep(OpenSpielGame(game), game)
    assert all(info == key for info, (key, _) in mapping.items())
    assert len(mapping) == sum(compile_game(game).n_infosets)


@pytest.mark.parametrize("n, mode", [(4, "shared"), (4, "identical"),
                                     pytest.param(4, "independent", marks=pytest.mark.slow),
                                     pytest.param(5, "shared", marks=pytest.mark.slow),
                                     pytest.param(5, "independent", marks=pytest.mark.slow)])
def test_openspiel_certifies_our_equilibrium(n, mode):
    game, tree, sigma, rep = solved(n, mode)
    br, values = openspiel_grades(game, tree, sigma)
    assert_same_grades(br, values, rep)
    assert (br[0] + br[1]) / 2 <= TOL  # OpenSpiel's own exploitability of our strategy


@pytest.mark.parametrize("mode", MODES)
def test_openspiel_agrees_on_random_strategies(mode):
    # Far from equilibrium, so every best response has real work to do.
    game = BoardLeduc(num_ranks=3, streets=2, deck_mode=mode)
    tree = compile_game(game)
    for seed in (0, 1):
        sigma = strategy_from_table(tree, random_table(tree, seed))
        rep = exploitability(tree, sigma)
        assert rep.exploitability > 0.1
        assert_same_grades(*openspiel_grades(game, tree, sigma), rep)


@pytest.mark.parametrize("mode", MODES)
def test_openspiel_dcfr_iterates_match_ours(mode):
    game = BoardLeduc(num_ranks=3, streets=2, deck_mode=mode)
    tree = compile_game(game)
    os_game = OpenSpielGame(game)
    mapping = lockstep(os_game, game)
    os_solver = discounted_cfr.DCFRSolver(os_game)
    ours = CFR(tree, "dcfr")
    for _ in range(5):
        os_solver.evaluate_and_update_policy()
        ours.iteration()
    theirs = from_openspiel_policy(os_solver.average_policy(), mapping, tree)
    mine = strategy_table(tree, ours.average_strategy())
    assert theirs.keys() == mine.keys()
    for key, probs in theirs.items():
        for a, prob in probs.items():
            assert mine[key][a] == pytest.approx(prob, abs=1e-10), key


# Best response as a linear program ----------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_lp_best_response_matches_ours(mode):
    _, tree, sigma, _ = solved(4, mode)
    for prof in (sigma, random_profile(tree, 0)):
        rep = exploitability(tree, prof)
        for p in (0, 1):
            value, _ = best_response_lp(tree, p, prof[1 - p])
            assert value == pytest.approx(rep.br_value[p], abs=1e-7)


def test_lp_best_response_matches_ours_on_leduc():
    tree = compile_game(Leduc())
    for prof in (tree.uniform(), random_profile(tree, 3)):
        rep = exploitability(tree, prof)
        for p in (0, 1):
            assert best_response_lp(tree, p, prof[1 - p])[0] == pytest.approx(rep.br_value[p], abs=1e-9)


# Value brackets -------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_every_solver_brackets_the_lp_value(mode):
    # Any profile brackets the value: -BR1(sigma0) <= v* <= BR0(sigma1). Equilibria
    # aren't unique, so compare values, not strategies.
    _, tree, sigma, rep = solved(4, mode)
    (v_lp, v_lp1), lp_sigma = solve_game(tree)
    assert v_lp == pytest.approx(v_lp1, abs=1e-6)
    reps = {"dcfr": rep, "lp": exploitability(tree, lp_sigma)}
    for variant in ("cfr+", "cfr"):
        solver = CFR(tree, variant)
        solver.run(iterations=300, log_every=300)
        reps[variant] = exploitability(tree, solver.average_strategy())
    for name, r in reps.items():
        assert -r.br_value[1] - 1e-9 <= v_lp <= r.br_value[0] + 1e-9, name
        assert -r.br_value[1] - 1e-12 <= r.game_value <= r.br_value[0] + 1e-12, name
    assert reps["lp"].exploitability < 1e-8


# Dominated actions --------------------------------------------------------------------


def test_audit_finds_kuhns_dominated_actions():
    tree = compile_game(Kuhn())
    found = {(d.key, d.action, d.better) for d in dominated_actions(tree, tree.uniform())}
    # Facing a bet, calling with the J and folding the K are dominated, for both players.
    assert found == {("0:J::cr", "c", "f"), ("0:K::cr", "f", "c"), ("1:J::r", "c", "f"), ("1:K::r", "f", "c")}
    _, sigma = solve_game(tree)
    rep = exploitability(tree, sigma)
    a = audit(tree, sigma, rep.br_value, rep.game_value)
    assert a["ok"] and a["p0"]["bound_sum"] < 1e-12 and a["p1"]["bound_sum"] < 1e-12


@pytest.mark.parametrize("mode", MODES)
def test_audit_stays_within_best_response_gain(mode):
    _, tree, sigma, rep = solved(4, mode)
    a = audit(tree, sigma, rep.br_value, rep.game_value)
    assert a["ok"] and a["p0"]["dominated"] > 0
    # Force the most-reached dominated action: the bound becomes reach x gap, and our
    # best response must gain at least that much.
    worst = max(dominated_actions(tree, sigma), key=lambda d: d.reach)
    assert worst.reach > 1e-4
    p = worst.player
    i = tree.infoset_index(p)[worst.key]
    f, acts = tree.first_seq[p][i], tree.actions[p][i]
    bad = [s.copy() for s in sigma]
    bad[p][f:f + len(acts)] = [float(x == worst.action) for x in acts]
    r = exploitability(tree, bad)
    b = audit(tree, bad, r.br_value, r.game_value)
    assert b[f"p{p}"]["bound_sum"] >= worst.reach * worst.gap * (1 - 1e-9)
    assert b["ok"]


# Board swap -------------------------------------------------------------------------------


class BoardAWorthMore(BoardLeduc):
    """A lopsided game: board A pays 60% of the pot and board B 40%."""

    def returns(self, s):
        if s.folder >= 0:
            return super().returns(s)
        a, b = self._shares(s.hands[0], s.hands[1], s.boards)
        u0 = (s.contrib[0] + s.contrib[1]) * (0.6 * a + 0.4 * b) - s.contrib[0]
        return (u0, -u0)


def test_swap_key():
    assert swap_key("0:1:23|40:cc/r") == "0:1:40|23:cc/r"


@pytest.mark.parametrize("mode", MODES)
def test_board_swap_keeps_exploitability(mode):
    _, tree, sigma, _ = solved(4, mode)
    for prof in (sigma, random_profile(tree, 0), random_profile(tree, 1)):
        a, b = exploitability(tree, prof), exploitability(tree, swap_boards(tree, prof))
        assert abs(a.exploitability - b.exploitability) <= 1e-12
        assert abs(a.game_value - b.game_value) <= 1e-12
        assert np.allclose(a.br_value, b.br_value, rtol=0, atol=1e-12)
    if mode == "identical":  # the boards are always equal, so the swap is the identity
        assert all(np.array_equal(x, y) for x, y in zip(swap_boards(tree, sigma), sigma))


def test_board_swap_catches_a_lopsided_game():
    tree = compile_game(BoardAWorthMore(num_ranks=4, streets=2))
    prof = random_profile(tree, 0)
    a, b = exploitability(tree, prof), exploitability(tree, swap_boards(tree, prof))
    assert abs(a.game_value - b.game_value) > 1e-4


# Saved strategies -----------------------------------------------------------------------


def test_saved_strategy_loads_back_exactly(tmp_path):
    _, tree, sigma, rep = solved(4, "shared")
    path = tmp_path / "strategy.npz"
    save_strategy(path, tree, sigma, solver="dcfr", exploitability=rep.exploitability)
    saved = load_strategy(path)
    assert saved.info["game"] == tree.name and saved.info["solver"] == "dcfr"
    assert all(np.array_equal(x, y) for x, y in zip(saved.sigma(tree), sigma))
    assert saved.table() == strategy_table(tree, sigma)
    # A freshly compiled tree of the same game gets the same strategy through the keys.
    fresh = compile_game(BoardLeduc(num_ranks=4, streets=2))
    assert exploitability(fresh, saved.sigma(fresh)).exploitability == pytest.approx(rep.exploitability, abs=1e-15)
    with pytest.raises(ValueError):
        saved.sigma(solved(4, "identical")[1])
