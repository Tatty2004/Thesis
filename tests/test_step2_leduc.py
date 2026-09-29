"""Correctness ladder, step 2: Leduc against OpenSpiel's leduc_poker."""
import numpy as np
import pytest

from helpers import from_openspiel_policy, lockstep, naive_value, random_table, to_openspiel_policy
from toygames.eval.exploitability import exploitability
from toygames.games import Leduc
from toygames.games.base import CHANCE
from toygames.solvers.cfr import CFR
from toygames.solvers.lp import solve_game
from toygames.tree import compile_game, profile_value, strategy_from_table, strategy_table

pyspiel = pytest.importorskip("pyspiel")
from open_spiel.python.algorithms import discounted_cfr  # noqa: E402
from open_spiel.python.algorithms import exploitability as os_expl  # noqa: E402
from open_spiel.python.algorithms import expected_game_score  # noqa: E402


@pytest.fixture(scope="module")
def game():
    return Leduc()


@pytest.fixture(scope="module")
def tree(game):
    return compile_game(game)


@pytest.fixture(scope="module")
def os_iso():
    return pyspiel.load_game("leduc_poker", {"suit_isomorphism": True})


@pytest.fixture(scope="module")
def os_cards():
    return pyspiel.load_game("leduc_poker")


@pytest.fixture(scope="module")
def map_iso(os_iso, game):
    return lockstep(os_iso, game)


@pytest.fixture(scope="module")
def map_cards(os_cards, game):
    # OpenSpiel card c has rank c // 2 (J1 J2 Q1 Q2 K1 K2).
    return lockstep(os_cards, game, outcome_map=lambda c: c // 2)


@pytest.fixture(scope="module")
def lp(tree):
    return solve_game(tree)


def os_value(os_game, pol) -> float:
    return expected_game_score.policy_value(os_game.new_initial_state(), [pol, pol])[0]


def test_same_game_as_openspiel(tree, map_iso, map_cards):
    # lockstep() already checked actions, chance probabilities and payoffs node by node.
    # Rank-level keys give exactly OpenSpiel's suit-isomorphic infosets...
    assert len(map_iso) == 288
    assert len({key for key, _ in map_iso.values()}) == 288
    assert sum(tree.n_infosets) == 288
    assert tree.n_terminals == 1116
    # ...and collapse OpenSpiel's 936 card-level infosets onto the same 288.
    assert len(map_cards) == 936
    assert {key for key, _ in map_cards.values()} == {key for key, _ in map_iso.values()}


def test_uniform_exploitability_matches_openspiel(tree, os_iso, os_cards):
    from open_spiel.python import policy as os_policy

    ours = exploitability(tree, tree.uniform()).exploitability
    for os_game in (os_iso, os_cards):
        assert ours == pytest.approx(os_expl.exploitability(os_game, os_policy.UniformRandomPolicy(os_game)),
                                     abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_same_strategy_value_and_exploitability(game, tree, os_iso, os_cards, map_iso, map_cards, seed):
    table = random_table(tree, seed)
    sigma = strategy_from_table(tree, table)
    rep = exploitability(tree, sigma)
    assert rep.game_value == pytest.approx(naive_value(game, table), abs=1e-12)
    for os_game, mapping in ((os_iso, map_iso), (os_cards, map_cards)):
        pol = to_openspiel_policy(os_game, mapping, tree, table)
        assert rep.exploitability == pytest.approx(os_expl.exploitability(os_game, pol), abs=1e-10)
        assert rep.game_value == pytest.approx(os_value(os_game, pol), abs=1e-10)


def test_lp_value_and_equilibrium_match_openspiel(tree, os_iso, os_cards, map_iso, map_cards, lp):
    (v0, v1), sigma = lp
    assert v0 == pytest.approx(v1, abs=1e-6)
    assert v0 == pytest.approx(-0.0856064, abs=1e-6)  # published Leduc value
    table = strategy_table(tree, sigma)
    for os_game, mapping in ((os_iso, map_iso), (os_cards, map_cards)):
        pol = to_openspiel_policy(os_game, mapping, tree, table)
        assert os_expl.exploitability(os_game, pol) < 1e-9
        assert os_value(os_game, pol) == pytest.approx(v0, abs=1e-6)


def test_dcfr_matches_lp(tree, lp):
    (v_lp, _), _ = lp
    rows = CFR(tree, "dcfr").run(iterations=3000, log_every=500)
    last = rows[-1]
    assert last["exploitability"] < 5e-5
    # Any profile brackets the game value: -BR1(sigma0) <= v* <= BR0(sigma1).
    for r in rows:
        assert -r["br_value_p1"] - 1e-9 <= v_lp <= r["br_value_p0"] + 1e-9
    assert last["game_value"] == pytest.approx(v_lp, abs=2 * last["exploitability"])


def test_dcfr_iterates_match_openspiel(tree, os_iso, map_iso):
    os_solver = discounted_cfr.DCFRSolver(os_iso)
    ours = CFR(tree, "dcfr")
    for _ in range(5):
        os_solver.evaluate_and_update_policy()
        ours.iteration()
    theirs = from_openspiel_policy(os_solver.average_policy(), map_iso, tree)
    mine = strategy_table(tree, ours.average_strategy())
    for key, probs in theirs.items():
        for a, prob in probs.items():
            assert mine[key][a] == pytest.approx(prob, abs=1e-10), key


# Infoset keys ---------------------------------------------------------------------


def decision_states(game):
    stack = [game.initial_state()]
    while stack:
        s = stack.pop()
        if game.is_terminal(s):
            continue
        if game.current_player(s) == CHANCE:
            stack.extend(game.next_state(s, o) for o, _ in game.chance_outcomes(s))
        else:
            yield s
            stack.extend(game.next_state(s, a) for a in game.legal_actions(s))


def test_keys_ranks_only(map_cards):
    # Card-level states that differ only in suits share a key (lockstep checked the
    # mapping is consistent), and keys spell cards as ranks.
    for key, _ in map_cards.values():
        _, own, board, _ = key.split(":")
        assert own in "JQK" and len(own) == 1 and all(c in "JQK" for c in board)


def test_keys_never_hold_opponent_card(game):
    for s in decision_states(game):
        p = game.current_player(s)
        o = 1 - p
        for r in range(3):
            hands = list(s.hands)
            hands[o] = r
            alt = s._replace(hands=tuple(hands))
            used = alt.hands + alt.boards[0]
            if max(used.count(x) for x in range(3)) > 2:
                continue  # not a possible deal
            assert game.infoset_key(alt, p) == game.infoset_key(s, p)


def test_keys_hold_full_history(game, tree):
    # Round-2 keys carry the round-1 betting and the board card.
    for p in (0, 1):
        for key in tree.keys[p]:
            _, _, board, hist = key.split(":")
            if board:
                first, second = hist.split("/")
                assert first in ("cc", "rc", "crc", "rrc", "crrc")
    # Dropping round-1 history merges infosets, which OpenSpiel's count would catch.
    stripped = {(p, k.rsplit(":", 1)[0] + ":" + k.rsplit(":", 1)[1].split("/")[-1])
                for p in (0, 1) for k in tree.keys[p]}
    assert len(stripped) < 288
