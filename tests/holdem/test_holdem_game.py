"""R1: the Hold'em game on its own, and graded by OpenSpiel.

Hand-checked pots on the real 52-card deck, then structural checks on tiny decks:
chance and zero-sum, `identical` reproduces one board, swapping boards changes nothing,
a bomb pot is the average of its flop-pair games, DCFR matches the LP, infoset keys,
and OpenSpiel's own best response and expected value reproduce ours.
"""
from collections import defaultdict

import numpy as np
import pytest

from bombpot.core.eval.exploitability import exploitability
from bombpot.core.game import CHANCE
from bombpot.core.solvers.cfr import CFR
from bombpot.core.solvers.lp import solve_game
from bombpot.core.tree import compile_game, strategy_table
from bombpot.experiments.certify import random_profile, swap_boards
from bombpot.holdem import Holdem

_trees, _lps = {}, {}


def tree_of(game):
    if game.name not in _trees:
        _trees[game.name] = compile_game(game)
    return _trees[game.name]


def lp_of(game):
    if game.name not in _lps:
        _lps[game.name] = solve_game(tree_of(game))
    return _lps[game.name]


def tiny(n=3, s=2, cards=(1,), boards=2, mode="shared", **kw):
    return Holdem(num_ranks=n, num_suits=s, board_cards=cards, num_boards=boards, deck_mode=mode, **kw)


def play(game, h0, h1, streets, actions):
    """Deal hole cards h0, h1, then for each street every board's cards (`streets[t][j]`,
    like "2c7h9d"), and bet `actions` (one string per street)."""
    s = game.initial_state()
    deals = [h0, h1] + [b for street in streets for b in (street[:1] if game.deck_mode == "identical" else street)]
    while not game.is_terminal(s):
        if game.current_player(s) == CHANCE:
            s = game.next_state(s, game.outcome(deals.pop(0)))
        else:
            t = game.street(s)
            s = game.next_state(s, actions[t][len(s.hist[t])])
    return s


def walk(game):
    stack = [game.initial_state()]
    while stack:
        s = stack.pop()
        yield s
        if game.is_terminal(s):
            continue
        if game.current_player(s) == CHANCE:
            stack.extend(game.next_state(s, o) for o, _ in game.chance_outcomes(s))
        else:
            stack.extend(game.next_state(s, a) for a in game.legal_actions(s))


# Hand-checked pots, 52 cards, flop and turn ---------------------------------------------

REAL = Holdem()  # 52 cards, two boards, flop + turn, ante 1, bets 2 then 4, a bet and a raise per street


@pytest.mark.parametrize("h0, h1, streets, actions, expected", [
    # Aces beat kings on both boards: scoop. 1 + 2 + 4 each.
    ("AsAd", "KsKd", [("2c7h9d", "4c8hTd"), ("3s", "5s")], ["rc", "rc"], 7.0),
    # Different hole cards hit different boards: P0's ace pairs on A, P1's kings hold on B. Split.
    ("Ah2d", "KcKh", [("As7c8d", "Qh5c6s"), ("9s", "3d")], ["cc", "rc"], 0.0),
    # One hole card each board: Ah pairs on A, Kd pairs on B, P1's queens lose both.
    ("AhKd", "QsQd", [("Ac7c2h", "Kc8h3s"), ("4d", "5h")], ["rc", "cc"], 3.0),
    # Board A is quad aces: both play A A A A K, chopped. P1's trips take board B. Pot 6.
    ("Kd2c", "KsQd", [("AcAdAh", "QhQc4s"), ("As", "9d")], ["rc", "cc"], -1.5),
    # The biggest pot: bet, raise, call on both streets (5 then 13 each). P0 scoops.
    ("AsAd", "KsKd", [("2c7h9d", "4c8hTd"), ("3s", "5s")], ["crrc", "crrc"], 13.0),
    # Folds: P1 folds to a bet (loses the ante); P1 bets, P0 raises, P1 folds (loses 3).
    ("2c3d", "AsAh", [("KcQcJd", "9h8h7s"), ("2h", "3h")], ["rf", ""], 1.0),
    ("2c3d", "AsAh", [("KcQcJd", "9h8h7s"), ("2h", "3h")], ["crrf", ""], 3.0),
    # Street-2 fold after a called street-1 bet: P0 folds and loses 3.
    ("2c3d", "AsAh", [("KcQcJd", "9h8h7s"), ("2h", "3h")], ["rc", "crf"], -3.0),
    # Flush on A beats a straight; on B the straight (wheel) beats trips.
    ("Ah2h", "6c6d", [("Kh9h3h", "3c4d6h"), ("Qc", "5s")], ["cc", "cc"], 1.0),
])
def test_hand_checked_pots(h0, h1, streets, actions, expected):
    s = play(REAL, h0, h1, streets, actions)
    assert REAL.returns(s) == (expected, -expected)


def test_board_cards_play_for_both_players():
    # A paired board gives both players the pair; kickers decide (unlike Board-Leduc).
    s = play(REAL, "AhKd", "QsJd", [("8c8d2h", "8h8s2c"), ("3s", "3d")], ["cc", "cc"])
    assert REAL.returns(s) == (1.0, -1.0)  # 8 8 3 3 A beats 8 8 3 3 Q on both boards


def test_deals_block_and_flops_can_be_fixed():
    g = Holdem(flops=("2c7h9d", "4c8hTd"))
    s = g.initial_state()
    assert s.boards == (g.deck.parse("2c7h9d"), g.deck.parse("4c8hTd"))
    assert len(g.chance_outcomes(s)) == 46 * 45 // 2          # P0's hand from the 46 cards left
    s = g.next_state(s, g.outcome("AsAd"))
    assert len(g.chance_outcomes(s)) == 44 * 43 // 2
    s = g.next_state(g.next_state(g.next_state(s, g.outcome("KsKd")), "c"), "c")
    assert len(g.chance_outcomes(s)) == 42                    # board A's turn
    s = g.next_state(s, g.outcome("3s"))
    assert g.outcome("3s") not in dict(g.chance_outcomes(s))  # board B can't repeat it
    assert len(g.chance_outcomes(s)) == 41
    with pytest.raises(ValueError):
        Holdem(flops=("2c7h9d", "2c8hTd"))                    # the flops share a card


# Structure, on tiny decks ----------------------------------------------------------------

STRUCTURE = [(3, 2, (1,)), (4, 2, (1,)), (4, 2, (2,)), (4, 2, (1, 1))]
sid = lambda c: f"{c[0]}x{c[1]}-{'+'.join(map(str, c[2]))}"  # noqa: E731


@pytest.mark.parametrize("cfg", STRUCTURE[:3], ids=sid)
@pytest.mark.parametrize("mode", ["shared", "identical"])
def test_chance_sums_to_one_and_zero_sum(cfg, mode):
    g = tiny(*cfg, mode=mode)
    for s in walk(g):
        if g.is_terminal(s):
            assert sum(g.returns(s)) == 0.0
        elif g.current_player(s) == CHANCE:
            assert sum(q for _, q in g.chance_outcomes(s)) == pytest.approx(1.0, abs=1e-12)
    for t in range(g.num_streets):
        assert sum(q for *_, q in g.deals(t)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("cfg", [STRUCTURE[0], (3, 2, (2,)), (3, 2, (1, 1))], ids=sid)
def test_identical_reproduces_one_board(cfg):
    one, same = tiny(*cfg, boards=1), tiny(*cfg, mode="identical")
    t1, t2 = tree_of(one), tree_of(same)
    assert t1.n_infosets == t2.n_infosets and t1.n_terminals == t2.n_terminals
    for p in (0, 1):
        np.testing.assert_array_equal(t1.term_seq[p], t2.term_seq[p])
        for k1, k2 in zip(t1.keys[p], t2.keys[p]):
            p1, own1, b1, h1 = k1.split(":")
            p2, own2, b2, h2 = k2.split(":")
            assert (p1, own1, h1) == (p2, own2, h2) and b2 == f"{b1}|{b1}"
    np.testing.assert_array_equal(t1.term_cu0, t2.term_cu0)
    assert lp_of(one)[0][0] == pytest.approx(lp_of(same)[0][0], abs=1e-9)


@pytest.mark.parametrize("cfg", STRUCTURE[:3], ids=sid)
def test_swapping_boards_changes_nothing(cfg):
    g = tiny(*cfg)
    tree = tree_of(g)
    # The deal is exchangeable between boards, and every showdown mirrors.
    for t in range(g.num_streets):
        dist = defaultdict(float)
        for board, h0, h1, q in g.deals(t):
            dist[(board, h0, h1)] += q
        for (board, h0, h1), q in dist.items():
            mirror = tuple(c for pair in zip(board[1::2], board[0::2]) for c in pair)  # (a1, b1, ...) -> (b1, a1, ...)
            assert dist[(mirror, h0, h1)] == pytest.approx(q, abs=1e-15)
            if t == g.num_streets - 1:
                np.testing.assert_array_equal(g.showdown(board, h0, h1), g.showdown(mirror, h0, h1)[::-1])
    # A strategy and its board-swapped twin are equally exploitable and worth the same.
    for sigma in [lp_of(g)[1], random_profile(tree, 0)]:
        a, b = exploitability(tree, sigma), exploitability(tree, swap_boards(tree, sigma))
        assert a.exploitability == pytest.approx(b.exploitability, abs=1e-12)
        assert a.game_value == pytest.approx(b.game_value, abs=1e-12)


@pytest.mark.parametrize("cfg, boards", [((3, 2, (1,)), 2), ((3, 2, (1, 1)), 1), ((4, 2, (1,)), 2)],
                         ids=["3x2-1-b2", "3x2-1+1-b1", "4x2-1-b2"])
def test_bomb_pot_is_the_average_of_its_flop_pair_games(cfg, boards):
    # Nobody acts before the flops, so the game splits into one independent game per
    # pair of flops, weighted by how likely that pair is.
    g = tiny(*cfg, boards=boards)
    flop_probs = defaultdict(float)
    for board, _, _, q in g.deals(0):  # board = the street-1 cards, boards interleaved
        flop_probs[board] += q
    total = 0.0
    for board, q in flop_probs.items():
        flops = tuple(tuple(board[j::boards]) for j in range(boards))
        total += q * lp_of(tiny(*cfg, boards=boards, flops=flops))[0][0]
    assert sum(flop_probs.values()) == pytest.approx(1.0, abs=1e-12)
    assert total == pytest.approx(lp_of(g)[0][0], abs=1e-9)


@pytest.mark.parametrize("cfg", [(3, 2, (1,), 2, "shared"), (4, 2, (1,), 2, "shared"), (3, 2, (1, 1), 2, "identical"),
                                 pytest.param((4, 2, (1, 1), 2, "shared"), marks=pytest.mark.slow)],
                         ids=["3x2-1-shared", "4x2-1-shared", "3x2-1+1-identical", "4x2-1+1-shared"])
def test_dcfr_matches_lp(cfg):
    g = tiny(*cfg[:3], boards=cfg[3], mode=cfg[4])
    tree = tree_of(g)
    (v0, v1), sigma = lp_of(g)
    assert v0 == pytest.approx(v1, abs=1e-6)
    assert exploitability(tree, sigma).exploitability < 1e-8
    rows = CFR(tree, "dcfr").run(iterations=3000, log_every=500)
    for r in rows:  # every profile brackets the game value
        assert -r["br_value_p1"] - 1e-9 <= v0 <= r["br_value_p0"] + 1e-9
    assert rows[-1]["exploitability"] < 1e-4
    assert rows[-1]["game_value"] == pytest.approx(v0, abs=2 * rows[-1]["exploitability"] + 1e-12)


# Infoset keys -------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [(4, 2, (1,)), (3, 2, (1, 1))], ids=sid)
def test_keys(cfg):
    g = tiny(*cfg, mode="shared" if cfg[2] == (1,) else "identical")
    by_key = defaultdict(set)
    for s in walk(g):
        if g.is_terminal(s) or g.current_player(s) == CHANCE:
            continue
        p = g.current_player(s)
        key = g.infoset_key(s, p)
        player, own, boards, hist = key.split(":")
        assert own == g.deck.cards_str(s.hands[p])
        assert boards == "|".join(g.deck.cards_str(b) for b in s.boards)  # each board in deal order
        assert hist == "/".join(s.hist) and len(hist.split("/")) == g.street(s) + 1
        by_key[key].add((s.hands[p], s.boards, s.hist))
    # A key pins down exactly the player's own cards, the boards street by street and the
    # betting: everything that varies within one key is the opponent's hand.
    assert all(len(v) == 1 for v in by_key.values())


# OpenSpiel's grading ----------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [(3, 2, (1,), 2, "shared"), (3, 2, (1, 1), 2, "identical"),
                                 pytest.param((4, 2, (1,), 2, "shared"), marks=pytest.mark.slow)],
                         ids=["3x2-1-shared", "3x2-1+1-identical", "4x2-1-shared"])
def test_openspiel_grades_like_us(cfg):
    pytest.importorskip("pyspiel")
    from open_spiel.python.algorithms import exploitability as os_expl
    from openspiel_game import KeyPolicy, OpenSpielGame

    g = tiny(*cfg[:3], boards=cfg[3], mode=cfg[4])
    tree = tree_of(g)
    solver = CFR(tree)
    solver.run(iterations=300, log_every=300)
    os_game = OpenSpielGame(g)
    for sigma in [solver.average_strategy(), random_profile(tree, 1)]:
        rep = exploitability(tree, sigma)
        pol = KeyPolicy(os_game, strategy_table(tree, sigma))
        brs = [os_expl.best_response(os_game, pol, p) for p in (0, 1)]
        assert brs[0]["best_response_value"] == pytest.approx(rep.br_value[0], abs=1e-9)
        assert brs[1]["best_response_value"] == pytest.approx(rep.br_value[1], abs=1e-9)
        assert brs[0]["on_policy_values"][0] == pytest.approx(rep.game_value, abs=1e-9)
