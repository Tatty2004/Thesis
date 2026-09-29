"""Correctness ladder, steps 3 and 4: Board-Leduc with one and two streets."""
from collections import defaultdict

import numpy as np
import pytest

from toygames.eval.exploitability import exploitability
from toygames.games import BoardLeduc
from toygames.games.base import CHANCE, settle
from toygames.solvers.cfr import CFR
from toygames.solvers.lp import solve_game
from toygames.tree import compile_game, strategy_table

MODES = ["shared", "independent", "identical"]
STEP3 = [dict(num_ranks=3, streets=1), dict(num_ranks=4, streets=1)]
CONFIGS = STEP3


def cfg_id(cfg):
    return f"N{cfg['num_ranks']}-s{cfg['streets']}"


_trees, _lps = {}, {}


def tree_of(game):
    if game.name not in _trees:
        _trees[game.name] = compile_game(game)
    return _trees[game.name]


def lp_of(game):
    if game.name not in _lps:
        _lps[game.name] = solve_game(tree_of(game))
    return _lps[game.name]


def walk(game):
    """Every state of the game, depth first."""
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


def play(game, hands, boards, actions):
    """Deal `hands` and each board's cards in order, bet `actions` (one string per street)."""
    s = game.initial_state()
    dealt = [0] * game.num_boards
    while not game.is_terminal(s):
        if game.current_player(s) == CHANCE:
            kind, j = game.events[s.step]
            card = hands[j] if kind == "hand" else boards[j][dealt[j]]
            if kind == "board":
                dealt[j] += 1
            assert card in dict(game.chance_outcomes(s)), "impossible deal"
            s = game.next_state(s, card)
        else:
            t = game.street(s)
            s = game.next_state(s, actions[t][len(s.hist[t])])
    return s


# Hand-checked payoffs ---------------------------------------------------------------


@pytest.mark.parametrize("hands, boards, actions, expected", [
    ((3, 1), ((0,), (2,)), ["rc"], 3.0),      # scoop: high card wins both boards
    ((3, 0), ((1,), (2,)), ["cc"], 1.0),      # checked-down scoop wins the ante
    ((1, 3), ((1,), (0,)), ["crc"], 0.0),     # split: P0 pairs board A, P1's 3 takes board B
    ((2, 2), ((0,), (1,)), ["rrc"], 0.0),     # chopped: equal ranks tie both boards
    ((3, 1), ((0,), (2,)), ["rf"], 1.0),      # P1 folds to a bet, loses the ante
    ((3, 1), ((0,), (2,)), ["crf"], -1.0),    # P0 folds to a bet
    ((3, 1), ((0,), (2,)), ["rrf"], -3.0),    # P0 bets, folds to the raise
    ((3, 1), ((0,), (2,)), ["crrf"], 3.0),    # P1 bets, folds to the raise
])
def test_payoffs_one_street(hands, boards, actions, expected):
    g = BoardLeduc(num_ranks=4, num_boards=2, streets=1)
    assert g.returns(play(g, hands, boards, actions)) == (expected, -expected)


@pytest.mark.parametrize("hands, boards, actions, expected", [
    ((3, 0), ((1, 2), (2, 1)), ["crrc", "rrc"], 13.0),  # biggest pot: 1 + 4 + 8 each
    ((0, 3), ((1, 0), (2, 1)), ["rc", "rc"], 0.0),      # P0 pairs board A on the turn: split
    ((1, 2), ((0, 3), (3, 1)), ["cc", "rc"], 0.0),      # P1 wins A with 2 high, P0 pairs B
    ((3, 1), ((0, 2), (2, 0)), ["rc", "rf"], 3.0),      # P1 folds on street 2 after calling street 1
    ((3, 1), ((0, 2), (2, 0)), ["rc", "crf"], -3.0),    # P0 folds on street 2
    ((1, 1), ((0, 2), (3, 0)), ["rrc", "rrc"], 0.0),    # chopped both boards
])
def test_payoffs_two_streets(hands, boards, actions, expected):
    g = BoardLeduc(num_ranks=4, num_boards=2, streets=2)
    assert g.returns(play(g, hands, boards, actions)) == (expected, -expected)


def test_one_board_and_identical_payoffs():
    one = BoardLeduc(num_ranks=4, num_boards=1, streets=1)
    assert one.returns(play(one, (0, 3), ((0,),), ["rc"])) == (3.0, -3.0)  # pair beats high card
    same = BoardLeduc(num_ranks=4, num_boards=2, streets=1, deck_mode="identical")
    s = play(same, (0, 3), ((0,), (0,)), ["rc"])
    assert s.boards == ((0,), (0,))
    assert same.returns(s) == (3.0, -3.0)


def test_deck_modes_block_differently():
    # P0 holds a 0 and both boards show a 0: impossible from one deck, fine from two copies.
    shared = BoardLeduc(num_ranks=4, num_boards=2, streets=1, deck_mode="shared")
    with pytest.raises(AssertionError, match="impossible deal"):
        play(shared, (0, 1), ((0,), (0,)), ["cc"])
    indep = BoardLeduc(num_ranks=4, num_boards=2, streets=1, deck_mode="independent")
    assert indep.returns(play(indep, (0, 1), ((0,), (0,)), ["rc"])) == (3.0, -3.0)


def test_tie_splits_only_that_board():
    assert settle((3.0, 3.0), [1.0, 0.5]) == 1.5    # win one board, chop the other
    assert settle((3.0, 3.0), [0.5, 0.5]) == 0.0
    assert settle((3.0, 3.0), [1.0, 0.0]) == 0.0
    assert settle((5.0, 5.0), [0.0, 0.5]) == -2.5
    assert settle((3.0, 3.0), [1.0]) == 3.0


def test_independent_boards_do_not_block_each_other():
    g = BoardLeduc(num_ranks=4, num_boards=2, streets=2, deck_mode="independent")
    s = g.initial_state()
    for card in (0, 1, 2):  # P0 gets a 0, P1 a 1, board A a 2
        s = g.next_state(s, card)
    assert g.events[s.step] == ("board", 1)
    probs = dict(g.chance_outcomes(s))
    # Board B's copy is the deck minus both private cards: board A's 2 doesn't matter.
    assert probs == pytest.approx({0: 1 / 6, 1: 1 / 6, 2: 2 / 6, 3: 2 / 6})


# Structural checks -----------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cfg", CONFIGS, ids=cfg_id)
def test_chance_sums_to_one_and_zero_sum(cfg, mode):
    g = BoardLeduc(num_boards=2, deck_mode=mode, **cfg)
    for s in walk(g):
        if g.is_terminal(s):
            assert sum(g.returns(s)) == 0.0
            if s.folder < 0:
                assert s.contrib[0] == s.contrib[1]
        elif g.current_player(s) == CHANCE:
            assert sum(q for _, q in g.chance_outcomes(s)) == pytest.approx(1.0, abs=1e-12)
    for t in range(g.num_streets):
        assert sum(q for *_, q in g.deals(t)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("cfg", CONFIGS, ids=cfg_id)
def test_identical_reproduces_one_board(cfg):
    one = BoardLeduc(num_boards=1, **cfg)
    same = BoardLeduc(num_boards=2, deck_mode="identical", **cfg)
    t1, t2 = tree_of(one), tree_of(same)
    assert t1.n_infosets == t2.n_infosets and t1.n_terminals == t2.n_terminals
    for p in (0, 1):
        np.testing.assert_array_equal(t1.term_seq[p], t2.term_seq[p])
        np.testing.assert_array_equal(t1.parent_seq[p], t2.parent_seq[p])
        # Same infosets in the same order: board B is just a copy of board A.
        for k1, k2 in zip(t1.keys[p], t2.keys[p]):
            p1, own1, b1, h1 = k1.split(":")
            p2, own2, b2, h2 = k2.split(":")
            assert (p1, own1, h1) == (p2, own2, h2) and b2 == f"{b1}|{b1}"
    np.testing.assert_array_equal(t1.term_cu0, t2.term_cu0)
    assert lp_of(one)[0][0] == pytest.approx(lp_of(same)[0][0], abs=1e-9)
    s1, s2 = CFR(t1), CFR(t2)
    for _ in range(100):
        s1.iteration()
        s2.iteration()
    for a, b in zip(s1.average_strategy(), s2.average_strategy()):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mode", ["shared", "independent"])
@pytest.mark.parametrize("cfg", CONFIGS, ids=cfg_id)
def test_swapping_boards_keeps_value(cfg, mode):
    g = BoardLeduc(num_boards=2, deck_mode=mode, **cfg)
    swapped = BoardLeduc(num_boards=2, deck_mode=mode, board_order=(1, 0), **cfg)
    assert lp_of(g)[0][0] == pytest.approx(lp_of(swapped)[0][0], abs=1e-9)
    # The deal is exchangeable between boards and so is every showdown.
    for t in range(g.num_streets):
        dist = defaultdict(float)
        for board, h0, h1, q in g.deals(t):
            dist[(board, h0, h1)] += q
        for (board, h0, h1), q in dist.items():
            mirror = tuple(board[i ^ 1] for i in range(len(board)))  # (a1, b1, a2, b2) -> (b1, a1, b2, a2)
            assert dist[(mirror, h0, h1)] == pytest.approx(q, abs=1e-15)
            if t == g.num_streets - 1:
                np.testing.assert_array_equal(g.showdown(board, h0, h1), g.showdown(mirror, h0, h1)[::-1])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cfg", CONFIGS, ids=cfg_id)
def test_dcfr_matches_lp(cfg, mode):
    g = BoardLeduc(num_boards=2, deck_mode=mode, **cfg)
    tree = tree_of(g)
    (v0, v1), sigma = lp_of(g)
    assert v0 == pytest.approx(v1, abs=1e-6)
    assert exploitability(tree, sigma).exploitability < 1e-8
    rows = CFR(tree, "dcfr").run(iterations=4000, log_every=500)
    for r in rows:  # every profile brackets the game value
        assert -r["br_value_p1"] - 1e-9 <= v0 <= r["br_value_p0"] + 1e-9
    last = rows[-1]
    assert last["exploitability"] < 1e-4
    assert last["game_value"] == pytest.approx(v0, abs=2 * last["exploitability"] + 1e-12)


def test_value_zero_when_no_hand_can_both_win_and_lose():
    # N = 3, two shared boards, one street: every (hand, board) either never wins or
    # never loses, so betting can't extract value and the value is the check-down value, 0.
    for bets in [(2.0,), (1.0,), (5.0,)]:
        g = BoardLeduc(num_ranks=3, num_boards=2, streets=1, bet_sizes=bets)
        assert lp_of(g)[0][0] == pytest.approx(0.0, abs=1e-12)


def shared_sizes(n, streets):
    """Closed-form infoset and terminal counts, two boards, shared deck."""
    from itertools import product

    def feasible(k):  # rank tuples of length k using each rank at most twice
        return sum(1 for t in product(range(n), repeat=k) if max(t.count(r) for r in set(t)) <= 2)

    if streets == 1:
        return 6 * feasible(3), 9 * feasible(4)
    return 6 * feasible(3) + 30 * feasible(5), 4 * feasible(4) + 45 * feasible(6)


@pytest.mark.parametrize("cfg", CONFIGS, ids=cfg_id)
def test_sizes(cfg):
    tree = tree_of(BoardLeduc(num_boards=2, **cfg))
    assert (sum(tree.n_infosets), tree.n_terminals) == shared_sizes(cfg["num_ranks"], cfg["streets"])


# Infoset keys ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cfg", CONFIGS, ids=cfg_id)
def test_keys(cfg, mode):
    g = BoardLeduc(num_boards=2, deck_mode=mode, **cfg)
    n = cfg["num_ranks"]
    for s in walk(g):
        if g.is_terminal(s) or g.current_player(s) == CHANCE:
            continue
        p = g.current_player(s)
        key = g.infoset_key(s, p)
        player, own, boards, hist = key.split(":")
        # Ranks only: one rank symbol for the own card and for every board card.
        assert own == g.rank_str(s.hands[p])
        assert boards == "|".join("".join(g.rank_str(c) for c in b) for b in s.boards)
        # Full history from every street.
        assert hist == "/".join(s.hist) and len(hist.split("/")) == g.street(s) + 1
        # Never the opponent's card: any other possible opponent card gives the same key.
        for r in range(n):
            hands = list(s.hands)
            hands[1 - p] = r
            alt = s._replace(hands=tuple(hands))
            if _possible(g, alt):
                assert g.infoset_key(alt, p) == key


def _possible(g, s):
    """Whether the cards in s could have been dealt in this deck mode."""
    for j in range(g.num_boards):
        used = s.hands + (sum(s.boards, ()) if g.deck_mode == "shared" else s.boards[j])
        if any(used.count(r) > 2 for r in range(g.num_ranks)):
            return False
    return True
