"""The no-limit / pot-limit engine (core/betting.py, PotPoker), hand-checked.

Pot and raise arithmetic, the minimum raise, all-in collapse, no betting after an
all-in, fold payoffs and pot-limit, on the 52-card Hold'em game with `betting="pot"`
(antes 1, so the pot starts at 2). Tiny-deck solver checks and the comparison with ACPC
are in tests/holdem/test_holdem_pot.py.
"""
import pytest

from bombpot.core.betting import ALLIN, PotPoker, size_label
from bombpot.core.game import CHANCE
from bombpot.holdem import Holdem


def game(**kw):
    kw = {"board_cards": (3, 1), "betting": "pot", **kw}
    return Holdem(**kw)


def street_state(g, actions=""):
    """After the hole cards (AsAd vs KsKd), the boards (2c7h9d / 4c8hTd, then 3s / 5s, then Qc / Jc) and
    the actions, a list of action strings across streets."""
    s = g.initial_state()
    deals = ["AsAd", "KsKd", "2c7h9d", "4c8hTd", "3s", "5s", "Qc", "Jc"]
    actions = list(actions)
    while not g.is_terminal(s):
        if g.current_player(s) == CHANCE:
            s = g.next_state(s, g.outcome(deals.pop(0)))
        elif actions:
            s = g.next_state(s, actions.pop(0))
        else:
            break
    assert not actions, "the hand ended before every action was played"
    return s


def test_labels_and_types():
    assert [size_label(x) for x in (0.5, 1.0, 0.75, 2.0, 1 / 3)] == ["r0.5", "r1", "r0.75", "r2", "r0.333333"]
    g = game()
    assert isinstance(g, PotPoker) and isinstance(g, Holdem)
    assert g.params["betting"] == "pot" and "pot" in g.name
    assert Holdem().name == ("holdem(num_ranks=13,num_suits=4,board_cards=[3, 1],num_boards=2,deck_mode=shared,"
                             "ante=1.0,bet_sizes=[2.0, 4.0],max_raises=[2, 2])")  # the limit game is unchanged
    assert not isinstance(Holdem(), PotPoker)


def test_pot_and_raise_arithmetic():
    g = game()
    s = street_state(g)
    assert g.legal_actions(s) == ("c", "r0.5", "r1")
    assert street_state(g, ["r0.5"]).contrib == (2.0, 1.0)         # half of 2
    assert street_state(g, ["r1"]).contrib == (3.0, 1.0)           # all of 2
    # Raise pot over a pot bet: call to 3, the pot is 6, raise by 6: 9.
    assert street_state(g, ["r1", "r1"]).contrib == (3.0, 9.0)
    # Raise half pot over a half-pot bet: call to 2, pot 4, raise by 2: 4.
    assert street_state(g, ["r0.5", "r0.5"]).contrib == (2.0, 4.0)
    # Check, bet, raise: player 1 bets half (2), player 0 raises pot (call 2, pot 4, +4): 6.
    assert street_state(g, ["c", "r0.5", "r1"]).contrib == (6.0, 2.0)
    # A call closes the street; the turn's pot fractions use the new pot (18 after r1 r1 c).
    s = street_state(g, ["r1", "r1", "c"])
    assert s.contrib == (9.0, 9.0) and s.hist == ("r1r1c", "")
    assert street_state(g, ["r1", "r1", "c", "r0.5"]).contrib == (18.0, 9.0)
    assert street_state(g, ["r1", "r1", "c", "c", "r1"]).contrib == (9.0, 27.0)
    # Two aggressive actions per street: after a bet and a raise only fold and call.
    assert g.legal_actions(street_state(g, ["r1", "r1"])) == ("f", "c")
    assert g.legal_actions(street_state(g, ["c", "r1", "r0.5"])) == ("f", "c")
    # Fold only when facing a bet; check, check closes; a check is never a fold.
    assert g.legal_actions(street_state(g, ["c"])) == ("c", "r0.5", "r1")
    assert street_state(g, ["c", "c"]).hist == ("cc", "")
    with pytest.raises(ValueError):
        g.next_state(street_state(g, ["c"]), "f")
    with pytest.raises(ValueError):
        g.next_state(street_state(g, ["r1", "r1"]), "r1")  # over the cap


def test_cap_per_street():
    g = game(max_aggressive=[3, 1])
    assert g.legal_actions(street_state(g, ["r0.5", "r0.5"])) == ("f", "c", "r0.5", "r1")
    assert g.legal_actions(street_state(g, ["r0.5", "r0.5", "r0.5"])) == ("f", "c")
    assert g.legal_actions(street_state(g, ["c", "c", "r1"])) == ("f", "c")


def test_minimum_raise():
    # Sizes a quarter pot and pot: after a pot bet of 2, a quarter-pot raise (call to 3, pot
    # 6, raise by 1.5) adds less than the 2 it faces, so it is illegal.
    g = game(sizes=(0.25, 1.0))
    s = street_state(g, ["r1"])
    assert g.legal_actions(s) == ("f", "c", "r1")
    with pytest.raises(ValueError):
        g.next_state(s, "r0.25")
    # After a quarter-pot bet (0.5), a quarter-pot raise adds 0.25 x 3 = 0.75 >= 0.5: legal.
    s = street_state(g, ["r0.25"])
    assert g.legal_actions(s) == ("f", "c", "r0.25", "r1")
    assert g.next_state(s, "r0.25").contrib == (1.5, 2.25)


def test_all_in_collapse():
    # 2 behind (3 in all): a half-pot bet (to 2) is a bet, a pot bet (to 3) is all-in.
    g = game(stack=2)
    assert g.legal_actions(street_state(g)) == ("c", "r0.5", ALLIN)
    assert street_state(g, ["a"]).contrib == (3.0, 1.0)
    # Over the half-pot bet, both raises reach the stack: one "a".
    assert g.legal_actions(street_state(g, ["r0.5"])) == ("f", "c", ALLIN)
    # Facing an all-in: fold or call only.
    assert g.legal_actions(street_state(g, ["a"])) == ("f", "c")
    # Half a chip behind: every size is all-in, and "a" appears once.
    g = game(stack=0.5)
    assert g.legal_actions(street_state(g)) == ("c", ALLIN)
    # A short all-in below the minimum raise is allowed (only "a" may be): 3 behind (4 in all),
    # sizes a quarter pot and pot, after a pot bet to 3 the quarter-pot raise (to 4.5) is all-in
    # to 4, adding 1 where a raise must add 2.
    g = game(stack=3, sizes=(0.25, 1.0))
    s = street_state(g, ["r1"])
    assert g.legal_actions(s) == ("f", "c", ALLIN)
    assert g.next_state(s, "a").contrib == (3.0, 4.0)
    # No separate shove: 100 behind, nothing reaches the stack on the flop, so no "a".
    assert ALLIN not in game().legal_actions(street_state(game()))


def test_no_betting_after_all_in():
    g = game(stack=2, board_cards=(3, 1, 1))
    s = street_state(g, ["a", "c"])  # the turn and river are dealt with no betting
    assert g.is_terminal(s)
    assert s.hist == ("ac", "", "") and s.contrib == (3.0, 3.0)
    # The deals still happen (5 board cards each), and AA scoops KK on 2c7h9d3s.. / 4c8hTd5s..
    assert all(len(b) == 5 for b in s.boards)
    assert g.returns(s) == (3.0, -3.0)
    key = g.infoset_key(street_state(g, ["c", "a"]), 0)
    assert key.endswith(":ca")
    # An all-in on the turn after a calm flop: the river has no betting.
    g = game(stack=5, board_cards=(3, 1, 1))
    s = street_state(g, ["r1", "c", "a", "c"])  # flop 3 each (pot 6), turn pot bet to 9 >= 6: all-in
    assert g.is_terminal(s) and s.hist == ("r1c", "ac", "") and s.contrib == (6.0, 6.0)


def test_fold_payoffs():
    g = game()
    s = street_state(g, ["r1", "f"])           # player 1 folds its ante
    assert g.is_terminal(s) and g.returns(s) == (1.0, -1.0)
    s = street_state(g, ["r1", "r1", "f"])     # player 0 bet 2 more, then folds: loses 3
    assert g.returns(s) == (-3.0, 3.0)
    s = street_state(g, ["c", "r0.5", "c", "c", "r1", "f"])  # flop 2 each, turn bet 4 folded
    assert g.returns(s) == (-2.0, 2.0)
    s = street_state(g, ["r0.5", "c", "c", "c"])  # showdown, pot 4: AA scoops KK
    assert g.is_terminal(s) and g.returns(s) == (2.0, -2.0)
    g = game(stack=2)
    s = street_state(g, ["a", "f"])
    assert g.returns(s) == (1.0, -1.0)


def test_pot_limit():
    g = game(sizes=(0.5, 2.0), limit="pot-limit")
    assert g.sizes == (0.5, 1.0) and g.legal_actions(street_state(g)) == ("c", "r0.5", "r1")
    g = game(sizes=(0.5, 2.0))  # no-limit keeps the two-pot overbet: to 1 + 4
    assert g.legal_actions(street_state(g)) == ("c", "r0.5", "r2")
    assert street_state(g, ["r2"]).contrib == (5.0, 1.0)
    # Sizes up to the pot are the same game in both.
    a, b = game(), game(limit="pot-limit")
    for line in ([], ["r1"], ["r0.5", "r1"], ["c", "r1", "r1", "c"]):
        assert a.legal_actions(street_state(a, line)) == b.legal_actions(street_state(b, line))
    with pytest.raises(ValueError):
        game(limit="fixed")
