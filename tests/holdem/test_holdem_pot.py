"""Hold'em on the no-limit engine (`Holdem(betting="pot")`, core/betting.py) on tiny decks.

- Structure: chance sums to one, every terminal is zero-sum, infoset keys hold the
  player's own cards and everything public and never the opponent's cards.
- Solvers: on the compiled tree, DCFR converges to the sequence-form LP value, with stacks
  deep enough for full streets and shallow enough for all-in collapse on later streets.
- ACPC: one board against OpenSpiel's universal_poker no-limit, walked together along our
  action sequences. universal_poker takes a raise as the total a player has in, so each of
  our bets maps to that amount ("a" to the whole stack). At every decision node the same
  player acts, fold is legal in both or neither, every action we offer is legal in ACPC, and
  every size we drop for the minimum raise is illegal in ACPC too. Chance probabilities
  (ACPC deals card by card, we deal sorted combinations), terminals and payoffs agree, so
  ACPC also plays no betting after an all-in and call. Integer chips (antes 2 or 4), so
  ACPC's integer chips are ours.

One genuine rule difference: ACPC's minimum bet or raise increment is the big blind (here
the ante) or the previous increment, whichever is bigger, and ours is only the previous
increment. With pot fractions of 1/2 and up a bet adds at least the ante (half of a pot of
two antes), so the real game's menu never meets it; with a quarter-pot size it does (an
opening quarter-pot bet adds half an ante). The comparison checks that every action ACPC
rejects adds less than the ante, and doesn't follow those lines.
"""
from collections import defaultdict
from math import factorial

import pytest

from bombpot.core.betting import ALLIN
from bombpot.core.game import CHANCE
from bombpot.core.solvers.cfr import CFR
from bombpot.core.solvers.lp import solve_game
from bombpot.core.tree import compile_game
from bombpot.holdem import Holdem

CONFIGS = {
    # name: Holdem parameters (betting="pot" added)
    "4x2-1-b2-deep": dict(num_ranks=4, num_suits=2, board_cards=(1,), num_boards=2, stack=100),
    "3x2-1+1-b1-allin": dict(num_ranks=3, num_suits=2, board_cards=(1, 1), num_boards=1, stack=6),
    "4x2-1-b2-tiny": dict(num_ranks=4, num_suits=2, board_cards=(1,), num_boards=2, stack=1.5),
    "3x2-1+1-identical-cap3": dict(num_ranks=3, num_suits=2, board_cards=(1, 1), deck_mode="identical", stack=12,
                                   max_aggressive=3),
    "4x2-1-b2-potlimit": dict(num_ranks=4, num_suits=2, board_cards=(1,), num_boards=2, stack=6, sizes=(0.5, 2.0),
                              limit="pot-limit"),
}
_cache = {}


def make(name):
    return Holdem(betting="pot", **CONFIGS[name])


def built(name):
    if name not in _cache:
        g = make(name)
        tree = compile_game(g)
        _cache[name] = (g, tree, solve_game(tree))
    return _cache[name]


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


@pytest.mark.parametrize("name", CONFIGS)
def test_chance_zero_sum_and_keys(name):
    g = make(name)
    by_key = defaultdict(set)
    allin_terminals = 0
    for s in walk(g):
        if g.is_terminal(s):
            assert sum(g.returns(s)) == 0.0
            allin_terminals += g.all_in(s)
            assert len(s.hist) == g.num_streets or s.folder >= 0
            continue
        if g.current_player(s) == CHANCE:
            assert sum(q for _, q in g.chance_outcomes(s)) == pytest.approx(1.0, abs=1e-12)
            continue
        p = g.current_player(s)
        key = g.infoset_key(s, p)
        player, own, boards, hist = key.split(":")
        assert own == g.deck.cards_str(s.hands[p]) and int(player) == p
        assert boards == "|".join(g.deck.cards_str(b) for b in s.boards)
        assert hist == "/".join(s.hist)
        by_key[key].add((s.hands[p], s.boards, s.hist))
    assert all(len(v) == 1 for v in by_key.values())  # only the opponent's hand varies within a key
    for t in range(g.num_streets):
        assert sum(q for *_, q in g.deals(t)) == pytest.approx(1.0, abs=1e-12)
    if name != "4x2-1-b2-deep":
        assert allin_terminals > 0  # these configurations reach all-in showdowns


@pytest.mark.parametrize("name", CONFIGS)
def test_dcfr_matches_lp(name):
    g, tree, ((v0, v1), _) = built(name)
    assert v0 == pytest.approx(v1, abs=1e-6)
    rows = CFR(tree, "dcfr").run(iterations=3000, log_every=1000)
    for r in rows:
        assert -r["br_value_p1"] - 1e-9 <= v0 <= r["br_value_p0"] + 1e-9
    assert rows[-1]["exploitability"] < 1e-4
    assert rows[-1]["game_value"] == pytest.approx(v0, abs=2 * rows[-1]["exploitability"] + 1e-12)


# ACPC ----------------------------------------------------------------------------------------

ACPC_CONFIGS = {
    # (num_ranks, num_suits, board_cards, ante, stack, sizes, max_aggressive)
    "3x2-1+1-a2-s10": (3, 2, (1, 1), 2, 10, (0.5, 1.0), 2),      # all-ins on the flop and turn
    "3x2-1+1-a2-s40": (3, 2, (1, 1), 2, 40, (0.5, 1.0), 2),      # all-ins only on the turn
    "4x2-1-a4-s20-minraise": (4, 2, (1,), 4, 20, (0.25, 1.0), 2),  # quarter-pot raises below the min raise
    "3x2-2-a2-s30-cap3": (3, 2, (2,), 2, 30, (0.5, 1.0), 3),
}


def universal_nolimit(num_ranks, num_suits, board_cards, ante, stack, max_aggressive):
    import pyspiel

    rounds = len(board_cards)
    return pyspiel.load_game("universal_poker", {
        "betting": "nolimit", "numPlayers": 2, "numRounds": rounds, "blind": f"{ante} {ante}",
        "stack": f"{ante + stack} {ante + stack}", "firstPlayer": " ".join(["1"] * rounds),
        "maxRaises": " ".join([str(max_aggressive)] * rounds), "numSuits": num_suits, "numRanks": num_ranks,
        "numHoleCards": 2, "numBoardCards": " ".join(str(k) for k in board_cards),
        "bettingAbstraction": "fullgame"})


def chips(x: float) -> int:
    assert x == int(x), f"non-integer chips {x}"
    return int(x)


@pytest.mark.parametrize("name", ACPC_CONFIGS)
def test_matches_universal_poker_nolimit(name):
    pytest.importorskip("pyspiel")
    n, s_, cards, ante, stack, sizes, cap = ACPC_CONFIGS[name]
    g = Holdem(num_ranks=n, num_suits=s_, board_cards=cards, num_boards=1, ante=ante, betting="pot", stack=stack,
               sizes=sizes, max_aggressive=cap)
    up = universal_nolimit(n, s_, cards, ante, stack, cap)
    seen = defaultdict(int)

    def target_of(s, a):
        return dict(g.aggressive_options(s))[a]

    def rec(us, s):
        assert us.is_terminal() == g.is_terminal(s)
        if g.is_terminal(s):
            seen["terminals"] += 1
            seen["all-in terminals"] += g.all_in(s)
            assert tuple(us.returns()) == pytest.approx(g.returns(s), abs=1e-12)
            return
        if g.current_player(s) == CHANCE:
            assert us.is_chance_node()
            outs = g.chance_outcomes(s)
            k = g._event_size(s, g.events[s.step])
            assert len(us.chance_outcomes()) == len(g.remaining(s))
            for o, q in outs:
                u = us.clone()
                prob = 1.0
                for card in g._combos[k][o]:
                    prob *= dict(u.chance_outcomes())[card]
                    u.apply_action(card)
                assert q == pytest.approx(prob * factorial(k), rel=1e-12)
                rec(u, g.next_state(s, o))
            return
        p = g.current_player(s)
        assert us.current_player() == p
        seen["decisions"] += 1
        theirs = set(us.legal_actions())
        ours = g.legal_actions(s)
        assert ("f" in ours) == (0 in theirs)
        assert 1 in theirs
        raises = sorted(a for a in theirs if a > 1)
        follow = []
        for a in ours:
            if a in ("f", "c"):
                follow.append(a)
                continue
            if a == ALLIN:
                assert chips(target_of(s, a)) == raises[-1]
            if chips(target_of(s, a)) in theirs:
                follow.append(a)
            else:  # the one rule difference: ACPC's increment is at least the big blind
                assert target_of(s, a) - s.contrib[1 - p] < ante and a != ALLIN
                seen["below the big blind"] += 1
        # Each listed size we drop (below the minimum raise, not all-in) is illegal in ACPC too,
        # and ACPC allows raises exactly when we do (the cap, and nothing after an all-in).
        o = 1 - p
        if g.aggressive_options(s) or s.raises < g.max_aggressive[g.street(s)] and s.contrib[o] < g.total:
            for x, label in zip(g.sizes, g.size_labels):
                target = s.contrib[o] * (1 + 2 * x)
                if target < g.total and label not in ours:
                    seen["min-raise drops"] += 1
                    assert target < raises[0]
        assert bool(raises) == any(a not in ("f", "c") for a in ours)
        for a in follow:
            u = us.clone()
            u.apply_action({"f": 0, "c": 1}.get(a) if a in ("f", "c") else chips(target_of(s, a)))
            rec(u, g.next_state(s, a))

    rec(up.new_initial_state(), g.initial_state())
    assert seen["terminals"] > 1000 and seen["all-in terminals"] > 0
    if "minraise" in name:
        assert seen["min-raise drops"] > 0 and seen["below the big blind"] > 0
    else:
        assert seen["below the big blind"] == 0
