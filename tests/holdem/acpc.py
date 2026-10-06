"""OpenSpiel's universal_poker (the ACPC poker engine) as an outside reference for Hold'em.

- `acpc_winner` asks ACPC's own hand evaluator who wins a showdown, by dealing chosen
  cards and checking the hand down.
- `UniversalPoker` wraps a one-board universal_poker game in our Game protocol (plus a
  `describe` for game_diff), so its whole tree can be compared with ours.

universal_poker numbers cards like we do (rank * num_suits + suit, suits c d h s) and
deals hole cards player 0, player 0, player 1, player 1, then each round's board cards
before that round's betting.
"""
from __future__ import annotations

from functools import lru_cache

import pyspiel

from bombpot.core.game import CHANCE, TERMINAL

FOLD, CALL, RAISE = 0, 1, 2  # universal_poker action ids
LABELS = {FOLD: "f", CALL: "c", RAISE: "r"}
IDS = {v: k for k, v in LABELS.items()}


@lru_cache(maxsize=None)
def universal_poker(num_ranks: int, num_suits: int, board_cards: tuple, ante: int = 1, bet_sizes=(2, 4),
                    max_raises=(2, 2)):
    rounds = len(board_cards)
    return pyspiel.load_game("universal_poker", {
        "betting": "limit", "numPlayers": 2, "numRounds": rounds, "blind": f"{ante} {ante}",
        "raiseSize": " ".join(str(int(b)) for b in bet_sizes[:rounds]),
        "firstPlayer": " ".join(["1"] * rounds), "maxRaises": " ".join(str(m) for m in max_raises[:rounds]),
        "numSuits": num_suits, "numRanks": num_ranks, "numHoleCards": 2,
        "numBoardCards": " ".join(str(k) for k in board_cards)})


def acpc_winner(num_ranks: int, num_suits: int, hole0, hole1, board) -> int:
    """+1 if player 0's hand beats player 1's on this board, -1 if it loses, 0 for a tie."""
    game = universal_poker(num_ranks, num_suits, (len(board),))
    s = game.new_initial_state()
    for card in (*hole0, *hole1, *board):
        assert s.is_chance_node()
        s.apply_action(card)
    s.apply_action(CALL)
    s.apply_action(CALL)
    assert s.is_terminal()
    r0 = s.returns()[0]
    return (r0 > 0) - (r0 < 0)


class UniversalPoker:
    """A one-board universal_poker game, walked through our Game protocol.

    A state is (pyspiel state, hole cards per player, board cards, betting per round).
    Cards are recorded as dealt, and `describe` sorts each hand and each round's board
    cards so the record matches ours whatever order the cards came in.
    """

    def __init__(self, num_ranks: int, num_suits: int, board_cards, ante=1, bet_sizes=(2, 4), max_raises=(2, 2)):
        self.board_cards = tuple(board_cards)
        self.game = universal_poker(num_ranks, num_suits, self.board_cards, ante, tuple(bet_sizes),
                                    tuple(max_raises))
        self.name = f"universal_poker({self.game})"

    def initial_state(self):
        return (self.game.new_initial_state(), ((), ()), (), ())

    def is_terminal(self, state) -> bool:
        return state[0].is_terminal()

    def current_player(self, state) -> int:
        s = state[0]
        if s.is_terminal():
            return TERMINAL
        return CHANCE if s.is_chance_node() else s.current_player()

    def legal_actions(self, state):
        return tuple(LABELS[a] for a in state[0].legal_actions())

    def chance_outcomes(self, state):
        return state[0].chance_outcomes()

    def next_state(self, state, action):
        s, hands, board, hist = state
        s = s.clone()
        if s.is_chance_node():
            s.apply_action(action)
            if len(hands[0]) < 2:
                hands = (hands[0] + (action,), hands[1])
            elif len(hands[1]) < 2:
                hands = (hands[0], hands[1] + (action,))
            else:
                board = board + (action,)
        else:
            s.apply_action(IDS[action])
            hist = hist[:-1] + (hist[-1] + action,)
        # A betting round starts once all of its board cards are out.
        r = len(hist)
        if (len(hands[1]) == 2 and r < len(self.board_cards) and len(board) == sum(self.board_cards[:r + 1])
                and not s.is_terminal() and not s.is_chance_node()):
            hist = hist + ("",)
        return (s, hands, board, hist)

    def returns(self, state):
        return tuple(state[0].returns())

    def infoset_key(self, state, player: int) -> str:
        return state[0].information_state_string(player)

    def describe(self, state):
        _, hands, board, hist = state
        segments, start = [], 0
        for k in self.board_cards[:len(hist)]:
            segments.extend(sorted(board[start:start + k]))
            start += k
        return (tuple(tuple(sorted(h)) for h in hands), (tuple(segments),), hist)
