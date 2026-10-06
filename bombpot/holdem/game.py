"""Double-board Hold'em bomb pots, at card level, with any deck size.

Ante, two private cards each (player 0 first, one deck), then on every street
`board_cards[t]` cards to each board followed by a limit betting round, then showdown.
There is no betting before the first board cards. On each board a player's hand is the
best five-card hand from their two cards plus that board's cards (all of them when
there are fewer than five). Each board is worth half the pot, and equal hands split
that board's half. Board cards play for both players.

Deck modes:
  shared     one deck for everything, so the boards block each other (real play)
  identical  board B copies board A, which reproduces num_boards=1 exactly

`flops` fixes each board's street-1 cards instead of dealing them, which is how a
bomb pot splits into one independent game per pair of flops.

A hand, or one board's cards on one street, is dealt as a single chance event whose
outcome indexes the sorted combination, so each combination is one history.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

from bombpot.core.limit import LimitPoker, PokerState
from bombpot.holdem.cards import Deck
from bombpot.holdem.evaluator import hand_value

DECK_MODES = ("shared", "identical")
HOLE_CARDS = 2


class Holdem(LimitPoker):

    def __init__(self, num_ranks: int = 13, num_suits: int = 4, board_cards=(3, 1), num_boards: int = 2,
                 deck_mode: str = "shared", flops=None, ante: float = 1.0, bet_sizes=(2.0, 4.0), max_raises=2):
        if num_boards not in (1, 2):
            raise ValueError("num_boards must be 1 or 2")
        if deck_mode not in DECK_MODES:
            raise ValueError(f"deck_mode must be one of {DECK_MODES}")
        board_cards = tuple(int(k) for k in board_cards)
        if not board_cards or any(k < 1 for k in board_cards):
            raise ValueError("board_cards needs at least one street, each with at least one card")
        streets = len(board_cards)
        if isinstance(max_raises, int):
            max_raises = [max_raises] * streets
        self.deck = Deck(num_ranks, num_suits)
        dealt_boards = 1 if deck_mode == "identical" else num_boards
        if 2 * HOLE_CARDS + dealt_boards * sum(board_cards) > self.deck.size:
            raise ValueError("the deck is too small for these hands and boards")
        self.num_ranks, self.num_suits, self.board_cards = num_ranks, num_suits, board_cards
        self.deck_mode = deck_mode
        self.flops = None
        if flops is not None:
            parsed = tuple(tuple(sorted(self.deck.parse(f) if isinstance(f, str) else f)) for f in flops)
            if len(parsed) != num_boards or any(len(f) != board_cards[0] for f in parsed):
                raise ValueError(f"flops needs {num_boards} boards of {board_cards[0]} cards")
            cards = [c for f in (parsed[:1] if deck_mode == "identical" else parsed) for c in f]
            if len(set(cards)) != len(cards) or (deck_mode == "identical" and len(set(parsed)) != 1):
                raise ValueError("flops share a card (or differ in identical mode)")
            self.flops = parsed
        events = [("hand", 0), ("hand", 1)]
        for t in range(streets):
            if not (t == 0 and self.flops is not None):
                events.extend([("board", 0)] if deck_mode == "identical" else [("board", j) for j in range(num_boards)])
            events.append(("bet", t))
        events.append(("showdown",))
        super().__init__(events, num_boards, ante, list(bet_sizes)[:streets], list(max_raises)[:streets])
        self._combos = {k: list(combinations(self.deck.cards, k)) for k in {HOLE_CARDS, *board_cards}}
        self._index = {k: {c: i for i, c in enumerate(v)} for k, v in self._combos.items()}
        self._values: dict = {}
        self.params = dict(num_ranks=num_ranks, num_suits=num_suits, board_cards=list(board_cards),
                           num_boards=num_boards, deck_mode=deck_mode, ante=self.ante,
                           bet_sizes=list(self.bet_sizes), max_raises=list(self.max_raises))
        if self.flops is not None:
            self.params["flops"] = [self.deck.cards_str(f) for f in self.flops]
        self.name = "holdem(" + ",".join(f"{k}={v}" for k, v in self.params.items()) + ")"

    # Dealing ------------------------------------------------------------------------

    def initial_state(self) -> PokerState:
        boards = self.flops if self.flops is not None else ((),) * self.num_boards
        s = PokerState(0, (), boards, (), (self.ante, self.ante), 0, 0, -1)
        return self._enter(s, 0)

    def _event_size(self, s: PokerState, event) -> int:
        if event[0] == "hand":
            return HOLE_CARDS
        return self.board_cards[len(s.hist)]  # the street being dealt is the next betting street

    def remaining(self, s: PokerState) -> list[int]:
        """Cards the next deal can use."""
        boards = s.boards[:1] if self.deck_mode == "identical" else s.boards
        used = {c for cards in s.hands + boards for c in cards}
        return [c for c in self.deck.cards if c not in used]

    def chance_outcomes(self, s: PokerState) -> list[tuple[int, float]]:
        k = self._event_size(s, self.events[s.step])
        left = self.remaining(s)
        q = 1.0 / comb(len(left), k)
        index = self._index[k]
        return [(index[c], q) for c in combinations(left, k)]

    def _deal(self, s: PokerState, event, outcome: int) -> PokerState:
        kind, j = event
        cards = self._combos[self._event_size(s, event)][outcome]
        if kind == "hand":
            return PokerState(s.step, s.hands + (cards,), s.boards, s.hist, s.contrib, s.raises, s.to_act, s.folder)
        if self.deck_mode == "identical":
            boards = tuple(b + cards for b in s.boards)
        else:
            boards = tuple(b + cards if i == j else b for i, b in enumerate(s.boards))
        return PokerState(s.step, s.hands, boards, s.hist, s.contrib, s.raises, s.to_act, s.folder)

    def outcome(self, cards) -> int:
        """The chance outcome that deals exactly these cards (tests and examples)."""
        cards = tuple(sorted(self.deck.parse(cards) if isinstance(cards, str) else cards))
        return self._index[len(cards)][cards]

    # Showdown -----------------------------------------------------------------------

    def hand_value(self, cards) -> int:
        key = tuple(sorted(cards))
        v = self._values.get(key)
        if v is None:
            v = self._values[key] = hand_value([self.deck.rank(c) for c in key], [self.deck.suit(c) for c in key])
        return v

    def _strength(self, hand: tuple, cards: tuple) -> int:
        return self.hand_value(hand + cards)

    # Keys ---------------------------------------------------------------------------

    def rank_str(self, card: int) -> str:  # LimitPoker.public_key writes board cards with this
        return self.deck.card_str(card)

    def infoset_key(self, s: PokerState, player: int) -> str:
        # Own cards, then public info only; the opponent's cards never appear.
        return f"{player}:{self.deck.cards_str(s.hands[player])}:{self.public_key(s)}"
