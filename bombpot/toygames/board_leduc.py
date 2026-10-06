"""Board-Leduc: a Leduc-style bomb pot with one or two boards.

Ante, one private card each, then one card to each board and a betting round.
With two streets, a second card goes to each board followed by another betting
round, then showdown. There is no betting before the first board cards.

On each board, matching any board card makes a pair of your rank, otherwise you
have high card of your rank. Pair beats high card, the higher rank wins, and equal
hands split. Each board is worth half the pot.

Deck modes:
  shared       one deck of N ranks x 2 suits, so the boards block each other
  independent  each board is dealt from its own copy of the deck minus both
               private cards, so the boards never block each other
  identical    board B copies board A, which reproduces num_boards=1 exactly
"""
from __future__ import annotations

from bombpot.core.limit import LimitPoker, PokerState

DECK_MODES = ("shared", "independent", "identical")


class BoardLeduc(LimitPoker):

    def __init__(self, num_ranks: int = 3, num_boards: int = 2, streets: int = 1,
                 deck_mode: str = "shared", ante: float = 1.0, bet_sizes=(2.0, 4.0),
                 max_raises=2, board_order=None):
        """`board_order` changes which board is dealt first on each street (tests only)."""
        if not 3 <= num_ranks <= len(self.rank_chars):
            raise ValueError("num_ranks must be at least 3")
        if num_boards not in (1, 2) or streets not in (1, 2):
            raise ValueError("num_boards and streets must be 1 or 2")
        if deck_mode not in DECK_MODES:
            raise ValueError(f"deck_mode must be one of {DECK_MODES}")
        if isinstance(max_raises, int):
            max_raises = [max_raises] * streets
        order = tuple(range(num_boards)) if board_order is None else tuple(board_order)
        if sorted(order) != list(range(num_boards)):
            raise ValueError("board_order must be a permutation of the boards")
        self.num_ranks = num_ranks
        self.deck_mode = deck_mode
        events = [("hand", 0), ("hand", 1)]
        for t in range(streets):
            if deck_mode == "identical":
                events.append(("board", 0))  # dealt to every board at once
            else:
                events.extend(("board", j) for j in order)
            events.append(("bet", t))
        events.append(("showdown",))
        super().__init__(events, num_boards, ante, list(bet_sizes)[:streets], list(max_raises)[:streets])
        self.params = dict(num_ranks=num_ranks, num_boards=num_boards, streets=streets, deck_mode=deck_mode,
                           ante=self.ante, bet_sizes=list(self.bet_sizes), max_raises=list(self.max_raises))
        if board_order is not None:
            self.params["board_order"] = list(order)
        self.name = "board_leduc(" + ",".join(f"{k}={v}" for k, v in self.params.items()) + ")"

    def _counts(self, s: PokerState, event) -> list[int]:
        kind, j = event
        if kind == "hand":
            used = s.hands
        elif self.deck_mode == "shared":
            used = s.hands + sum(s.boards, ())
        elif self.deck_mode == "independent":
            used = s.hands + s.boards[j]  # this board's own deck copy
        else:
            used = s.hands + s.boards[0]
        return [2 - used.count(r) for r in range(self.num_ranks)]

    def _deal(self, s: PokerState, event, rank: int) -> PokerState:
        if event[0] == "board" and self.deck_mode == "identical":
            boards = tuple(b + (rank,) for b in s.boards)
            return PokerState(s.step, s.hands, boards, s.hist, s.contrib, s.raises, s.to_act, s.folder)
        return super()._deal(s, event, rank)

    def _strength(self, hand: int, cards: tuple) -> int:
        return hand + self.num_ranks * (hand in cards)
