"""Cards for decks of any size, from tiny test decks to the full 52.

A deck has the lowest `num_ranks` of 2, 3, ..., A, each in `num_suits` suits (c, d, h, s
in that order). Card `rank * num_suits + suit` is an int, the same numbering as ACPC's
universal_poker, so rank 0 is a deuce and rank 12 an ace.
"""
from __future__ import annotations

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"


class Deck:
    def __init__(self, num_ranks: int = 13, num_suits: int = 4):
        if not 2 <= num_ranks <= len(RANK_CHARS):
            raise ValueError(f"num_ranks must be 2 to {len(RANK_CHARS)}")
        if not 1 <= num_suits <= len(SUIT_CHARS):
            raise ValueError(f"num_suits must be 1 to {len(SUIT_CHARS)}")
        self.num_ranks, self.num_suits = num_ranks, num_suits
        self.size = num_ranks * num_suits
        self.cards = tuple(range(self.size))

    def rank(self, card: int) -> int:
        return card // self.num_suits

    def suit(self, card: int) -> int:
        return card % self.num_suits

    def card(self, rank: int, suit: int) -> int:
        return rank * self.num_suits + suit

    def card_str(self, card: int) -> str:
        return RANK_CHARS[self.rank(card)] + SUIT_CHARS[self.suit(card)]

    def cards_str(self, cards) -> str:
        return "".join(self.card_str(c) for c in cards)

    def parse(self, text: str) -> tuple[int, ...]:
        """'AhKd' -> the two cards, in the order written."""
        if len(text) % 2:
            raise ValueError(f"cards come in pairs of characters: {text!r}")
        out = []
        for i in range(0, len(text), 2):
            r, s = RANK_CHARS.find(text[i]), SUIT_CHARS.find(text[i + 1])
            if not (0 <= r < self.num_ranks and 0 <= s < self.num_suits):
                raise ValueError(f"{text[i:i + 2]!r} is not in a {self.num_ranks}x{self.num_suits} deck")
            out.append(self.card(r, s))
        return tuple(out)
