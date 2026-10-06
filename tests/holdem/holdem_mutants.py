"""Plausible bugs injected into copies of our Hold'em game. The comparison with the
reference must catch every one, which shows it can see that kind of mistake.

Each entry is (mutant class, config for test_holdem_reference.params, categories it must hit).
"""
from bombpot.holdem import Holdem
from bombpot.holdem.evaluator import category


class KickersIgnored(Holdem):
    """Hands of the same category tie."""
    def _strength(self, hand, cards):
        return category(self.hand_value(hand + cards))


class BoardIgnored(Holdem):
    """Showdowns compare hole cards only."""
    def _strength(self, hand, cards):
        return self.hand_value(hand)


class BoardAForBoth(Holdem):
    """Board A's result is used for both halves of the pot."""
    def _shares(self, hand0, hand1, boards):
        return super()._shares(hand0, hand1, (boards[0],) * len(boards))


class BoardsDoNotBlock(Holdem):
    """Each board is dealt as if the other board's cards were still in the deck."""
    def remaining(self, s):
        kind, j = self.events[s.step][0], self.events[s.step][-1]
        if kind != "board":
            return super().remaining(s)
        used = {c for cards in s.hands + (s.boards[j],) for c in cards}
        return [c for c in self.deck.cards if c not in used]


class HoleCardsStayInDeck(Holdem):
    """Board cards can repeat a hole card."""
    def remaining(self, s):
        if self.events[s.step][0] != "board":
            return super().remaining(s)
        used = {c for b in s.boards for c in b}
        return [c for c in self.deck.cards if c not in used]


class HandsShareCards(Holdem):
    """Player 1's hand is dealt from the full deck."""
    def remaining(self, s):
        if self.events[s.step][0] == "hand":
            return list(self.deck.cards)
        return super().remaining(s)


class KeyShowsOpponentHand(Holdem):
    def infoset_key(self, s, player):
        return super().infoset_key(s, player) + ":" + self.deck.cards_str(s.hands[1 - player])


class KeyForgetsStreetOrder(Holdem):
    """Board cards sorted together, so which street each came on is lost."""
    def public_key(self, s):
        boards = "|".join(self.deck.cards_str(sorted(b)) for b in s.boards)
        return f"{boards}:{'/'.join(s.hist)}"


class BetSizeFromWrongStreet(Holdem):
    """Street 2 bets the street-1 size."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.bet_sizes = (self.bet_sizes[0],) * len(self.bet_sizes)


ONE_STREET = (3, 2, (1,), 2, "shared")
SPARE_CARDS = (4, 2, (1,), 2, "shared")  # with every card dealt, the opponent's hand is implied
TWO_STREETS = (3, 2, (1, 1), 2, "identical")
MUTANTS = [
    (KickersIgnored, ONE_STREET, {"payoffs"}),
    (BoardIgnored, ONE_STREET, {"payoffs"}),
    (BoardAForBoth, ONE_STREET, {"payoffs"}),
    (BoardsDoNotBlock, ONE_STREET, {"histories"}),
    (HoleCardsStayInDeck, ONE_STREET, {"histories"}),
    (HandsShareCards, ONE_STREET, {"histories"}),
    (KeyShowsOpponentHand, SPARE_CARDS, {"infosets"}),
    (KeyForgetsStreetOrder, TWO_STREETS, {"infosets"}),
    (BetSizeFromWrongStreet, TWO_STREETS, {"payoffs"}),
]
