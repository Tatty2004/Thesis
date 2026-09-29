"""Plausible bugs injected into copies of our Board-Leduc, one per class.

test_reference.py requires the comparison with the independent reference to catch
every one of them, in the categories listed in MUTANTS.
"""
from toygames.games import BoardLeduc


class BoardBFromACopy(BoardLeduc):
    """independent: board B's second card is drawn from board A's deck copy."""

    def _counts(self, s, event):
        if event == ("board", 1) and self.deck_mode == "independent" and s.boards[1]:
            used = s.hands + s.boards[0]
            return [2 - used.count(r) for r in range(self.num_ranks)]
        return super()._counts(s, event)


class BoardsDoNotBlock(BoardLeduc):
    """shared: each board ignores the other board's cards."""

    def _counts(self, s, event):
        if event[0] == "board" and self.deck_mode == "shared":
            used = s.hands + s.boards[event[1]]
            return [2 - used.count(r) for r in range(self.num_ranks)]
        return super()._counts(s, event)


class RaisesNotReset(BoardLeduc):
    """Street 2 starts with street 1's raise count."""

    def _enter(self, s, step):
        new = super()._enter(s, step)
        if self.kinds[step] == "bet" and s.hist:
            new = new._replace(raises=s.raises)
        return new


class Street2BetTooSmall(BoardLeduc):
    """Street 2 bets the street-1 size (2 instead of 4)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bet_sizes = (self.bet_sizes[0],) * len(self.bet_sizes)


class FoldWhenNotFacing(BoardLeduc):
    """Fold is offered even when nobody has bet."""

    def legal_actions(self, s):
        acts = super().legal_actions(s)
        return acts if "f" in acts else ("f",) + acts


class PairedBoardCounts(BoardLeduc):
    """A paired board gives both players that pair (so they tie without kickers)."""

    def _strength(self, hand, cards):
        if hand in cards:
            return self.num_ranks + hand
        pairs = [c for c in set(cards) if cards.count(c) > 1]
        return self.num_ranks + max(pairs) if pairs else hand


class Player1FirstOnStreet2(BoardLeduc):
    """Player 1 opens the betting on street 2."""

    def _enter(self, s, step):
        new = super()._enter(s, step)
        if self.kinds[step] == "bet" and len(new.hist) == 2:
            new = new._replace(to_act=1)
        return new


def identical_dealt_independently(**kw):
    """identical: board B gets its own card instead of copying board A."""
    return BoardLeduc(**{**kw, "deck_mode": "shared"})


class KeyShowsOpponentCard(BoardLeduc):
    def infoset_key(self, s, player):
        return super().infoset_key(s, player) + ":" + self.rank_str(s.hands[1 - player])


class KeyForgetsStreet1(BoardLeduc):
    """Street-2 keys keep only the current street's betting."""

    def public_key(self, s):
        boards = "|".join("".join(self.rank_str(c) for c in b) for b in s.boards)
        return f"{boards}:{s.hist[-1]}"


class OnlyLastCardPairs(BoardLeduc):
    """Showdown only looks at the latest card on each board."""

    def _strength(self, hand, cards):
        return hand + self.num_ranks * (hand == cards[-1])


class EachBoardWinsWholePot(BoardLeduc):
    def returns(self, s):
        if s.folder >= 0:
            return super().returns(s)
        shares = self._shares(s.hands[0], s.hands[1], s.boards)
        u0 = (s.contrib[0] + s.contrib[1]) * sum(shares) - s.contrib[0]
        return (u0, -u0)


class PrivateCardsNotBlocked(BoardLeduc):
    """Player 1's card is dealt as if player 0's were still in the deck."""

    def _counts(self, s, event):
        if event == ("hand", 1):
            return [2] * self.num_ranks
        return super()._counts(s, event)


class BoardCopiesKeepPrivateCards(BoardLeduc):
    """independent: the board copies still hold the private cards."""

    def _counts(self, s, event):
        if event[0] == "board" and self.deck_mode == "independent":
            used = s.boards[event[1]]
            return [2 - used.count(r) for r in range(self.num_ranks)]
        return super()._counts(s, event)


class OpeningBetNotCounted(BoardLeduc):
    """The raise cap counts only raises after the opening bet."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.max_raises = tuple(m + 1 for m in self.max_raises)


# (mutant, config, categories the comparison must flag)
MUTANTS = [
    (BoardBFromACopy, (3, 2, 2, "independent"), {"probabilities"}),
    (BoardsDoNotBlock, (3, 2, 2, "shared"), {"probabilities", "histories"}),
    (RaisesNotReset, (3, 2, 2, "shared"), {"actions"}),
    (Street2BetTooSmall, (3, 2, 2, "shared"), {"payoffs"}),
    (FoldWhenNotFacing, (3, 2, 1, "shared"), {"actions"}),
    (PairedBoardCounts, (3, 2, 2, "shared"), {"payoffs"}),
    (Player1FirstOnStreet2, (3, 2, 2, "shared"), {"players"}),
    (identical_dealt_independently, (3, 2, 1, "identical"), {"probabilities", "histories"}),
    (KeyShowsOpponentCard, (3, 2, 1, "shared"), {"infosets"}),
    (KeyForgetsStreet1, (3, 2, 2, "shared"), {"infosets"}),
    (OnlyLastCardPairs, (3, 2, 2, "shared"), {"payoffs"}),
    (EachBoardWinsWholePot, (3, 2, 1, "shared"), {"payoffs"}),
    (PrivateCardsNotBlocked, (3, 2, 1, "shared"), {"probabilities"}),
    (BoardCopiesKeepPrivateCards, (3, 2, 1, "independent"), {"probabilities"}),
    (OpeningBetNotCounted, (3, 2, 1, "shared"), {"actions"}),
]
