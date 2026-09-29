"""The Game protocol, plus the limit-betting engine shared by the poker games.

solvers/, eval/ and abstraction/ see a game only through `Game` and `CardGame`
(and the tree compiled from them), so they carry over unchanged to Hold'em and PLO.
"""
from __future__ import annotations

from typing import Hashable, NamedTuple, Protocol, Sequence

import numpy as np

CHANCE = -1
TERMINAL = -2

FOLD, CALL, RAISE = "f", "c", "r"


class Game(Protocol):
    """A finite two-player zero-sum game with perfect recall."""

    name: str

    def initial_state(self): ...

    def is_terminal(self, state) -> bool: ...

    def current_player(self, state) -> int:
        """0 or 1 at decision nodes, CHANCE at chance nodes."""
        ...

    def legal_actions(self, state) -> Sequence[str]: ...

    def chance_outcomes(self, state) -> Sequence[tuple[Hashable, float]]:
        """(outcome, probability) pairs, all with positive probability."""
        ...

    def next_state(self, state, action): ...

    def returns(self, state) -> tuple[float, float]: ...

    def infoset_key(self, state, player: int) -> str: ...


class CardGame(Game, Protocol):
    """A Game where each player gets a private hand up front and public board cards
    are dealt before betting streets. abstraction/ assumes nothing beyond this."""

    num_streets: int
    num_boards: int

    def street(self, state) -> int:
        """Index of the betting street at a decision node."""
        ...

    def private_hand(self, state, player: int) -> Hashable: ...

    def board(self, state) -> tuple:
        """Public cards so far. The board at street t is a prefix of the board at t + 1."""
        ...

    def public_key(self, state) -> str:
        """Everything public at a decision node: the board and the betting on every street."""
        ...

    def deals(self, street: int) -> list[tuple[tuple, Hashable, Hashable, float]]:
        """Every (board, hand0, hand1, probability) at the start of betting on `street`."""
        ...

    def showdown(self, board: tuple, hand0, hand1) -> np.ndarray:
        """Player 0's share of each board's half of the pot: 1 win, 0.5 tie, 0 loss."""
        ...


class PokerState(NamedTuple):
    step: int       # index into the game's event list
    hands: tuple    # private ranks dealt so far, player 0 first
    boards: tuple   # one tuple of ranks per board
    hist: tuple     # betting string for every street reached
    contrib: tuple  # chips each player has put in, antes included
    raises: int     # raises so far on the current street
    to_act: int     # player to act while betting
    folder: int     # player who folded, or -1


def settle(contrib: tuple, shares: Sequence[float]) -> float:
    """Player 0's net result at showdown. Each board is worth an equal part of the
    pot, and `shares[j]` is player 0's share of board j (so a tie splits only that part)."""
    pot = contrib[0] + contrib[1]
    return pot * sum(shares) / len(shares) - contrib[0]


class LimitPoker:
    """Two-player limit poker over ranks, driven by a fixed list of events:
    ("hand", p) deals player p a card, ("board", j) deals a card to board j,
    ("bet", t) is betting street t and ("showdown",) ends the hand.

    Player 0 acts first on every street. A bet or raise counts against the street's
    raise cap, fold is legal only when facing a bet, and the raise counter resets
    each street. Chance outcomes are ranks, weighted by how many cards of that rank
    are left.
    """

    num_players = 2
    rank_chars = "0123456789abcdefghijklmnopqrstuvwxyz"

    def __init__(self, events, num_boards, ante, bet_sizes, max_raises):
        self.events = tuple(events)
        self.kinds = tuple(e[0] for e in self.events)
        self.num_boards = num_boards
        self.num_streets = self.kinds.count("bet")
        self.ante = float(ante)
        self.bet_sizes = tuple(float(b) for b in bet_sizes)
        self.max_raises = tuple(int(m) for m in max_raises)
        if len(self.bet_sizes) < self.num_streets or len(self.max_raises) < self.num_streets:
            raise ValueError("need a bet size and a raise cap for every street")

    # Subclass hooks ---------------------------------------------------------------

    def _counts(self, s: PokerState, event) -> list[int]:
        """Cards of each rank left for the deal `event`."""
        raise NotImplementedError

    def _deal(self, s: PokerState, event, rank: int) -> PokerState:
        kind, target = event
        if kind == "hand":
            hands = s.hands + (rank,)
            return PokerState(s.step, hands, s.boards, s.hist, s.contrib, s.raises, s.to_act, s.folder)
        boards = tuple(b + (rank,) if j == target else b for j, b in enumerate(s.boards))
        return PokerState(s.step, s.hands, boards, s.hist, s.contrib, s.raises, s.to_act, s.folder)

    def _strength(self, hand: int, cards: tuple) -> int:
        """Comparable strength of `hand` on one board."""
        raise NotImplementedError

    def rank_str(self, rank: int) -> str:
        return self.rank_chars[rank]

    # Game protocol ----------------------------------------------------------------

    def initial_state(self) -> PokerState:
        s = PokerState(0, (), ((),) * self.num_boards, (), (self.ante, self.ante), 0, 0, -1)
        return self._enter(s, 0)

    def is_terminal(self, s: PokerState) -> bool:
        return s.folder >= 0 or self.kinds[s.step] == "showdown"

    def current_player(self, s: PokerState) -> int:
        kind = self.kinds[s.step]
        if s.folder >= 0 or kind == "showdown":
            return TERMINAL
        return s.to_act if kind == "bet" else CHANCE

    def legal_actions(self, s: PokerState) -> tuple[str, ...]:
        p = s.to_act
        facing = s.contrib[1 - p] > s.contrib[p]
        if s.raises < self.max_raises[len(s.hist) - 1]:
            return (FOLD, CALL, RAISE) if facing else (CALL, RAISE)
        return (FOLD, CALL) if facing else (CALL,)

    def chance_outcomes(self, s: PokerState) -> list[tuple[int, float]]:
        counts = self._counts(s, self.events[s.step])
        total = sum(counts)
        return [(r, c / total) for r, c in enumerate(counts) if c > 0]

    def next_state(self, s: PokerState, action) -> PokerState:
        event = self.events[s.step]
        if event[0] == "bet":
            return self._bet(s, action)
        return self._enter(self._deal(s, event, action), s.step + 1)

    def returns(self, s: PokerState) -> tuple[float, float]:
        if s.folder >= 0:
            lost = s.contrib[s.folder]
            return (-lost, lost) if s.folder == 0 else (lost, -lost)
        u0 = settle(s.contrib, self._shares(s.hands[0], s.hands[1], s.boards))
        return (u0, -u0)

    def infoset_key(self, s: PokerState, player: int) -> str:
        # Own rank, then public info only; the opponent's card never appears.
        return f"{player}:{self.rank_str(s.hands[player])}:{self.public_key(s)}"

    # CardGame protocol ------------------------------------------------------------

    def street(self, s: PokerState) -> int:
        return len(s.hist) - 1

    def private_hand(self, s: PokerState, player: int) -> int:
        return s.hands[player]

    def board(self, s: PokerState) -> tuple:
        return tuple(c for cards in zip(*s.boards) for c in cards)

    def public_key(self, s: PokerState) -> str:
        boards = "|".join("".join(self.rank_str(c) for c in b) for b in s.boards)
        return f"{boards}:{'/'.join(s.hist)}"

    def deals(self, street: int) -> list[tuple[tuple, int, int, float]]:
        out = []
        stack = [(self.initial_state(), 1.0)]
        while stack:
            s, prob = stack.pop()
            if self.kinds[s.step] == "bet":
                if len(s.hist) - 1 == street:
                    out.append((self.board(s), s.hands[0], s.hands[1], prob))
                else:  # betting doesn't change the deal, so check it down
                    stack.append((self.next_state(self.next_state(s, CALL), CALL), prob))
                continue
            for rank, q in self.chance_outcomes(s):
                stack.append((self.next_state(s, rank), prob * q))
        return out

    def showdown(self, board: tuple, hand0: int, hand1: int) -> np.ndarray:
        boards = tuple(board[j::self.num_boards] for j in range(self.num_boards))
        return np.array(self._shares(hand0, hand1, boards))

    # Internals --------------------------------------------------------------------

    def _shares(self, hand0: int, hand1: int, boards: tuple) -> list[float]:
        shares = []
        for cards in boards:
            s0, s1 = self._strength(hand0, cards), self._strength(hand1, cards)
            shares.append(1.0 if s0 > s1 else 0.5 if s0 == s1 else 0.0)
        return shares

    def _enter(self, s: PokerState, step: int) -> PokerState:
        """Move to event `step`, opening a fresh betting string if it is a betting street."""
        if self.kinds[step] == "bet":
            return PokerState(step, s.hands, s.boards, s.hist + ("",), s.contrib, 0, 0, -1)
        return PokerState(step, s.hands, s.boards, s.hist, s.contrib, s.raises, s.to_act, s.folder)

    def _bet(self, s: PokerState, a: str) -> PokerState:
        p, o = s.to_act, 1 - s.to_act
        t = len(s.hist) - 1
        hist = s.hist[:t] + (s.hist[t] + a,)
        c = s.contrib
        if a == FOLD:
            return PokerState(s.step, s.hands, s.boards, hist, c, s.raises, p, p)
        if a == RAISE:
            raised = c[o] + self.bet_sizes[t]
            c = (raised, c[1]) if p == 0 else (c[0], raised)
            return PokerState(s.step, s.hands, s.boards, hist, c, s.raises + 1, o, -1)
        if a != CALL:
            raise ValueError(f"unknown action {a!r}")
        if c[o] > c[p]:  # a call closes the street
            s = PokerState(s.step, s.hands, s.boards, hist, (c[o], c[o]), s.raises, o, -1)
            return self._enter(s, s.step + 1)
        s = PokerState(s.step, s.hands, s.boards, hist, c, s.raises, o, -1)
        if len(hist[t]) >= 2:  # check behind closes the street
            return self._enter(s, s.step + 1)
        return s
