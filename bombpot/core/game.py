"""The game protocols every solver, evaluator and abstraction works through.

core/ sees a game only through `Game` and `CardGame` (and the tree compiled from them),
so the same code runs the toy games, Hold'em and PLO.
"""
from __future__ import annotations

from typing import Hashable, Protocol, Sequence

import numpy as np

CHANCE = -1
TERMINAL = -2


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
