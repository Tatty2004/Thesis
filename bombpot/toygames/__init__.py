"""Toy games: small enough to solve and check exactly.

Kuhn and Leduc check the solvers against known results and OpenSpiel; Board-Leduc is
the one-private-card double-board bomb pot used for the first bucketing experiment.
"""
from bombpot.toygames.board_leduc import BoardLeduc
from bombpot.toygames.kuhn import Kuhn
from bombpot.toygames.leduc import Leduc

GAMES = {"kuhn": Kuhn, "leduc": Leduc, "board_leduc": BoardLeduc}

__all__ = ["BoardLeduc", "Kuhn", "Leduc", "GAMES"]
