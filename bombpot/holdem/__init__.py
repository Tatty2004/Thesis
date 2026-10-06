"""Double-board Hold'em: real hand ranking, any deck from tiny test decks to 52 cards."""
from bombpot.holdem.game import Holdem

GAMES = {"holdem": Holdem}

__all__ = ["Holdem", "GAMES"]
