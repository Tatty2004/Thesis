from toygames.games.base import CHANCE, TERMINAL, CardGame, Game
from toygames.games.kuhn import Kuhn
from toygames.games.leduc import Leduc

GAMES = {"kuhn": Kuhn, "leduc": Leduc}


def make_game(name: str, **params):
    return GAMES[name](**params)


__all__ = ["CHANCE", "TERMINAL", "CardGame", "Game", "Kuhn", "Leduc", "GAMES", "make_game"]
