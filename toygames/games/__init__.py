from toygames.games.base import CHANCE, TERMINAL, CardGame, Game
from toygames.games.kuhn import Kuhn

GAMES = {"kuhn": Kuhn}


def make_game(name: str, **params):
    return GAMES[name](**params)


__all__ = ["CHANCE", "TERMINAL", "CardGame", "Game", "Kuhn", "GAMES", "make_game"]
