"""Every game an experiment config can name, from all layers."""
from __future__ import annotations

from bombpot import holdem, toygames

GAMES = {**toygames.GAMES, **holdem.GAMES}


def make_game(name: str, **params):
    return GAMES[name](**params)
