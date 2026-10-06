"""Every game an experiment config can name, from all layers."""
from __future__ import annotations

from bombpot import holdem, toygames

GAMES = {**toygames.GAMES, **holdem.GAMES}


def make_game(name: str, **params):
    return GAMES[name](**params)


def public_tree(game):
    """The game's public tree for the range solver: card-removal streets for Hold'em,
    dense ones (built by walking every deal) for the small games."""
    from bombpot.core.public import build_public_tree
    from bombpot.holdem import Holdem
    from bombpot.holdem.public import holdem_public_tree

    return holdem_public_tree(game) if isinstance(game, Holdem) else build_public_tree(game)
