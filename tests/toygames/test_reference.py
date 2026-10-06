"""Check A: Board-Leduc against an independent reference implementation.

tests/toygames/reference_board_leduc.py was written from the rules alone, without
seeing the package (then named toygames/): it deals real suited cards (every card
equally likely) and has its own betting, pot and showdown code. game_diff walks both
games in full and compares every terminal (chance probability summed over suits,
payoff), every decision node (player, legal actions) and the information partition.

The mutation tests inject plausible bugs into copies of our game and require the
comparison to catch each one, so a pass means something.
"""
from functools import lru_cache

import pytest

from game_diff import diff, walk
from mutants import MUTANTS
from reference_board_leduc import ReferenceBoardLeduc
from bombpot.toygames import BoardLeduc
from bombpot.core.solvers.lp import solve_game
from bombpot.core.tree import compile_game

MODES = ("shared", "independent", "identical")
# (num_ranks, num_boards, streets, deck_mode). With one board the deck mode is moot.
FAST = ([(n, 1, t, "shared") for n in (3, 4) for t in (1, 2)]
        + [(3, 2, t, m) for t in (1, 2) for m in MODES]
        + [(4, 2, 1, m) for m in MODES])
SLOW = [(4, 2, 2, m) for m in MODES]  # 20,160 to 50,400 suited deals each


def cid(cfg) -> str:
    n, b, t, m = cfg
    return f"N{n}-b{b}-s{t}-{m}"


def params(cfg) -> dict:
    n, b, t, m = cfg
    return dict(num_ranks=n, num_boards=b, streets=t, deck_mode=m)


def configs(fast, slow):
    return ([pytest.param(c, id=cid(c)) for c in fast]
            + [pytest.param(c, id=cid(c), marks=pytest.mark.slow) for c in slow])


@lru_cache(maxsize=None)
def reference_walk(cfg):
    game = ReferenceBoardLeduc(**params(cfg))
    return walk(game, game.describe)


@lru_cache(maxsize=None)
def our_walk(cfg):
    return walk(BoardLeduc(**params(cfg)))


@pytest.mark.parametrize("cfg", configs(FAST, SLOW))
def test_same_game_as_reference(cfg):
    d = diff(our_walk(cfg), reference_walk(cfg))
    assert d.ok, f"\n{d}"


@pytest.mark.parametrize("cfg", configs([c for c in FAST if c[0] == 3 or c[2] == 1], SLOW))
def test_lp_value_matches_reference(cfg):
    # End to end: the reference compiled at card level and solved by the same LP.
    (ours, _), _ = solve_game(compile_game(BoardLeduc(**params(cfg))))
    ref_tree = compile_game(ReferenceBoardLeduc(**params(cfg)), keep_meta=False)
    (theirs, other), _ = solve_game(ref_tree)
    assert theirs == pytest.approx(other, abs=1e-6)
    assert theirs == pytest.approx(ours, abs=1e-6)


# Mutation tests ---------------------------------------------------------------------


@pytest.mark.parametrize("mutant, cfg, expected", [pytest.param(*m, id=m[0].__name__) for m in MUTANTS])
def test_mutant_is_caught(mutant, cfg, expected):
    d = diff(walk(mutant(**params(cfg))), reference_walk(cfg))
    assert expected <= d.caught, f"expected {expected}, caught {d.caught}\n{d}"
