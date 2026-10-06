"""R1: our Hold'em game against two outside references, compared history by history.

- One board: OpenSpiel's universal_poker (the ACPC engine), which deals one card at a
  time and judges showdowns with ACPC's evaluator.
- Two boards (which universal_poker can't play): tests/holdem/reference_holdem.py, a
  naive card-by-card implementation with its own betting and chip code whose showdowns
  are judged by ACPC.

game_diff compares every terminal (chance probability summed over the orders the cards
can come in, payoff), every decision node (player, legal actions) and the information
partition. The mutation tests inject plausible bugs and require the comparison to
catch each one.

universal_poker's information state string sorts all public cards together, so with
board cards on two rounds it can't tell which card came on which street (no perfect
recall). There the partition is compared with our key made equally street-blind, and
separately our real key must be exactly as fine plus the street order.
"""
from functools import lru_cache

import pytest

from bombpot.holdem import Holdem

pytest.importorskip("pyspiel")
from acpc import UniversalPoker  # noqa: E402
from game_diff import diff, walk  # noqa: E402
from holdem_mutants import MUTANTS  # noqa: E402
from reference_holdem import ReferenceHoldem  # noqa: E402


def params(cfg) -> dict:
    n, s, cards, boards, mode = cfg
    return dict(num_ranks=n, num_suits=s, board_cards=cards, num_boards=boards, deck_mode=mode)


def cid(cfg) -> str:
    n, s, cards, boards, mode = cfg
    return f"{n}x{s}-{'+'.join(map(str, cards))}-b{boards}-{mode}"


def configs(fast, slow):
    return ([pytest.param(c, id=cid(c)) for c in fast]
            + [pytest.param(c, id=cid(c), marks=pytest.mark.slow) for c in slow])


class StreetBlindKeys(Holdem):
    """Our game with universal_poker's coarser key: board cards sorted together."""
    def infoset_key(self, s, player):
        board = self.deck.cards_str(sorted(c for b in s.boards for c in b))
        return f"{player}:{self.deck.cards_str(s.hands[player])}:{board}:{'/'.join(s.hist)}"


# One board: universal_poker ------------------------------------------------------------

ONE_BOARD_FAST = [(3, 2, (1,), 1, "shared"), (4, 2, (1,), 1, "shared"), (5, 2, (1,), 1, "shared"),
                  (3, 2, (2,), 1, "shared"), (4, 2, (3,), 1, "shared"), (3, 2, (1, 1), 1, "shared")]
ONE_BOARD_SLOW = [(4, 3, (1,), 1, "shared"), (4, 2, (1, 1), 1, "shared"), (3, 3, (2, 1), 1, "shared")]


@lru_cache(maxsize=None)
def universal_walk(cfg):
    up = UniversalPoker(cfg[0], cfg[1], cfg[2])
    return walk(up, up.describe)


@pytest.mark.parametrize("cfg", configs(ONE_BOARD_FAST, ONE_BOARD_SLOW))
def test_one_board_matches_universal_poker(cfg):
    theirs = universal_walk(cfg)
    if len(cfg[2]) == 1:
        d = diff(walk(Holdem(**params(cfg))), theirs)
        assert d.ok, f"\n{d}"
        return
    d = diff(walk(StreetBlindKeys(**params(cfg))), theirs)
    assert d.ok, f"\n{d}"
    # Our real key: everything agrees, and the only partition difference is that we keep
    # infosets apart that universal_poker merges (never the other way round).
    ours = walk(Holdem(**params(cfg)))
    d = diff(ours, theirs)
    assert d.caught <= {"infosets"}, f"\n{d}"
    assert all(e[0] == "the reference merges what we separate" for e in d.examples["infosets"])
    forward = {}
    for rec in ours.decisions:
        (pa, _, ka), (pb, _, kb) = ours.decisions[rec], theirs.decisions[rec]
        assert forward.setdefault((pa, ka), (pb, kb)) == (pb, kb), "we merge what universal_poker separates"


# Two boards: the reference -------------------------------------------------------------

TWO_BOARD_FAST = [(3, 2, (1,), 2, "shared"), (3, 2, (1,), 2, "identical"), (4, 2, (1,), 2, "shared"),
                  (3, 2, (1, 1), 2, "identical"), (4, 2, (2,), 2, "identical")]
TWO_BOARD_SLOW = [(3, 3, (1,), 2, "shared"), (4, 2, (1, 1), 2, "shared")]


@lru_cache(maxsize=None)
def reference_walk(cfg):
    ref = ReferenceHoldem(**params(cfg))
    return walk(ref, ref.describe)


@pytest.mark.parametrize("cfg", configs(TWO_BOARD_FAST, TWO_BOARD_SLOW))
def test_two_boards_match_reference(cfg):
    d = diff(walk(Holdem(**params(cfg))), reference_walk(cfg))
    assert d.ok, f"\n{d}"


@pytest.mark.parametrize("mutant, cfg, expected", [pytest.param(*m, id=m[0].__name__) for m in MUTANTS])
def test_mutant_is_caught(mutant, cfg, expected):
    d = diff(walk(mutant(**params(cfg))), reference_walk(cfg))
    assert expected <= d.caught, f"expected {expected}, caught {d.caught}\n{d}"
