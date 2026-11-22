"""The viewer's helpers: chips and labels against the engine's templates, and in-place npz maps."""
import numpy as np
import pytest

pytest.importorskip("torch")

from bombpot.experiments.valuenet import datagen as dg
from bombpot.experiments.valuenet import viewer_server as vs
from bombpot.holdem.public import holdem_public_tree


def test_chips_match_the_templates():
    game = dg.make_game(((0, 5, 9), (2, 7, 11)), 5, 3, "pot")
    upper = holdem_public_tree(game, upto=2)
    n = 0
    for row in upper.classes:
        for lc in row:
            for h, added in lc.template.closes:
                for l in range(lc.num_lines):
                    assert max(vs.chips(h, lc.contrib[l], lc.stack[l])) == pytest.approx(added[l], abs=1e-12)
                    n += 1
    assert n > 20


def test_labels():
    assert vs.label("c", "", 1, 100) == "Check"
    assert vs.label("r0.5", "", 1, 100) == "Bet 1"
    assert vs.label("c", "r1", 1, 100) == "Call 2"
    assert vs.label("r0.5", "r1", 3, 98) == "Raise to 15"  # bet 6, call 6 makes 18, half of it on top
    assert vs.label("a", "r1", 54, 47) == "All-in to 47"
    assert vs.label("f", "r1", 1, 100) == "Fold"


def test_npz_memmaps(tmp_path):
    a, b = np.arange(24, dtype=np.float32).reshape(2, 3, 4), np.arange(5.0)
    np.savez(tmp_path / "x.npz", t0_c0_k0=a, other=b)
    m = vs.npz_memmaps(tmp_path / "x.npz")
    assert np.array_equal(m["t0_c0_k0"], a) and np.array_equal(m["other"], b)
