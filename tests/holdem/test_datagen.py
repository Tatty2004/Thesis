"""The value-net data generator (experiments/valuenet/datagen.py) on the small deck: the stored
values' units and their inverse transform back to the range solver's units, and the ranges."""
import numpy as np
import pytest

from bombpot.core.solvers.range_cfr import street_reach, street_values
from bombpot.experiments.valuenet.datagen import FlopPair, random_range, sample_spot, solve_sample


@pytest.fixture(scope="module")
def spot():
    rng = np.random.default_rng(0)
    s = sample_spot(rng, 5, 3)
    return s, FlopPair(s["flops"], 5, 3), rng


def test_inverse_transform_gives_solver_units(spot):
    s, fp, rng = spot
    row = solve_sample(fp, s, rng, tol=1e-3, iters=2000)
    assert row["subgame_exploitability"] <= 1e-3
    d = fp.turn_deal(s["turn"])
    valid = fp.upper.streets[1].valid[d]
    assert row["range0"].sum() == pytest.approx(1) and (row["range0"][~valid] == 0).all()
    pot = s["pot"]
    # Re-solve with the same normalized setup to get the strategy behind the stored values.
    from bombpot.core.solvers.subgame import solve_subgame
    st = fp.river_street(d)
    sub = solve_subgame(st, fp.upper.templates[2], pot / 2, row["range0"], row["range1"], tol=1e-3 * pot, iters=2000)
    np.testing.assert_allclose(sub.values[0] / pot, row["cfv0"], rtol=0, atol=1e-15)
    # The real river (full chance weights) entered with unnormalized reaches (nonzero on
    # impossible hands too, as the solver's reach is), under that strategy.
    real = fp.river.street(*fp.river.all_children([d]))
    R = [row["range0"] * 3.7 + 0.5 * ~valid, row["range1"] * 0.02 + 0.3 * ~valid]
    D, H = real.num_deals, len(fp.hands)
    hsig = [x[:, None] for x in sub.strategy]
    nodes, _ = street_reach(fp.upper.templates[2], hsig, tuple(np.broadcast_to(r, (D, 1, H)).copy() for r in R))
    kappa_turn = fp.upper.streets[1].kappa[d]
    for p, key in ((0, "cfv0"), (1, "cfv1")):
        V, _ = street_values(fp.upper.templates[2], real, np.array([pot / 2]), hsig, nodes, p, "cfr", True)
        agg = V[0][:, 0].sum(0)
        R_o = np.where(valid, R[1 - p], 0.0)
        want = sub.values[p] * R_o.sum() * kappa_turn  # = cfv x pot x sum(R_o) x turn.kappa[d]
        np.testing.assert_allclose(agg, want, rtol=1e-11, atol=1e-12 * np.abs(want).max())
        np.testing.assert_allclose(agg, row[key] * pot * R_o.sum() * kappa_turn, rtol=1e-6,
                                   atol=1e-7 * np.abs(want).max())  # float64 row values here; float32 in shards


def test_random_ranges():
    rng = np.random.default_rng(1)
    valid = rng.random(200) < 0.8
    strength = rng.random(200)
    peaks = []
    for _ in range(200):
        r = random_range(rng, valid, strength)
        assert r.sum() == pytest.approx(1) and (r[~valid] == 0).all() and (r >= 0).all()
        peaks.append(r.max())
    peaks = np.array(peaks)
    assert (peaks < 2.0 / valid.sum()).any() and (peaks > 0.2).any()  # some near-uniform, some peaked
