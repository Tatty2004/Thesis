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
    assert np.array_equal(row["possible"], valid)
    assert abs(row["zero_sum_residual"]) < 1e-14  # cfv0 and cfv1 from one profile
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


def test_rescaled_normalized_values_reproduce_the_real_river():
    # NetLeaf's rescale: agg[d, l, c, i] = cfv_pots[i] x pot[l, c] x sum_{possible j} R_o[d, l, c, j]
    # x turn.kappa[d]. With exact cfv_pots (a fixed river strategy, river-only chance weights,
    # normalized ranges, as datagen stores them) it equals the real river's agg (the replay leaf
    # of tests/holdem/test_leaf.py) to 1e-12, for every line and close of several turn deals.
    from bombpot.core.solvers.range_cfr import close_values, forward, next_root
    from bombpot.holdem.public import holdem_public_tree

    fp = FlopPair(("2c3c4d", "5c6d4c"), 5, 3)
    full = holdem_public_tree(fp.game)
    upper, river = fp.upper, fp.river
    rng = np.random.default_rng(5)
    hsig = []
    for t, tp in enumerate(full.templates):
        hsig.append([])
        for a in tp.actions:
            x = rng.random(full.shape(t) + (len(a),)) ** 3
            hsig[-1].append(x / x.sum(-1, keepdims=True))
    _, closing = forward(upper, hsig[:2], closing=True)
    D1, L, C, H = closing[0].shape
    # The replay leaf: the real river, every turn deal.
    st_all = river.street(*river.all_children(np.arange(D1)))
    tp = full.templates[2]
    root = next_root([(closing[0][:, :, c], closing[1][:, :, c]) for c in range(C)], st_all.parent)
    nodes, _ = street_reach(tp, hsig[2], root)
    want = []
    for p in (0, 1):
        V, _ = street_values(tp, st_all, full.contrib[2], hsig[2], nodes, p, "cfr", True)
        want.append(close_values(V[0], st_all.parent, D1, C))
    kappa_turn = upper.streets[1].kappa
    for d in (0, 29, 71):
        st = fp.river_street(d)  # river-only chance weights
        rows = np.flatnonzero(st_all.parent == d)
        possible = upper.streets[1].valid[d]
        for l in range(L):
            for c in range(C):
                line = l * C + c
                pot = 2 * full.contrib[2][line]
                R = [closing[q][d, l, c] for q in (0, 1)]
                r = [np.where(possible, x, 0) / np.where(possible, x, 0).sum() for x in R]
                h = [s[rows][:, line:line + 1] for s in hsig[2]]
                nd, _ = street_reach(tp, h, tuple(np.broadcast_to(x, (len(rows), 1, H)).copy() for x in r))
                for p in (0, 1):
                    V, _ = street_values(tp, st, np.array([pot / 2]), h, nd, p, "cfr", True)
                    cfv = V[0][:, 0].sum(0) / pot
                    agg = cfv * pot * np.where(possible, R[1 - p], 0).sum() * kappa_turn[d]
                    w = want[p][d, l, c]
                    np.testing.assert_allclose(agg, w, rtol=1e-12, atol=1e-12 * np.abs(w).max())
