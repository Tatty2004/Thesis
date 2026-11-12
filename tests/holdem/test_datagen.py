"""The value-net data generator (experiments/valuenet/datagen.py) on the small deck: the stored
values' units and their inverse transform back to the range solver's units, the river
templates of sampled pots, and the ranges. For the limit baseline and the real no-limit game."""
import numpy as np
import pytest

from bombpot.core.solvers.range_cfr import class_reach, class_values, close_values, forward, next_root, street_reach, \
    street_values
from bombpot.core.solvers.subgame import line_template, solve_subgame
from bombpot.experiments.valuenet.datagen import FlopPair, random_range, river_template, sample_spot, solve_sample
from bombpot.holdem.public import holdem_public_tree

FLOPS = ("2c3c4d", "5c6d4c")


@pytest.mark.parametrize("template", ["limit", "pot"])
def test_inverse_transform_gives_solver_units(template):
    rng = np.random.default_rng(0)
    s = sample_spot(rng, 5, 3)
    fp = FlopPair(s["flops"], 5, 3, template)
    s.update(fp.sample_betting(rng))
    row = solve_sample(fp, s, rng, tol=1e-3, iters=2000)
    assert row["subgame_exploitability"] <= 1e-3
    d = fp.turn_deal(s["turn"])
    valid = fp.upper.streets[1].valid[d]
    assert row["range0"].sum() == pytest.approx(1) and (row["range0"][~valid] == 0).all()
    assert np.array_equal(row["possible"], valid)
    assert abs(row["zero_sum_residual"]) < 1e-14  # cfv0 and cfv1 from one profile
    pot, tp = s["pot"], s["template"]
    # Re-solve with the same normalized setup to get the strategy behind the stored values.
    st = fp.river_street(d)
    sub = solve_subgame(st, tp, pot / 2, row["range0"], row["range1"], tol=1e-3 * pot, iters=2000)
    np.testing.assert_allclose(sub.values[0] / pot, row["cfv0"], rtol=0, atol=1e-15)
    # The real river (full chance weights) entered with unnormalized reaches (nonzero on
    # impossible hands too, as the solver's reach is), under that strategy.
    real = fp.river.street(*fp.river.all_children([d]))
    R = [row["range0"] * 3.7 + 0.5 * ~valid, row["range1"] * 0.02 + 0.3 * ~valid]
    D, H = real.num_deals, len(fp.hands)
    hsig = [x[:, None] for x in sub.strategy]
    nodes, _ = class_reach(tp, hsig, tuple(np.broadcast_to(r, (D, 1, H)).copy() for r in R))
    kappa_turn = fp.upper.streets[1].kappa[d]
    for p, key in ((0, "cfv0"), (1, "cfv1")):
        V, _ = class_values(tp, real, np.array([pot / 2]), hsig, nodes, p, "cfr", True)
        agg = V[0][:, 0].sum(0)
        R_o = np.where(valid, R[1 - p], 0.0)
        want = sub.values[p] * R_o.sum() * kappa_turn  # = cfv x pot x sum(R_o) x turn.kappa[d]
        np.testing.assert_allclose(agg, want, rtol=1e-11, atol=1e-12 * np.abs(want).max())
        np.testing.assert_allclose(agg, row[key] * pot * R_o.sum() * kappa_turn, rtol=1e-6,
                                   atol=1e-7 * np.abs(want).max())  # float64 row values here; float32 in shards


def test_walked_river_templates_match_the_real_lines():
    # river_template (used for pots spread around the real lines) reproduces the engine's own
    # river class template, chips included, at every real river line's pot.
    fp = FlopPair(FLOPS, 5, 3, "pot")
    for lc in fp.upper.classes[2]:
        for l in range(lc.num_lines):
            a, b = river_template(fp.game, lc.contrib[l]), line_template(lc.template, l)
            assert (a.players, a.actions, a.hist) == (b.players, b.actions, b.hist)
            assert [h for h, _ in a.closes] == [h for h, _ in b.closes]
            np.testing.assert_allclose([x for _, x in a.closes], [x for _, x in b.closes], rtol=1e-12)
            for ra, rb in zip(a.children, b.children):
                for x, y in zip(ra, rb):
                    assert x[:2] == y[:2]
                    if x[0] == "fold":
                        np.testing.assert_allclose(x[2], y[2], rtol=1e-12)
    rng = np.random.default_rng(3)
    for _ in range(20):
        b = fp.sample_betting(rng)
        assert 1.0 <= b["pot"] / 2 < fp.game.total and b["stack"] == pytest.approx(fp.game.total - b["pot"] / 2)
        assert 0.5 <= b["pot_spread"] <= 2.0 + 1e-12 and b["template"].num_nodes > 0


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


@pytest.mark.parametrize("template", ["limit", "pot"])
def test_rescaled_normalized_values_reproduce_the_real_river(template):
    # NetLeaf's rescale: agg[c][d, l, j, i] = cfv_pots[i] x pot x sum_{possible h} R_o[d, l, j, h]
    # x turn.kappa[d]. With exact cfv_pots (a fixed river strategy, river-only chance weights,
    # normalized ranges, as datagen stores them) it equals the real river's agg (the replay leaf
    # of tests/holdem/test_leaf.py) to 1e-12, for every river line of several turn deals.
    fp = FlopPair(FLOPS, 5, 3, template)
    full = holdem_public_tree(fp.game)
    upper, river = fp.upper, fp.river
    rng = np.random.default_rng(5)
    hsig = []
    for t in range(3):
        hsig.append([])
        for c, lc in enumerate(full.classes[t]):
            row = []
            for a in lc.template.actions:
                x = rng.random(full.shape(t, c) + (len(a),)) ** 3
                row.append(x / x.sum(-1, keepdims=True))
            hsig[-1].append(row)
    _, closing = forward(upper, hsig[:2], closing=True)
    D1, H = upper.streets[1].num_deals, len(upper.hands)
    # The replay leaf: the real river, every turn deal.
    st_all = river.street(*river.all_children(np.arange(D1)))
    cl = [[(r0[:, :, j], r1[:, :, j]) for j in range(r0.shape[2])] for r0, r1 in closing]
    roots = next_root(cl, full.classes[1], full.classes[2], st_all.parent)
    nodes, _ = street_reach(full.classes[2], hsig[2], roots)
    want = []
    for p in (0, 1):
        V, _ = street_values(full.classes[2], st_all, hsig[2], nodes, p, "cfr", True)
        want.append(close_values([v[0] for v in V], full.classes[1], full.classes[2], st_all.parent, D1))
    kappa_turn = upper.streets[1].kappa
    checked = 0
    for d in (0, 29, 71):
        st = fp.river_street(d)  # river-only chance weights
        rows = np.flatnonzero(st_all.parent == d)
        possible = upper.streets[1].valid[d]
        for c, rc in enumerate(full.classes[2]):
            for m in range(rc.num_lines):
                pc, pl, pj = rc.parent_class[m], rc.parent_line[m], rc.parent_close[m]
                pot = 2 * rc.contrib[m]
                tp = line_template(rc.template, m)
                R = [closing[pc][q][d, pl, pj] for q in (0, 1)]
                r = [np.where(possible, x, 0) / np.where(possible, x, 0).sum() for x in R]
                h = [s[rows][:, m:m + 1] for s in hsig[2][c]]
                nd, _ = class_reach(tp, h, tuple(np.broadcast_to(x, (len(rows), 1, H)).copy() for x in r))
                for p in (0, 1):
                    V, _ = class_values(tp, st, np.array([pot / 2]), h, nd, p, "cfr", True)
                    cfv = V[0][:, 0].sum(0) / pot
                    agg = cfv * pot * np.where(possible, R[1 - p], 0).sum() * kappa_turn[d]
                    w = want[p][pc][d, pl, pj]
                    np.testing.assert_allclose(agg, w, rtol=1e-12, atol=1e-12 * np.abs(w).max())
                checked += 1
    assert checked == 3 * full.num_lines(2)


@pytest.mark.parametrize("template", ["limit", "pot"])
def test_fused_solver_gives_the_same_rows_as_the_subgame_solver(template):
    # The default solver (core/solvers/fused.py) against the batched one: the same sample,
    # every stored number equal in float64; float32 solves to the same tolerance.
    rows = {}
    for key, kw in (("fused", {}), ("subgame", {"solver": "subgame"}), ("f32", {"dtype": "float32"})):
        rng = np.random.default_rng(3)
        s = sample_spot(rng, 5, 3)
        fp = FlopPair(s["flops"], 5, 3, template)
        s.update(fp.sample_betting(rng))
        rows[key] = solve_sample(fp, s, rng, tol=1e-3, iters=2000, variant="dcfr", **kw)
    a, b, c = rows["fused"], rows["subgame"], rows["f32"]
    assert 0 < a["iters"] == b["iters"] < 2000
    for k in ("range0", "range1", "possible", "equity", "cfv0", "cfv1"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    for k in ("zero_sum_residual", "subgame_exploitability", "mass"):
        assert a[k] == b[k], k
    assert c["subgame_exploitability"] <= 1e-3
    assert np.abs(c["cfv0"] - a["cfv0"]).max() < 0.05 and np.abs(c["cfv1"] - a["cfv1"]).max() < 0.05
    with pytest.raises(ValueError):
        solve_sample(fp, s, rng, tol=1e-3, iters=10, solver="subgame", dtype="float32")


def test_best_response_values_are_stored_and_bound_the_profile_values():
    # br_p[i] >= cfv_p[i] for every hand, and the range-weighted gap is the exploitability
    # (README: range0 . br0 + range1 . br1 = 2 x subgame_exploitability x mass).
    rng = np.random.default_rng(5)
    s = sample_spot(rng, 5, 3)
    fp = FlopPair(s["flops"], 5, 3, "limit", "lines")
    s.update(fp.sample_betting(rng, 1.5))
    assert s["pot"] / s["pot_spread"] in set((2 * fp.upper.classes[2][0].contrib).tolist())
    row = solve_sample(fp, s, rng, tol=1e-3, iters=2000, variant="dcfr")
    for p in (0, 1):
        assert (row[f"br{p}"] >= row[f"cfv{p}"] - 1e-12).all()
        assert (row[f"br{p}"][~row["possible"]] == 0).all()
    total = row["range0"] @ row["br0"] + row["range1"] @ row["br1"]
    assert total == pytest.approx(2 * row["subgame_exploitability"] * row["mass"], rel=1e-9, abs=1e-14)
