"""The fused last-street solver (core/solvers/fused.py) against SubgameCFR on the small
three-street deck (5 ranks x 3 suits, 36 hands, 42 river deals per turn deal).

FusedSubgames runs the range solver's arithmetic one deal at a time inside one compiled
kernel. In float64 everything it produces is compared with `==`: regrets, average-strategy
sums and current strategy after every iteration (free-running, not in lockstep: identical
arithmetic leaves no ties to resolve differently), root values, best-response values,
exploitability, and the iteration at which a tolerance stops it. Limit and no-limit
templates (with all-in actions and the shorter shapes they leave), CFR, CFR+ and DCFR.
float32 is compared to rounding. Batches of several subgames with different templates,
chips, ranges and tolerances equal each subgame solved alone. The kernels' parallel and
serial variants agree (as tests/test_kernels.py checks for the other kernels).
"""
import numpy as np
import pytest

from bombpot.core.solvers import fused
from bombpot.core.solvers.fused import FusedSubgames, solve_subgame_fused
from bombpot.core.solvers.subgame import SubgameCFR, line_template, solve_subgame
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

FLOPS = ("2c3c4d", "5c6d4c")
GAMES = {
    "limit": Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=FLOPS, bet_sizes=(2, 4, 4)),
    "pot": Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=FLOPS, betting="pot", stack=20,
                  sizes=(0.5, 1), max_aggressive=2),
}
VARIANTS = ["cfr", "cfr+", "dcfr"]
# (game, river class, line in the class, turn deal). The pot game's classes 1 to 5 have
# all-in actions: 14, 12, 10, 8 and 4 nodes (class 6 is all-in before the river: no betting).
SPOTS = [("limit", 0, 0, 0), ("limit", 0, 12, 50), ("pot", 0, 0, 7), ("pot", 1, 2, 30), ("pot", 3, 5, 17),
         ("pot", 4, 9, 3), ("pot", 5, 40, 61)]


@pytest.fixture(scope="module")
def trees():
    out = {}
    for name, game in GAMES.items():
        upper = holdem_public_tree(game, upto=2)
        out[name] = (upper, HoldemRiver(game, upper))
    return out


def spot(trees, game, c, line, d):
    """(river street of turn deal d, one-line template, chips in) of a river line."""
    upper, river = trees[game]
    lc = upper.classes[2][c]
    return river.street(*river.all_children([d])), line_template(lc.template, line), float(lc.contrib[line])


def ranges(st, seed, kind="dense"):
    """Root ranges over the hands; "sparse" has exact zeros (most hands), as peaked ranges do."""
    rng = np.random.default_rng(seed)
    H = len(st.hands)
    r = [rng.random(H) ** 2 + 0.01 for _ in (0, 1)]
    if kind == "sparse":
        r = [x * (rng.random(H) < 0.3) for x in r]
        r[1][:] = 0.0
        r[1][rng.choice(H, 3, replace=False)] = 1.0
    return [x / x.sum() for x in r]


def assert_same_state(a: SubgameCFR, f: FusedSubgames, s: int = 0):
    for k in range(a.template.num_nodes):
        for name in ("regret", "cum", "sigma"):
            np.testing.assert_array_equal(getattr(a, name)[0][0][k][:, 0], f.node_array(getattr(f, name), s, k),
                                          err_msg=f"{name} at node {k}, iteration {a.t}")


def assert_same_evaluation(ea: dict, ef: dict, s: int = 0):
    for key in ("values", "br_values"):
        for p in (0, 1):
            np.testing.assert_array_equal(ea[key][p][0, 0], ef[key][s][p], err_msg=key)
    for key in ("exploitability", "exploitability_raw", "mass"):
        assert ea[key][0, 0] == ef[key][s], key


def test_spots_cover_the_all_in_shapes(trees):
    seen = set()
    for game, c, line, d in SPOTS:
        _, tp, _ = spot(trees, game, c, line, d)
        seen.add((tp.num_nodes, any("a" in acts for acts in tp.actions)))
    assert {(14, False), (14, True), (10, True), (8, True), (4, True)} <= seen


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("game, c, line, d", SPOTS)
def test_float64_iterates_and_values_equal_subgame_cfr_bit_for_bit(trees, game, c, line, d, variant):
    st, tp, contrib = spot(trees, game, c, line, d)
    r = ranges(st, d, "sparse" if c % 2 else "dense")
    a = SubgameCFR(st, tp, [contrib], (r[0], r[1]), variant=variant)
    f = FusedSubgames(st, [tp], [contrib], [(r[0], r[1])], variant=variant)
    assert_same_state(a, f)  # the uniform start
    for it in range(25):
        a.iteration()
        f.iterate(1)
        assert_same_state(a, f)
        if it % 6 == 0:
            assert_same_evaluation(a.evaluate(), f.evaluate())
    assert_same_evaluation(a.evaluate(), f.evaluate())
    for x, y in zip(a.hand_sigma(a.average_strategy())[0][0], f.average_strategy(0)):
        np.testing.assert_array_equal(x[:, 0], y)


@pytest.mark.parametrize("variant", VARIANTS)
def test_many_iterations_in_one_call_equal_one_at_a_time(trees, variant):
    st, tp, contrib = spot(trees, "pot", 1, 2, 30)
    r = ranges(st, 3)
    a = FusedSubgames(st, [tp], [contrib], [(r[0], r[1])], variant=variant)
    b = FusedSubgames(st, [tp], [contrib], [(r[0], r[1])], variant=variant)
    for _ in range(12):
        a.iterate(1)
    b.iterate(5)
    b.iterate(7)
    assert (a.t == 12).all() and (b.t == 12).all()
    for name in ("regret", "cum", "sigma"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("variant", ["cfr+", "dcfr"])
@pytest.mark.parametrize("game, c, line, d", [SPOTS[1], SPOTS[2], SPOTS[4]])
def test_solve_to_a_tolerance_equals_solve_subgame(trees, game, c, line, d, variant):
    st, tp, contrib = spot(trees, game, c, line, d)
    r = ranges(st, 100 + d)
    tol = 2e-3 * 2 * contrib
    want = solve_subgame(st, tp, contrib, r[0], r[1], variant, 2000, tol)
    got = solve_subgame_fused(st, tp, contrib, r[0], r[1], variant, 2000, tol)
    assert 0 < want.iterations < 2000 and got.iterations == want.iterations
    assert got.exploitability == want.exploitability <= tol
    assert got.exploitability_raw == want.exploitability_raw and got.mass == want.mass
    for p in (0, 1):
        np.testing.assert_array_equal(got.values[p], want.values[p])
        np.testing.assert_array_equal(got.br_values[p], want.br_values[p])
    for x, y in zip(got.strategy, want.strategy):
        np.testing.assert_array_equal(x, y)
    # the iteration cap, and no tolerance at all
    capped = solve_subgame_fused(st, tp, contrib, r[0], r[1], variant, 7, tol)
    ref = solve_subgame(st, tp, contrib, r[0], r[1], variant, 7, tol)
    free = solve_subgame_fused(st, tp, contrib, r[0], r[1], variant, 7, None)
    assert capped.iterations == free.iterations == ref.iterations == 7
    for p in (0, 1):
        np.testing.assert_array_equal(capped.values[p], ref.values[p])
        np.testing.assert_array_equal(free.values[p], ref.values[p])


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_batch_equals_each_subgame_alone_and_each_stops_on_its_own_tolerance(trees, variant):
    # Four subgames at one turn deal: different templates (14, 10 and 4 nodes, 2 to 4 actions),
    # chips, ranges and tolerances, so they stop at different iterations.
    upper, river = trees["pot"]
    st = river.street(*river.all_children([22]))
    picks = [(0, 1, "dense", 3e-2), (3, 7, "sparse", 1e-3), (5, 11, "dense", 3e-3), (0, 4, "dense", None)]
    tps, contribs, roots, tols = [], [], [], []
    for s, (c, line, kind, tol) in enumerate(picks):
        lc = upper.classes[2][c]
        tps.append(line_template(lc.template, line))
        contribs.append(float(lc.contrib[line]))
        roots.append(tuple(ranges(st, 40 + s, kind)))
        tols.append(np.inf if tol is None else tol * 2 * contribs[-1])
    cap = 60 if variant != "cfr" else 30
    tols[3] = 0.0  # never reached: this one runs to the cap
    f = FusedSubgames(st, tps, contribs, roots, variant=variant)
    out = f.solve(cap, tols)
    for s in range(len(picks)):
        want = solve_subgame(st, tps[s], contribs[s], roots[s][0], roots[s][1], variant, cap, tols[s])
        assert out["iterations"][s] == want.iterations
        assert out["exploitability"][s] == want.exploitability and out["mass"][s] == want.mass
        for p in (0, 1):
            np.testing.assert_array_equal(out["values"][s][p], want.values[p])
            np.testing.assert_array_equal(out["br_values"][s][p], want.br_values[p])
        for x, y in zip(f.average_strategy(s), want.strategy):
            np.testing.assert_array_equal(x, y)
    assert out["iterations"][3] == cap
    if variant != "cfr":  # the tolerances really differ in when they stop
        assert len(set(out["iterations"])) >= 3
        assert all(out["exploitability"][s] <= tols[s] for s in range(3))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("game, c, line, d", [SPOTS[0], SPOTS[2], SPOTS[3], SPOTS[6]])
def test_float32_agrees_to_rounding(trees, game, c, line, d, variant):
    st, tp, contrib = spot(trees, game, c, line, d)
    r = ranges(st, 7 + d, "sparse" if c % 2 else "dense")
    a = FusedSubgames(st, [tp], [contrib], [(r[0], r[1])], variant=variant)
    b = FusedSubgames(st, [tp], [contrib], [(r[0], r[1])], variant=variant, dtype=np.float32)
    assert b.regret.dtype == b.cum.dtype == b.sigma.dtype == np.float32
    # One iteration from the same (uniform) state.
    a.iterate(1)
    b.iterate(1)
    # Rounding is relative to the terminal values (chips x chance weight x a reach of about 1,
    # where the card-removal sums cancel), not to the regrets, which are their differences.
    chips = np.abs(a.coef).max()
    # Player 1 updates after player 0, whose new strategy can differ where a regret is an exact
    # 0 in one precision and rounding in the other (uniform against pure: the exact ties of
    # tests/test_range_cfr.py, which sparse ranges produce), so its regrets are compared only
    # for dense ranges. The average after one iteration is the uniform start either way.
    for k, q in enumerate(tp.players):
        if q == 0 or c % 2 == 0:
            np.testing.assert_allclose(b.node_array(b.regret, 0, k), a.node_array(a.regret, 0, k), rtol=0,
                                       atol=4e-6 * chips * st.kappa.max())
    np.testing.assert_allclose(b.cum, a.cum, rtol=0, atol=2e-6 * np.abs(a.cum).max())
    ea, eb = a.evaluate(), b.evaluate()
    pot = 2 * contrib
    for key in ("values", "br_values"):
        for p in (0, 1):
            np.testing.assert_allclose(eb[key][0][p], ea[key][0][p], rtol=0, atol=4e-6 * chips * st.kappa.sum())
    # Solved: the values differ by rounding plus what the tolerance leaves open, and the
    # float32 solve's exploitability, graded in float64 arithmetic on its own average, is small.
    tol = 2e-3 * pot
    x = solve_subgame_fused(st, tp, contrib, r[0], r[1], variant, 400, tol)
    y = solve_subgame_fused(st, tp, contrib, r[0], r[1], variant, 400, tol, dtype=np.float32)
    assert abs(y.exploitability - x.exploitability) < 2 * tol
    if variant != "cfr":
        assert y.exploitability <= tol  # its own path there: iteration counts can differ (ties)
    g = SubgameCFR(st, tp, [contrib], (r[0], r[1]))  # float64 grading of float32's strategy
    exact = g.evaluate([[[s[:, None] for s in y.strategy]]])
    mass_scale = chips * st.kappa.sum()
    assert exact["exploitability_raw"][0, 0] == pytest.approx(y.exploitability_raw, abs=1e-5 * mass_scale)
    for p in (0, 1):
        np.testing.assert_allclose(y.values[p], exact["values"][p][0, 0], rtol=0, atol=1e-5 * mass_scale)


def test_a_template_without_betting_is_refused(trees):
    upper, river = trees["pot"]
    lc = [x for x in upper.classes[2] if x.template.num_nodes == 0][0]
    st = river.street(*river.all_children([3]))
    r = ranges(st, 0)
    with pytest.raises(ValueError, match="nothing to solve"):
        FusedSubgames(st, [line_template(lc.template, 0)], [lc.contrib[0]], [(r[0], r[1])])


# Parallel and serial kernel variants ------------------------------------------------------


def batch(trees, variant, dtype):
    upper, river = trees["pot"]
    st = river.street(*river.all_children([9]))
    tps, contribs, roots = [], [], []
    for s, (c, line) in enumerate([(0, 2), (2, 3), (4, 1), (5, 20)]):
        lc = upper.classes[2][c]
        tps.append(line_template(lc.template, line))
        contribs.append(float(lc.contrib[line]))
        roots.append(tuple(ranges(st, 70 + s, "sparse" if s == 1 else "dense")))
    return FusedSubgames(st, tps, contribs, roots, variant=variant, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("variant", VARIANTS)
def test_kernels_parallel_equals_serial(trees, variant, dtype, monkeypatch):
    # FusedSubgames picks a variant by problem size (core/jit.py): force each in turn.
    states, evals = [], []
    active = np.array([True, True, False, True])
    for which in ("serial", "parallel"):
        for k in ("_iterate_kernel", "_evaluate_kernel"):
            real = getattr(fused, k)
            monkeypatch.setattr(fused, k, lambda work, *args, _v=getattr(real, which): _v(*args))
        f = batch(trees, variant, dtype)
        f.iterate(4)
        f.iterate(3, active)  # an inactive subgame is left untouched
        states.append((f.regret.copy(), f.cum.copy(), f.sigma.copy(), f.t.copy()))
        evals.append((f.evaluate(), f.evaluate(active)))
        monkeypatch.undo()
    for x, y in zip(*states):
        np.testing.assert_array_equal(x, y)
    assert list(states[0][3]) == [7, 7, 4, 7]
    for ea, eb in zip(*evals):
        for key in ea:
            for x, y in zip(ea[key], eb[key]):
                assert (x is None) == (y is None)
                if x is not None:
                    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert evals[0][1]["values"][2] is None and evals[0][0]["values"][2] is not None


def test_kernel_variants_are_compiled_separately():
    for k in (fused._iterate_kernel, fused._evaluate_kernel):
        assert k.serial.py_func.__qualname__ != k.parallel.py_func.__qualname__
        assert k.serial.targetoptions.get("parallel", False) is False
        assert k.parallel.targetoptions["parallel"] is True
