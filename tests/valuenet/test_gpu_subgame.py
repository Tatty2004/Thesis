"""The torch subgame solver (valuenet/gpu_subgame.py) against the compiled one (core/solvers/fused.py):
the same sums and, in float64, the same iterates to rounding. Runs on the CPU through torch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bombpot.core.solvers.fused import FusedSubgames  # noqa: E402
from bombpot.experiments.valuenet.datagen import FlopPair, random_range, turn_equity  # noqa: E402
from bombpot.valuenet.gpu_subgame import GpuSubgame, StreetSums  # noqa: E402


def _spot(template: str, seed: int):
    rng = np.random.default_rng(seed)
    fp = FlopPair(((0, 3, 7), (9, 11, 6)), 5, 3, template, "lines")  # 2c3c4d / 5c6d4c
    rest = rng.permutation([c for c in range(15) if c not in (0, 3, 7, 9, 11, 6)])
    d = fp.turn_deal((int(rest[0]), int(rest[1])))
    st = fp.river_street(d)
    valid = fp.upper.streets[1].valid[d]
    strength = turn_equity(st).mean(axis=1)
    r0, r1 = random_range(rng, valid, strength), random_range(rng, valid, strength)
    bet = fp.sample_betting(rng, 2.0)
    return st, bet["template"], bet["pot"] / 2, r0, r1


@pytest.mark.parametrize("seed", [0, 1])
def test_sums_match_the_street(seed):
    st, _, _, r0, r1 = _spot("pot", seed)
    sums = StreetSums(st, "cpu", torch.float64)
    rng = np.random.default_rng(seed)
    x = rng.random((3, st.num_deals, len(r0)))
    f = sums.fold(torch.as_tensor(x)).numpy()
    b = sums.beats(torch.as_tensor(x)).numpy()
    for t in range(3):
        xt = np.ascontiguousarray(x[t][:, None, :])
        np.testing.assert_allclose(f[t], st.fold(xt, 0)[:, 0], rtol=0, atol=1e-13)
        np.testing.assert_allclose(b[t], st.share(xt, 0)[:, 0], rtol=0, atol=1e-13)


def _node_index(ref, k):
    A = ref.nact[0, k]
    return ref.base[0] + np.arange(ref.D)[:, None] * ref.size[0] + ref.off[0, k] + np.arange(ref.H * A)


def _copy_state(ref, gpu):
    """The compiled solver's state into the torch one."""
    for name in ("regret", "cum", "sigma"):
        for k in range(gpu.K):
            getattr(gpu, name)[k] = torch.as_tensor(ref.node_array(getattr(ref, name), 0, k))
    gpu.t = int(ref.t[0])


def _cpu_evaluation(st, tp, contrib, r0, r1, gpu):
    """The torch solver's average strategy graded by the compiled evaluator."""
    ref = FusedSubgames(st, [tp], [contrib], [(r0, r1)], gpu.variant)
    for k in range(gpu.K):
        ref.cum[_node_index(ref, k)] = gpu.cum[k].double().cpu().numpy().reshape(ref.D, -1)
    return ref.evaluate()


@pytest.mark.parametrize("template,seed", [("pot", 0), ("pot", 3), ("limit", 1)])
@pytest.mark.parametrize("variant", ["dcfr", "cfr+"])
def test_one_iteration_matches_fused(template, seed, variant):
    """From the same state, one iteration gives the same regrets and average sums, and the
    evaluation the same values. (Free-running trajectories part ways at regrets that are an
    exact 0 in one solver and rounding in the other, as fused.py says of float32, so each
    step is compared from the compiled solver's state.)"""
    st, tp, contrib, r0, r1 = _spot(template, seed)
    ref = FusedSubgames(st, [tp], [contrib], [(r0, r1)], variant)
    gpu = GpuSubgame(st, tp, contrib, r0, r1, variant, dtype=torch.float64, device="cpu")
    for n in (0, 1, 8, 20):
        ref.iterate(n) if n else None
        _copy_state(ref, gpu)
        a, b = ref.evaluate(), gpu.evaluate()
        for p in (0, 1):
            np.testing.assert_allclose(b["values"][p], a["values"][0][p], rtol=0, atol=1e-10)
            np.testing.assert_allclose(b["br_values"][p], a["br_values"][0][p], rtol=0, atol=1e-10)
        assert b["exploitability"] == pytest.approx(a["exploitability"][0], abs=1e-10)
        assert b["mass"] == pytest.approx(a["mass"][0], rel=1e-12)
        ref.iterate(1)
        gpu.iterate(1)
        for k in range(gpu.K):
            for name in ("regret", "cum"):
                x = ref.node_array(getattr(ref, name), 0, k)
                # player 0 updates first, from the common state: rounding only. Player 1 then
                # plays against player 0's new strategy, which differs between the solvers
                # where a regret is an exact 0 in one and rounding in the other (hands with
                # next to no counterfactual reach), so its regrets agree less tightly.
                tight = name == "cum" or gpu.player[k] == 0
                np.testing.assert_allclose(getattr(gpu, name)[k].numpy(), x, rtol=0,
                                           atol=(1e-11 if tight else 1e-6) * max(1.0, np.abs(x).max()))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("seed", [2, 5])
def test_solves_to_the_tolerance(seed, dtype):
    """A solve to a tolerance, graded by the compiled evaluator: within the tolerance, in about
    as many iterations as the compiled solver, with values that agree to about the tolerance."""
    st, tp, contrib, r0, r1 = _spot("pot", seed)
    tol = 1e-3 * 2 * contrib
    gpu = GpuSubgame(st, tp, contrib, r0, r1, "dcfr", dtype=dtype, device="cpu")
    ev = gpu.solve(500, tol)
    cpu = _cpu_evaluation(st, tp, contrib, r0, r1, gpu)
    assert cpu["exploitability"][0] <= tol * (1 + 1e-4)
    assert ev["exploitability"] == pytest.approx(cpu["exploitability"][0], abs=1e-4 * tol + (1e-9 if dtype == torch.float64 else 1e-5))
    ref = FusedSubgames(st, [tp], [contrib], [(r0, r1)], "dcfr").solve(500, tol)
    assert abs(ev["iterations"] - ref["iterations"][0]) <= 20
    for p, r in ((0, r0), (1, r1)):
        assert abs(((ev["values"][p] - ref["values"][0][p]) * r).sum()) / ev["mass"] <= 2 * tol
