"""M1 plumbing: saving and resuming the sampled solver, and saving river buckets.

On the small three-street game of test_holdem_river.py (5 ranks x 3 suits, fixed flops),
with the lossless river and with fitted kmeans_2d buckets:
  - a saved solver loads into a fresh one with bit-identical arrays, counters and random
    generator state;
  - stopping, saving, loading and going on gives bit-identical regrets, averages and
    strategies to a run that never stopped, and the same next sample;
  - the exact evaluation of a loaded solver's average strategy equals the in-memory one's;
  - saved and loaded buckets give the same ids;
  - loading refuses a checkpoint of another shape and changes nothing;
  - the evaluation's progress callback reports every chunk count and changes no result.
"""
import numpy as np
import pytest

from bombpot.core.abstraction.river import EquityBuckets, fit_equity_buckets
from bombpot.core.solvers.sampled_cfr import Lossless, SampledRangeCFR, exact_exploitability
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

GAME = Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=("2c3c4d", "5c6d4c"), bet_sizes=(2, 4, 4))
NAMES = ["lossless", "kmeans_2d"]


@pytest.fixture(scope="module")
def setup():
    upper = holdem_public_tree(GAME, upto=2)
    river = HoldemRiver(GAME, upper)
    D1 = upper.streets[-1].num_deals
    abstractions = {"lossless": Lossless(D1, river.num_children, len(upper.hands)),
                    "avg_1d": fit_equity_buckets(river, D1, "avg_1d", 3, samples=12, seed=0),
                    "kmeans_2d": fit_equity_buckets(river, D1, "kmeans_2d", 3, samples=12, seed=0)}
    return upper, river, abstractions


def make(setup, name, abstraction=None, batch=8, samples=2, seed=5):
    upper, river, abstractions = setup
    return SampledRangeCFR(upper, river, abstraction or abstractions[name], batch=batch, samples=samples, seed=seed)


def run(solver, iterations):
    for _ in range(iterations):
        solver.iteration()
    return solver


def bits(x: np.ndarray) -> np.ndarray:
    return x.view(np.uint64)


def assert_identical(a, b):
    """Every regret, average and strategy array, bit for bit."""
    for name in ("regret", "cum", "sigma"):
        rows_a, rows_b = getattr(a, name), getattr(b, name)
        assert [len(r) for r in rows_a] == [len(r) for r in rows_b]
        for row_a, row_b in zip(rows_a, rows_b):
            for x, y in zip(row_a, row_b):
                assert x.shape == y.shape and x.dtype == y.dtype == np.float64
                assert np.array_equal(bits(x), bits(y)), name


@pytest.mark.parametrize("name", NAMES)
def test_round_trip(setup, name, tmp_path):
    solver = run(make(setup, name), 4)
    solver.save(tmp_path / "state.npz")
    assert [p.name for p in tmp_path.iterdir()] == ["state.npz"]  # no temporary file left behind
    fresh = make(setup, name, batch=3, samples=1, seed=99)  # load restores batch, samples and seed
    fresh.load(tmp_path / "state.npz")
    assert_identical(solver, fresh)
    assert (fresh.t, fresh.solve_seconds, fresh.value) == (solver.t, solver.solve_seconds, solver.value)
    assert fresh.rng.bit_generator.state == solver.rng.bit_generator.state
    assert (fresh.batch, fresh.samples, fresh.seed) == (8, 2, 5)


@pytest.mark.parametrize("name", NAMES)
def test_resume_is_bit_identical(setup, name, tmp_path):
    straight = run(make(setup, name), 10)
    run(make(setup, name), 5).save(tmp_path / "state.npz")
    resumed = make(setup, name, seed=123)  # the generator's state comes from the file
    resumed.load(tmp_path / "state.npz")
    run(resumed, 5)
    assert resumed.t == straight.t == 10
    assert_identical(straight, resumed)
    for x, y in zip(straight.sample(), resumed.sample()):  # and the next sampled deals
        assert np.array_equal(x, y)


@pytest.mark.parametrize("name", NAMES)
def test_evaluation_from_a_checkpoint(setup, name, tmp_path):
    # The loaded side also uses buckets that went through save and load, as M1's jobs do.
    upper, river, abstractions = setup
    solver = run(make(setup, name), 6)
    solver.save(tmp_path / "state.npz")
    abstraction = abstractions[name]
    if name != "lossless":
        abstraction.save(tmp_path / "buckets.npz")
        abstraction = EquityBuckets.load(tmp_path / "buckets.npz")
    loaded = make(setup, name, abstraction=abstraction)
    loaded.load(tmp_path / "state.npz")
    want = exact_exploitability(upper, river, abstractions[name], solver.average_strategy())
    got = exact_exploitability(upper, river, abstraction, loaded.average_strategy())
    assert got == want  # exactly


@pytest.mark.parametrize("method", ["avg_1d", "kmeans_2d"])
def test_buckets_round_trip(setup, method, tmp_path):
    upper, river, abstractions = setup
    a = abstractions[method]
    a.save(tmp_path / "buckets.npz")
    b = EquityBuckets.load(tmp_path / "buckets.npz")
    assert (b.method, b.k, b.num_rows, b.num_ids) == (a.method, a.k, a.num_rows, a.num_ids)
    assert np.array_equal(bits(b.centers), bits(a.centers))
    assert b.fit == a.fit == {"samples": 12, "seed": 0, "restarts": 3}
    rng = np.random.default_rng(4)
    parents = rng.integers(0, upper.streets[-1].num_deals, size=40)
    children = rng.integers(0, river.num_children, size=40)
    st = river.street(parents, children)
    for p in (0, 1):
        assert np.array_equal(a.ids(st, parents, children, p), b.ids(st, parents, children, p))


def test_load_refuses_another_shape(setup, tmp_path):
    run(make(setup, "lossless"), 2).save(tmp_path / "state.npz")
    other = run(make(setup, "kmeans_2d"), 1)
    before = [[s.copy() for s in row] for row in other.sigma]
    with pytest.raises(ValueError, match="in the checkpoint but"):
        other.load(tmp_path / "state.npz")
    assert other.t == 1 and other.batch == 8
    for row_a, row_b in zip(before, other.sigma):
        for x, y in zip(row_a, row_b):
            assert np.array_equal(bits(x), bits(y))


def test_evaluation_progress(setup):
    upper, river, abstractions = setup
    sigma = run(make(setup, "lossless"), 3).average_strategy()
    calls = []
    got = exact_exploitability(upper, river, abstractions["lossless"], sigma, chunk=500,
                               progress=lambda done, total: calls.append((done, total)), progress_every=2)
    assert got == exact_exploitability(upper, river, abstractions["lossless"], sigma, chunk=500)
    n = -(-upper.streets[-1].num_deals * river.num_children // 500)
    assert calls == [(d, n) for d in range(2, n, 2)] + [(n, n)]
