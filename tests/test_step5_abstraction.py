"""Step 5: equity features, bucketing and the abstract game."""
from collections import defaultdict

import numpy as np
import pytest

from toygames.abstraction.abstract_game import build_abstract_game
from toygames.abstraction.bucketing import emd, kmeans, make_bucketing
from toygames.abstraction.features import compute_features
from toygames.eval.exploitability import exploitability
from toygames.games import BoardLeduc
from toygames.solvers.cfr import CFR
from toygames.tree import compile_game, profile_value

N = 4
MODES = ["shared", "independent"]
_cache = {}


def setup(mode, streets=2):
    key = (mode, streets)
    if key not in _cache:
        g = BoardLeduc(num_ranks=N, num_boards=2, streets=streets, deck_mode=mode)
        _cache[key] = (g, compile_game(g), compute_features(g))
    return _cache[key]


# Features -------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES + ["identical"])
def test_features_well_formed(mode):
    g, _, feats = setup(mode)
    for t in range(2):
        for (p, board), hf in feats.streets[t].items():
            assert len(board) == feats.board_len[t]
            assert hf.weights.sum() == pytest.approx(1.0)
            assert np.all((hf.points >= -1e-12) & (hf.points <= 1 + 1e-12))
            if mode == "identical":
                np.testing.assert_allclose(hf.points[:, 0], hf.points[:, 1])
            if t == 0:
                for a, w, mean in zip(hf.atoms, hf.atom_weights, hf.points):
                    assert w.sum() == pytest.approx(1.0)
                    np.testing.assert_allclose(w @ a, mean, atol=1e-12)
            # Symmetric deal: both players see the same features.
            other = feats.streets[t][(1 - p, board)]
            assert other.hands == hf.hands
            np.testing.assert_allclose(other.points, hf.points, atol=1e-12)


def test_shared_range_is_uniform_over_remaining_cards():
    g, _, feats = setup("shared")
    # Recompute last-street equity from scratch: opponent uniform over the 2N - 5 unseen cards.
    for (p, board), hf in feats.streets[1].items():
        for x, point in zip(hf.hands, hf.points):
            used = (x,) + board
            counts = np.array([2 - used.count(r) for r in range(N)], dtype=float)
            eq = sum(c * (g.showdown(board, x, y) if p == 0 else 1 - g.showdown(board, y, x))
                     for y, c in enumerate(counts) if c > 0) / counts.sum()
            np.testing.assert_allclose(point, eq, atol=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_histogram_mean_is_street1_equity(mode):
    # Law of total expectation: the mean over next deals is the street-1 equity vs a uniform range.
    g, _, feats = setup(mode)
    num = defaultdict(lambda: np.zeros(2))
    den = defaultdict(float)
    for board, h0, h1, q in g.deals(1):
        share0 = g.showdown(board, h0, h1)
        for p, x, share in ((0, h0, share0), (1, h1, 1 - share0)):
            num[(p, board[:2], x)] += q * share
            den[(p, board[:2], x)] += q
    for (p, board), hf in feats.streets[0].items():
        for x, mean in zip(hf.hands, hf.points):
            np.testing.assert_allclose(mean, num[(p, board, x)] / den[(p, board, x)], atol=1e-12)


# Bucketing ------------------------------------------------------------------------


def test_kmeans_recovers_separated_clusters():
    rng = np.random.default_rng(0)
    X = np.array([[0.0, 0.0], [0.01, 0.0], [1.0, 1.0], [1.0, 0.99], [0.0, 1.0]])
    labels = kmeans(X, np.ones(5) / 5, 3, rng)
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert len({labels[0], labels[2], labels[4]}) == 3
    # At most k distinct points: one bucket each, no randomness needed.
    assert len(set(kmeans(X[:3], np.ones(3) / 3, 3, rng))) == 3


def _sse(X, w, labels):
    return sum((w[labels == c, None] * (X[labels == c] - np.average(X[labels == c], axis=0,
                                                                   weights=w[labels == c])) ** 2).sum()
               for c in np.unique(labels))


def _optimal_sse(X, w, k):
    """Exact k-means optimum by trying every partition (fine for a handful of points)."""
    import itertools

    best = np.inf
    for labels in itertools.product(range(k), repeat=len(X)):
        if labels[0] == 0 and len(set(labels)) == k:
            best = min(best, _sse(X, w, np.array(labels)))
    return best


def test_restarts_escape_bad_local_optimum():
    # A real street-1 public state (N = 5, independent). One k-means++ start can split
    # off the outlier (SSE 0.0814); the optimum pairs the three weak hands (SSE 0.0558).
    X = np.array([[0.1224, 0.8571], [0.2755, 0.1939], [0.4796, 0.3980], [0.9592, 0.5510], [0.7041, 0.7041]])
    w = np.array([0.125, 0.25, 0.25, 0.125, 0.25])
    single = [_sse(X, w, kmeans(X, w, 2, np.random.default_rng(s))) for s in range(20)]
    assert max(single) > _optimal_sse(X, w, 2) + 1e-3  # some single starts stall
    for s in range(20):
        assert _sse(X, w, kmeans(X, w, 2, np.random.default_rng(s), restarts=20)) == \
               pytest.approx(_optimal_sse(X, w, 2), abs=1e-12)


@pytest.mark.parametrize("k", [2, 3])
def test_restarts_reach_the_optimum_almost_always(k):
    _, _, feats = setup("independent")
    hit = total = 0
    for t in range(2):
        for key, hf in list(feats.streets[t].items())[::2]:
            if len(np.unique(np.round(hf.points, 12), axis=0)) <= k:
                continue
            labels = kmeans(hf.points, hf.weights, k, np.random.default_rng(0), restarts=20)
            total += 1
            hit += _sse(hf.points, hf.weights, labels) <= _optimal_sse(hf.points, hf.weights, k) + 1e-12
    assert hit / total >= 0.97


def test_emd_basics():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert emd(a, np.array([0.5, 0.5]), a, np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert emd(a[:1], np.ones(1), np.array([[3.0, 4.0]]), np.ones(1)) == pytest.approx(5.0)
    # Same mean, different spread: EMD sees it, the mean point does not.
    spread = np.array([[0.0, 0.5], [1.0, 0.5]])
    assert emd(spread, np.array([0.5, 0.5]), np.array([[0.5, 0.5]]), np.ones(1)) == pytest.approx(0.5)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method, k", [("avg_1d", 2), ("avg_1d", 3), ("product", 4), ("kmeans_2d", 2),
                                       ("kmeans_2d", 5), ("emd_2d", 3), ("lossless", None)])
def test_bucket_counts(mode, method, k):
    _, _, feats = setup(mode)
    b = make_bucketing(feats, method, k, seed=0)
    for t in range(2):
        for key, labels in b.table[t].items():
            n_hands = len(feats.streets[t][key].hands)
            n_buckets = len(set(labels.values()))
            if method == "lossless":
                assert n_buckets == n_hands
            else:
                assert n_buckets <= min(k, n_hands)
    assert make_bucketing(feats, method, k, seed=0).table == b.table  # seeded


@pytest.mark.parametrize("method, k", [("avg_1d", 2), ("product", 4), ("kmeans_2d", 3), ("emd_2d", 2)])
def test_one_bucketing_per_public_state(method, k):
    # Both players see the same features here, so they must get the same buckets,
    # and the buckets of a public state don't depend on which other states exist.
    _, _, feats = setup("shared")
    b = make_bucketing(feats, method, k, seed=3)
    for t in range(2):
        for (p, board), labels in b.table[t].items():
            if p == 0:
                assert labels == b.table[t][(1, board)]
    board = next(iter(feats.streets[0]))[1]
    single = type(feats)(feats.num_streets, feats.num_boards, feats.board_len,
                         [{key: hf for key, hf in feats.streets[0].items() if key[1] == board}, {}])
    alone = make_bucketing(single, method, k, seed=3)
    assert alone.table[0][(0, board)] == b.table[0][(0, board)]


def test_product_needs_square_k():
    _, _, feats = setup("shared")
    with pytest.raises(ValueError):
        make_bucketing(feats, "product", 3, seed=0)


# Abstract game ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_lossless_matches_full_game_exactly(mode):
    _, full, feats = setup(mode)
    ag = build_abstract_game(full, make_bucketing(feats, "lossless", None, seed=0))
    t = ag.tree
    assert t.n_infosets == full.n_infosets and t.n_terminals == full.n_terminals
    for p in (0, 1):
        np.testing.assert_array_equal(ag.seq_map[p], np.arange(full.n_seqs[p]))
        np.testing.assert_array_equal(t.term_seq[p], full.term_seq[p])
    np.testing.assert_array_equal(t.term_cu0, full.term_cu0)
    a, b = CFR(full), CFR(t)
    ra = a.run(iterations=200, log_every=50)
    rb = b.run(iterations=200, log_every=50)
    assert [{k: v for k, v in r.items() if k != "wall_time"} for r in ra] == \
           [{k: v for k, v in r.items() if k != "wall_time"} for r in rb]
    lifted = ag.lift(b.average_strategy())
    for x, y in zip(lifted, a.average_strategy()):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method, k", [("avg_1d", 2), ("product", 4), ("kmeans_2d", 3), ("emd_2d", 2)])
def test_abstract_game_is_sound(mode, method, k):
    _, full, feats = setup(mode)
    ag = build_abstract_game(full, make_bucketing(feats, method, k, seed=1))
    assert sum(ag.tree.n_infosets) < sum(full.n_infosets)
    solver = CFR(ag.tree)
    solver.run(iterations=300, log_every=300)
    sigma = solver.average_strategy()
    inside = exploitability(ag.tree, sigma)
    lifted = ag.lift(sigma)
    total = exploitability(full, lifted)
    # Payoffs are untouched, so the profile is worth the same in both games...
    assert profile_value(full, lifted) == pytest.approx(inside.game_value, abs=1e-12)
    # ...and a full-game best response sees more, so it can only do better.
    assert total.br_value[0] >= inside.br_value[0] - 1e-12
    assert total.br_value[1] >= inside.br_value[1] - 1e-12
    assert total.exploitability >= inside.exploitability - 1e-12
    # Every full infoset in a bucket plays the bucket's strategy.
    for p in (0, 1):
        for i in range(full.n_infosets[p]):
            f, j = full.first_seq[p][i], ag.infoset_map[p][i]
            g0 = ag.tree.first_seq[p][j]
            n = full.num_actions[p][i]
            np.testing.assert_array_equal(lifted[p][f:f + n], sigma[p][g0:g0 + n])
