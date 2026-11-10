"""B: the range solver on the no-limit game, whose streets have several line classes.

With pot-fraction sizes and stacks (core/betting.py) the chips differ per line, and an
all-in truncates the betting of later streets: core/public.py groups a street's lines into
classes of the same betting shape (pt.classes[t]), with a zero-node class for lines that are
all-in. Here the range solver, on both the dense public tree (core/public.py) and the
card-removal one (holdem/public.py), must be the tree solver on the same game:
  - in lockstep along the tree solver's trajectory: the same instant regrets and averaging
    weights (own reach) to 1e-12, full and bucketed (against the abstract tree);
  - the same best responses and value of any profile (1e-12);
  - DCFR converges to the sequence-form LP value.
Games: two fixed flops on a 4 x 3 deck, flop and turn, with 100 behind (one class per street,
but 13 turn lines with 13 different pots), 6 behind (five turn classes, one all-in), and 2
behind (half the turn lines all-in); and, slow, a three-street game.
"""
import numpy as np
import pytest

from bombpot.core.abstraction.abstract_game import build_abstract_game, range_buckets
from bombpot.core.abstraction.bucketing import make_bucketing
from bombpot.core.abstraction.features import compute_features
from bombpot.core.eval.exploitability import exploitability
from bombpot.core.public import build_public_tree, from_tree_sigma, to_tree_sigma, tree_coordinates
from bombpot.core.solvers.cfr import CFR
from bombpot.core.solvers.lp import solve_game
from bombpot.core.solvers.range_cfr import RangeCFR, forward
from bombpot.core.tree import compile_game, realization_plan
from bombpot.experiments.certify import random_profile
from bombpot.holdem import Holdem
from bombpot.holdem.public import holdem_public_tree

FLOPS = dict(num_ranks=4, num_suits=3, board_cards=(3, 1), flops=("2c3c4d", "5c2d3d"), betting="pot")
GAMES = {
    "deep": dict(stack=100, **FLOPS),
    "stack6": dict(stack=6, **FLOPS),
    "stack2": dict(stack=2, **FLOPS),
}
_built = {}


def built(name, kind="fast"):
    key = (name, kind)
    if key not in _built:
        g = Holdem(**GAMES[name])
        tree = _built[(name, "tree")] if (name, "tree") in _built else compile_game(g)
        _built[(name, "tree")] = tree
        _built[key] = (g, tree, holdem_public_tree(g) if kind == "fast" else build_public_tree(g))
    return _built[key]


def test_class_structure():
    shapes = {name: [[(lc.num_lines, lc.template.num_nodes, lc.template.num_closes) for lc in row]
                     for row in built(name)[2].classes] for name in GAMES}
    assert shapes["deep"] == [[(1, 14, 13)], [(13, 14, 13)]]
    assert shapes["stack6"] == [[(1, 14, 13)], [(1, 14, 13), (2, 10, 9), (2, 8, 7), (6, 4, 3), (2, 0, 1)]]
    assert shapes["stack2"] == [[(1, 8, 7)], [(1, 8, 7), (2, 4, 3), (4, 0, 1)]]
    for name in GAMES:
        g, _, pt = built(name)
        for t, row in enumerate(pt.classes):
            lines = np.sort(np.concatenate([lc.lines for lc in row]))
            assert np.array_equal(lines, np.arange(pt.num_lines(t)))
            for lc in row:
                np.testing.assert_array_equal(lc.contrib + lc.stack, g.total)
                if t:  # every (parent line, close) continues into exactly one line
                    prev = pt.classes[t - 1]
                    np.testing.assert_array_equal(
                        lc.contrib, [prev[c].contrib[l] + prev[c].template.closes[j][1][l]
                                     for c, l, j in zip(lc.parent_class, lc.parent_line, lc.parent_close)])
        pairs = {(c, l, j) for lc in pt.classes[1] for c, l, j in zip(lc.parent_class, lc.parent_line,
                                                                      lc.parent_close)}
        assert len(pairs) == pt.num_lines(1) == sum(lc.template.num_closes * lc.num_lines for lc in pt.classes[0])
    # The all-in class has no betting and stacks of zero; the deep game's turn pots differ by line.
    allin = built("stack6")[2].classes[1][-1]
    assert allin.template.num_nodes == 0 and (allin.stack == 0).all()
    np.testing.assert_array_equal(built("deep")[2].classes[1][0].contrib, [1, 2, 3, 2, 3, 4, 6, 6, 9, 4, 6, 6, 9])


@pytest.mark.parametrize("kind", ["fast", "dense"])
@pytest.mark.parametrize("name", GAMES)
def test_regrets_and_reach_match_in_lockstep(name, kind):
    _, tree, pt = built(name, kind)
    a, b = CFR(tree), RangeCFR(pt)
    co = [tree_coordinates(pt, tree, p) for p in (0, 1)]
    for _ in range(25):
        for p in (0, 1):
            b.sigma = from_tree_sigma(pt, tree, a.sigma)
            reach, regrets = b.instant_regrets(p)
            ra = a._instant_regret(p)
            xa = realization_plan(tree, p, a.sigma[p])
            own = {}
            for n, (t, d, c, l, k, i) in enumerate(co[p]):
                f, m = tree.first_seq[p][n], tree.num_actions[p][n]
                assert np.abs(ra[f:f + m] - regrets[t][c][k][d, l, i]).max() < 1e-12
                if (t, c, k) not in own:
                    own[(t, c, k)] = b.own_reach(t, c, k, reach)
                assert np.abs(xa[f:f + m] - own[(t, c, k)][d, l, i] * b.sigma[t][c][k][d, l, i]).max() < 1e-12
        a.iteration()


@pytest.mark.parametrize("name", GAMES)
def test_best_response_and_value_match(name):
    _, tree, pt = built(name)
    solver = CFR(tree)
    solver.run(iterations=50, log_every=50)
    b = RangeCFR(pt)
    for sigma in [solver.average_strategy(), random_profile(tree, 3), tree.uniform()]:
        ours = b.exploitability(from_tree_sigma(pt, tree, sigma))
        ref = exploitability(tree, sigma)
        assert ours["br_value_p0"] == pytest.approx(ref.br_value[0], abs=1e-12)
        assert ours["br_value_p1"] == pytest.approx(ref.br_value[1], abs=1e-12)
        assert ours["game_value"] == pytest.approx(ref.game_value, abs=1e-12)


@pytest.mark.parametrize("name", GAMES)
def test_dcfr_reaches_the_lp_value(name):
    _, tree, pt = built(name)
    (v, v1), sigma_lp = solve_game(tree)
    assert v == pytest.approx(v1, abs=1e-6)
    b = RangeCFR(pt)
    lp = b.exploitability(from_tree_sigma(pt, tree, sigma_lp))
    assert lp["exploitability"] < 1e-7 and lp["game_value"] == pytest.approx(v, abs=1e-8)
    for _ in range(2000):
        b.iteration()
    rep = b.exploitability()
    assert rep["exploitability"] < 1e-4
    assert -rep["br_value_p1"] - 1e-9 <= v <= rep["br_value_p0"] + 1e-9
    ref = exploitability(tree, to_tree_sigma(pt, tree, b.hand_sigma(b.average_strategy())))
    assert rep["game_value"] == pytest.approx(ref.game_value, abs=1e-12)


BUCKETED = [("stack6", "kmeans_2d", 3), ("stack2", "avg_1d", 2), ("deep", "avg_1d", 3)]


@pytest.mark.parametrize("name, method, k", BUCKETED, ids=[f"{n}-{m}-k{k}" for n, m, k in BUCKETED])
def test_bucketed_lockstep(name, method, k):
    game, full, pt = built(name)
    bucketing = make_bucketing(compute_features(game), method, k, seed=0, restarts=5)
    ag = build_abstract_game(full, bucketing)
    ids = range_buckets(pt, bucketing)
    a, b = CFR(ag.tree), RangeCFR(pt, buckets=ids)
    where = []
    for p in (0, 1):
        co = tree_coordinates(pt, full, p)
        c = np.full((ag.tree.n_infosets[p], 6), -1, dtype=np.int64)
        flat = np.column_stack([co[:, :5], [ids[t][p][d, i] for t, d, _, _, _, i in co]])
        c[ag.infoset_map[p]] = flat
        assert (c[ag.infoset_map[p]] == flat).all() and (c >= 0).all()
        where.append(c)
    assert len(ag.tree.keys[0]) < full.n_infosets[0]  # the bucketing merges something
    for _ in range(15):
        for p in (0, 1):
            for q in (0, 1):
                for n, (t, d, cc, l, kk, i) in enumerate(where[q]):
                    f, m = ag.tree.first_seq[q][n], ag.tree.num_actions[q][n]
                    b.sigma[t][cc][kk][d, l, i] = a.sigma[q][f:f + m]
            reach, regrets = b.instant_regrets(p)
            ra = a._instant_regret(p)
            xa = realization_plan(ag.tree, p, a.sigma[p])
            own = {}
            for n, (t, d, cc, l, kk, i) in enumerate(where[p]):
                f, m = ag.tree.first_seq[p][n], ag.tree.num_actions[p][n]
                assert np.abs(ra[f:f + m] - regrets[t][cc][kk][d, l, i]).max() < 1e-12
                if (t, cc, kk) not in own:
                    own[(t, cc, kk)] = b.own_reach(t, cc, kk, reach)
                assert np.abs(xa[f:f + m] - own[(t, cc, kk)][d, l, i] * b.sigma[t][cc][kk][d, l, i]).max() < 1e-12
        a.iteration()


def test_last_street_closing_reach():
    # forward(..., closing=True) gives the last street's reach after every close, per class:
    # (D, L_c, closes_c, H), the reach a leaf value provider at the end of that street takes.
    _, tree, pt = built("stack6")
    b = RangeCFR(pt)
    hsig = b.hand_sigma(from_tree_sigma(pt, tree, random_profile(tree, 4)))
    reach, closing = forward(pt, hsig, closing=True)
    assert len(closing) == len(pt.classes[-1])
    for c, lc in enumerate(pt.classes[-1]):
        tp = lc.template
        for p in (0, 1):
            assert closing[c][p].shape == pt.shape(1, c)[:2] + (tp.num_closes, len(pt.hands))
        if tp.num_nodes == 0:  # all-in: the close is the root
            np.testing.assert_array_equal(closing[c][0][:, :, 0], reach[1][c][0][0])
            continue
        for k, row in enumerate(tp.children):
            for a, child in enumerate(row):
                if child[0] == "close":
                    q = tp.players[k]
                    want = reach[1][c][k][q] * hsig[1][c][k][..., a]
                    np.testing.assert_array_equal(closing[c][q][:, :, child[1]], want)
                    np.testing.assert_array_equal(closing[c][1 - q][:, :, child[1]], reach[1][c][k][1 - q])


THREE = dict(num_ranks=4, num_suits=2, board_cards=(2, 1, 1), num_boards=1, flops=("2c3c",), betting="pot", stack=6)


@pytest.mark.slow
def test_three_streets_match_the_tree_solver():
    game = Holdem(**THREE)
    tree, pt = compile_game(game), holdem_public_tree(game)
    assert len(pt.classes[2]) > 2 and any(lc.template.num_nodes == 0 for lc in pt.classes[2])
    a, b = CFR(tree), RangeCFR(pt)
    co = [tree_coordinates(pt, tree, p) for p in (0, 1)]
    for _ in range(4):
        for p in (0, 1):
            b.sigma = from_tree_sigma(pt, tree, a.sigma)
            _, regrets = b.instant_regrets(p)
            ra = a._instant_regret(p)
            f, m = tree.first_seq[p], tree.num_actions[p]
            for n, (t, d, c, l, k, i) in enumerate(co[p]):
                assert np.abs(ra[f[n]:f[n] + m[n]] - regrets[t][c][k][d, l, i]).max() < 1e-12
        a.iteration()
    for sigma in [a.average_strategy(), random_profile(tree, 5)]:
        ours = RangeCFR(pt).exploitability(from_tree_sigma(pt, tree, sigma))
        ref = exploitability(tree, sigma)
        assert ours["br_value_p0"] == pytest.approx(ref.br_value[0], abs=1e-12)
        assert ours["br_value_p1"] == pytest.approx(ref.br_value[1], abs=1e-12)
        assert ours["game_value"] == pytest.approx(ref.game_value, abs=1e-12)
    (v, _), _ = solve_game(tree)
    b = RangeCFR(pt)
    for _ in range(1500):
        b.iteration()
    rep = b.exploitability()
    assert rep["exploitability"] < 1e-3
    assert -rep["br_value_p1"] - 1e-9 <= v <= rep["br_value_p0"] + 1e-9


@pytest.fixture(scope="module")
def real():
    game = Holdem(flops=("Ah7c2d", "KsKd9h"), board_cards=(3, 1, 1), betting="pot")
    return game, holdem_public_tree(game, upto=2)


def test_real_pot_game_builds(real):
    game, pt = real
    assert [[(lc.num_lines, lc.template.num_nodes) for lc in row] for row in pt.classes[:2]] == [[(1, 14)], [(13, 14)]]
    # The river's classes come with the tree (pt.classes covers every street): 169 lines.
    assert sum(lc.num_lines for lc in pt.classes[2]) == 169 and len(pt.classes[2]) == 6
    assert pt.num_streets == 2 and len(pt.streets[1].deals) == 46 * 45
