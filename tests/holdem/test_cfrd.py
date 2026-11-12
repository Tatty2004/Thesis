"""CFR-D bookkeeping in experiments/valuenet/gate_c1.py (CFRDSolver), small limit deck: with a
leaf that plays a fixed river strategy through ExactRiverLeaf's plumbing (no re-solve), the
CFR-D average river must equal that strategy, and grading the composite must equal grading the
trunk average with the fixed river directly."""
from types import SimpleNamespace

import numpy as np
import pytest

from bombpot.core.solvers.range_cfr import RangeCFR, _segment_sum, class_reach, class_values, forward
from bombpot.core.solvers.subgame import ExactRiverLeaf
from bombpot.experiments.valuenet.gate_c1 import CFRDSolver, deepstack_grade, make_game
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree


class FixedRiverLeaf(ExactRiverLeaf):
    """ExactRiverLeaf's plumbing (roots, memo, scatter) with a fixed river strategy instead of a
    re-solve: values are that strategy's, from the closing ranges."""

    def __init__(self, pt, last, fixed):
        super().__init__(pt, last, tol=None, values="average")
        self.fixed = fixed

    def solve(self, closing):
        if self._memo is not None and all(np.array_equal(a, b) for x, y in zip(self._memo[0], closing)
                                          for a, b in zip(x, y)):
            return self._memo[1]
        D1 = len(closing[0][0])
        subs = []
        for lc, roots, strat in zip(self.classes, self.roots(closing), self.fixed):
            root = tuple(r[self.group] for r in roots)
            nodes, _ = class_reach(lc.template, strat, root)
            values = [_segment_sum(class_values(lc.template, self.street, lc.contrib, strat, nodes, p, "cfr",
                                                True)[0][0], self.group, D1) for p in (0, 1)]
            subs.append(SimpleNamespace(strategy=strat, values=values))
        self.stats.append({})
        self._memo = (tuple(tuple(np.array(r) for r in c) for c in closing), subs)
        return subs


@pytest.mark.parametrize("simultaneous", [True, False])
@pytest.mark.parametrize("variant", ["cfr", "dcfr"])
def test_cfrd_average_of_a_fixed_river_is_that_river(variant, simultaneous):
    game = make_game("limit")
    full, upper = holdem_public_tree(game), holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
    rng = np.random.default_rng(0)
    fixed = []
    for c, lc in enumerate(full.classes[2]):
        row = []
        for a in lc.template.actions:
            x = rng.random(full.shape(2, c) + (len(a),)) + 0.05
            row.append(x / x.sum(-1, keepdims=True))
        fixed.append(row)
    leaf = FixedRiverLeaf(upper, river, fixed)
    s = CFRDSolver(upper, leaf, variant, simultaneous=simultaneous)
    for _ in range(6):
        s.iteration()
    for got_c, want_c in zip(s.river_average(), fixed):
        for got, want in zip(got_c, want_c):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)
    avg = s.hand_sigma(s.average_strategy())
    composite = RangeCFR(full).exploitability([avg[0], avg[1], s.river_average()])
    direct = RangeCFR(full).exploitability([avg[0], avg[1], fixed])
    for key in direct:
        assert composite[key] == pytest.approx(direct[key], rel=1e-12, abs=1e-15)
    # and the trunk-only value against this leaf is the same game: flop+turn exploitability
    # with the fixed river's values = the full tree's exploitability with the river fixed for
    # the opponent but best-responded by the deviator is >= ...; at least the values agree:
    ft = s.exploitability()
    assert ft["game_value"] == pytest.approx(direct["game_value"], rel=1e-12)


# CFR-D with re-solved rivers (the river strategy changes every iteration) --------------------

class FullLast:
    """The full tree's own last street as ExactRiverLeaf's last-street provider (HoldemRiver
    needs two boards; the game below has one, to be small)."""

    def __init__(self, full):
        self.st = full.streets[-1]

    def all_children(self, parents):
        return None, None

    def street(self, *a):
        return self.st


def one_board(max_raises):
    game = Holdem(num_ranks=6, num_suits=2, num_boards=1, board_cards=(3, 1, 1), flops=("2c3c4d",),
                  bet_sizes=(2, 4, 4), max_raises=max_raises)
    return holdem_public_tree(game), holdem_public_tree(game, upto=2)


def test_cfrd_average_is_the_average_realization_plan():
    # The composite (trunk average, CFR-D river average) must reach every river node with the
    # weighted average of the iterations' reaches (own reach is a product of own action
    # probabilities, i.e. the realization plan): computed here by the full tree's own forward
    # pass on each iteration's profile (trunk strategy, that iteration's river solves).
    full, upper = one_board(1)
    leaf = ExactRiverLeaf(upper, FullLast(full), 1e-2, variant="dcfr", iters=200)
    s = CFRDSolver(upper, leaf, "dcfr")
    want, total = None, 0.0
    for _ in range(4):
        trunk = [[[x.copy() for x in cls] for cls in row] for row in s.hand_sigma()]
        s.iteration()
        w = s.average_weight()
        reach = forward(full, [trunk[0], trunk[1], [sub.strategy for sub in leaf._memo[1]]])[2]
        now = [[[w * r[q] for q in (0, 1)] for r in cls] for cls in reach]
        want = now if want is None else [[[a + b for a, b in zip(x, y)] for x, y in zip(cx, cy)]
                                         for cx, cy in zip(want, now)]
        total += w
    avg = s.hand_sigma(s.average_strategy())
    got = forward(full, [avg[0], avg[1], s.river_average()])[2]
    for c, lc in enumerate(full.classes[2]):
        for k, q in enumerate(lc.template.players):
            np.testing.assert_allclose(got[c][k][q], want[c][k][q] / total, rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("variant, iters, factor", [("dcfr", 20, 1.5), pytest.param("cfr", 20, 1.2, marks=pytest.mark.slow)])
def test_cfrd_with_resolved_rivers_converges_like_the_full_width_solve(variant, iters, factor):
    # Gate C1's failure in small: with river subgames re-solved every iteration, CFR-D's
    # composite and the safely re-solved (DeepStack-style) composite must come down like the
    # full-width solve of the same variant. With root values that were uniform play's for the
    # hands a player didn't bring to a river (subgame.py), the composite stayed at 0.61 (dcfr)
    # and 1.01 (cfr) here after 20 iterations and never came down.
    full, upper = one_board(2)
    tol = 1e-2
    leaf = ExactRiverLeaf(upper, FullLast(full), tol, variant="dcfr", iters=2000)
    s = CFRDSolver(upper, leaf, variant)
    ref = RangeCFR(full, variant)
    for _ in range(iters):
        s.iteration()
        ref.iteration()
    want = ref.exploitability()["exploitability"]
    avg = s.hand_sigma(s.average_strategy())
    cfrd = RangeCFR(full).exploitability([avg[0], avg[1], s.river_average()])["exploitability"]
    deep = deepstack_grade(full, upper, s, tol, 2000, "dcfr")["exploitability"]
    print(f"{variant} it {iters}: full-width {want:.4f} cfr-d {cfrd:.4f} deepstack {deep:.4f}")
    assert cfrd <= factor * want + tol
    assert deep <= factor * want + 2 * tol
