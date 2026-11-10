"""CFR-D bookkeeping in experiments/valuenet/gate_c1.py (CFRDSolver), small limit deck: with a
leaf that plays a fixed river strategy through ExactRiverLeaf's plumbing (no re-solve), the
CFR-D average river must equal that strategy, and grading the composite must equal grading the
trunk average with the fixed river directly."""
from types import SimpleNamespace

import numpy as np
import pytest

from bombpot.core.solvers.range_cfr import RangeCFR, _segment_sum, class_reach, class_values
from bombpot.core.solvers.subgame import ExactRiverLeaf
from bombpot.experiments.valuenet.gate_c1 import CFRDSolver, make_game
from bombpot.holdem.public import HoldemRiver, holdem_public_tree


class FixedRiverLeaf(ExactRiverLeaf):
    """ExactRiverLeaf's plumbing (roots, memo, scatter) with a fixed river strategy instead of a
    re-solve: values are that strategy's, from the closing ranges."""

    def __init__(self, pt, last, fixed):
        super().__init__(pt, last, tol=None)
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
