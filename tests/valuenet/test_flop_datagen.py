"""Flop-end samples (experiments/valuenet/flop_datagen.py) on the small deck (5 ranks x 3 suits,
the pot game) with a small random HandSetNet: the turn subgame's units against the flop+turn
trunk, a subset of turn deals solving exactly as those deals do in the full solve (so the
sampled labels are estimators of the full one), and NetLeaf.values_both."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bombpot.core.solvers.range_cfr import backward, forward  # noqa: E402
from bombpot.experiments.valuenet import datagen as dg  # noqa: E402
from bombpot.experiments.valuenet import flop_datagen as fd  # noqa: E402
from bombpot.experiments.valuenet.gate_f import LeafCFR, turn_pots  # noqa: E402
from bombpot.valuenet.features import FeatureConfig  # noqa: E402
from bombpot.valuenet.leaf import NetLeaf  # noqa: E402
from bombpot.valuenet.model import ModelConfig, build_model  # noqa: E402

FLOPS = "2c3c4d,5c6d4c"


@pytest.fixture(scope="module")
def setup():
    fe = fd.FlopEnd(fd.parse_flops(FLOPS, 5, 3), 5, 3)
    torch.manual_seed(0)
    model = build_model(FeatureConfig("hand_set", fe.H, num_ranks=5, num_suits=3),
                        ModelConfig(kind="set", depth=1, width=16, heads=2)).double().eval()
    state = fd.sample_state(fe, np.random.default_rng(3), line=4)  # a line with a bet and a raise behind it
    return fe, model, state


def test_shapes_and_weights(setup):
    fe, _, state = setup
    assert (fe.D, fe.H, fe.per_pair) == (72, 36, 20)
    assert np.allclose(fe.turn.kappa, 1 / 20)
    # a hand pair's valid turn deals weigh 1 in all
    hs = fe.hands
    i, j = 0, int(np.flatnonzero([not set(hs[0]) & set(h) for h in hs])[0])
    both = fe.turn.valid[:, i] & fe.turn.valid[:, j]
    blocked = np.array([bool(set(hs[i]) & set(b[:, 3])) or bool(set(hs[j]) & set(b[:, 3])) for b in fe.boards])
    assert both.sum() == fe.per_pair and np.array_equal(both, ~blocked)
    assert state["pot"] == 2 * fe.lc.contrib[4] and np.isclose(state["range0"].sum(), 1)


def test_units_against_the_trunk(setup):
    """The turn subgame below flop-end line l, entered with the trunk's normalized reach there
    and playing the trunk's turn strategy, has the trunk's turn-root values divided by the
    opponent's flop-end reach mass and the hole-card chance weight."""
    fe, model, _ = setup
    up = fe.upper
    tree_turn = up.streets[1]
    trunk_leaf = NetLeaf(model, fe.equity(np.arange(fe.D)), turn_pots(up), [dg.STACK - p / 2 for p in turn_pots(up)],
                         tree_turn.kappa, "cpu", boards_by_deal=fe.boards, possible_by_deal=tree_turn.valid,
                         hands=fe.hands)
    trunk = LeafCFR(up, trunk_leaf, "dcfr")
    rng = np.random.default_rng(2)  # a random profile, every line reached
    for row in trunk.sigma:
        for cls in row:
            for k, x in enumerate(cls):
                y = rng.random(x.shape) + 0.05
                cls[k] = y / y.sum(-1, keepdims=True)
    hsig = trunk.hand_sigma()
    reach = forward(up, hsig)
    hole = 1.0 / (tree_turn.kappa[0] * fe.per_pair)
    for line in (0, 4, 12):
        R = [reach[1][0][0][p][0, line] for p in (0, 1)]  # turn-root reach (the same at every deal)
        state = {"line": line, "pot": 2 * float(fe.lc.contrib[line]), "range0": R[0] / R[0].sum(),
                 "range1": R[1] / R[1].sum()}
        deals = np.arange(fe.D)
        s = fd.TurnCFR(fe, state, deals, fe.leaf(model, deals, line, device="cpu"))
        for k in range(len(s.sigma[0][0])):
            s.sigma[0][0][k] = np.ascontiguousarray(trunk.sigma[1][0][k][:, line:line + 1])
        sh = s.hand_sigma()
        sr = s.reach(sh)
        for p in (0, 1):
            mine = s.values(p, sh, sr)[0][0][0][0][:, 0]
            theirs = backward(up, hsig, reach, p, "cfr", leaf=trunk.leaf)[0][1][0][0][:, line]
            assert np.allclose(mine * R[1 - p].sum() / hole, theirs, rtol=1e-9, atol=1e-15), (line, p)


@pytest.mark.parametrize("simultaneous", [False, True])
def test_deal_subset_solves_as_in_the_full_solve(setup, simultaneous):
    fe, model, state = setup
    full = fd.solve_state(fe, model, state, np.arange(fe.D), 12, [6, 12], device="cpu", simultaneous=simultaneous,
                          log=lambda m: None)
    S = np.sort(np.random.default_rng(1).choice(fe.D, 17, replace=False))
    part = fd.solve_state(fe, model, state, S, 12, [6, 12], device="cpu", simultaneous=simultaneous,
                          log=lambda m: None)
    for key in ("deal_cfv", "deal_br"):
        for c in range(2):
            assert np.allclose(part[key][c], full[key][c][:, S], rtol=1e-9, atol=1e-12)
    # the label with every deal is the mean of the per-deal arrays (each on the label's scale) x D / per_pair
    for c in range(2):
        lab = full["deal_cfv"][c].sum(1) / fe.per_pair
        assert np.allclose(full["label_cfv"][c], lab, rtol=1e-5, atol=1e-7)
        # a subset's plain estimator: (D / K) x its deals' sum
        est = part["deal_cfv"][c].sum(1) * fe.D / len(S) / fe.per_pair
        assert np.allclose(part["label_cfv"][c], est, rtol=1e-5, atol=1e-7)
    # the solve converges against the network, and the labels are zero-sum (both players' range-weighted sums)
    assert full["exploitability"][1] < full["exploitability"][0] + 1e-12
    lab = full["label_cfv"][-1]
    assert abs(lab[0] @ state["range0"] + lab[1] @ state["range1"]) < 1e-5


def test_values_both_equals_values(setup):
    fe, model, state = setup
    deals = np.arange(0, fe.D, 5)
    leaf = fe.leaf(model, deals, state["line"], device="cpu")
    rng = np.random.default_rng(0)
    C = fe.pots.shape[1]
    closing = [tuple(rng.random((len(deals), 1, C, fe.H)) for _ in (0, 1))]
    both = leaf.values_both(1, closing)
    for p in (0, 1):
        assert np.array_equal(both[p][0], leaf.values(p, 1, closing)[0])
