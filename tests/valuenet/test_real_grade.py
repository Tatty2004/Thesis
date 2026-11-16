"""real_grade (the grade of a trunk whose river is safely re-solved, finished on the flop+turn
tree from the re-solves' best-response root values) against the full three-street tree: the
same river strategies written out in full and graded by RangeCFR. Small deck, torch on the CPU."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bombpot.core.solvers.range_cfr import RangeCFR, forward  # noqa: E402
from bombpot.experiments.valuenet import datagen as dg  # noqa: E402
from bombpot.experiments.valuenet import real_grade as rg  # noqa: E402
from bombpot.holdem.public import HoldemRiver, holdem_public_tree  # noqa: E402

FLOPS = ((0, 3, 7), (9, 11, 6))  # 2c3c4d / 5c6d4c on 5 ranks x 3 suits


def check(template: str, iters: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    game = dg.make_game(FLOPS, 5, 3, template)
    full = holdem_public_tree(game)
    upper = holdem_public_tree(game, upto=2)
    river = HoldemRiver(game, upper)
    turn = upper.streets[1]
    D, H = turn.valid.shape
    avg = rg.rounded([[[rng.dirichlet(np.ones(len(a)), size=(upper.streets[t].num_deals, lc.num_lines, H))
                        for a in lc.template.actions] for lc in upper.classes[t]] for t in range(2)])
    _, closing = forward(upper, avg, closing=True)
    roots = rg.roots_of(upper, closing)
    lines = rg.river_lines(upper)
    pot = np.array([2.0 * upper.classes[2][c].contrib[l] for c, l in lines])
    # T values of the size real ones have: chance weight x the other player's reach mass x part of the pot
    mass = np.where(turn.valid[None, :, None, :], roots, 0.0).sum(-1)  # (2, D, L)
    cfv = np.stack([turn.kappa[None, :, None, None][0] * mass[1 - q][..., None] * pot[None, :, None]
                    * rng.uniform(-0.3, 0.3, (D, len(lines), H)) for q in (0, 1)])
    br = np.zeros((2, D, len(lines), H))
    strategies = []
    for d in range(D):
        b, stats, strat = rg.resolve_deal(upper, river, d, roots[:, d], cfv[:, d], 0.0, iters, device="cpu",
                                          dtype=torch.float64, keep_strategy=True)
        br[:, d] = b
        strategies.append(strat)
        assert (stats[..., 0] == iters).all()
    got = rg.finish(upper, avg, br)
    sigma_river, i = [], 0
    for rc in upper.classes[2]:
        nodes = []
        for k, q in enumerate(rc.template.players):
            nodes.append(np.concatenate([np.stack([strategies[d][i + l][q][k] for l in range(rc.num_lines)], axis=1)
                                         for d in range(D)]))
        sigma_river.append(nodes)
        i += rc.num_lines
    want = RangeCFR(full).exploitability([avg[0], avg[1], sigma_river])
    for key in ("br_value_p0", "br_value_p1", "exploitability"):
        assert got[key] == pytest.approx(want[key], rel=1e-10, abs=1e-12), key
    return got, want


def test_grade_equals_the_full_tree_limit():
    check("limit", 3)


@pytest.mark.slow
def test_grade_equals_the_full_tree_pot():
    check("pot", 2)
