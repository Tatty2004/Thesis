"""M1's run script (bombpot/experiments/m1.py): the pieces its gates rely on.
  - Fitting one turn deal at a time on a shared generator gives the same centers, bit for
    bit, as one fit_equity_buckets call, and any one turn deal refits alone from the
    generator state recorded before it (Gate 3's determinism check).
  - Retention keeps the newest 2 checkpoints and the first at or after each milestone.
  - The real-deck configs hold M1's fixed choices.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from bombpot.core.abstraction.river import fit_equity_buckets
from bombpot.experiments.m1 import fit_by_parent, keep_set, make_game, refit_one, same_bits
from bombpot.holdem import Holdem
from bombpot.holdem.public import HoldemRiver, holdem_public_tree

CONFIGS = Path(__file__).resolve().parents[2] / "bombpot" / "experiments" / "configs" / "holdem"
GAME = Holdem(num_ranks=5, num_suits=3, board_cards=(3, 1, 1), flops=("2c3c4d", "5c6d4c"), bet_sizes=(2, 4, 4))


@pytest.mark.parametrize("method", ["avg_1d", "kmeans_2d"])
def test_fitting_by_turn_deal_is_one_fit(method):
    upper = holdem_public_tree(GAME, upto=2)
    river = HoldemRiver(GAME, upper)
    D1 = upper.streets[-1].num_deals
    whole = fit_equity_buckets(river, D1, method, 3, samples=12, seed=0)
    parts, states = fit_by_parent(river, D1, method, 3, samples=12, seed=0, restarts=3)
    assert same_bits(parts.centers, whole.centers)
    assert (parts.method, parts.k, parts.fit) == (whole.method, whole.k, whole.fit)
    for p in (0, 1, D1 // 2, D1 - 1):
        assert same_bits(refit_one(river, p, method, 3, 12, states[p], 3), whole.centers[p])
    assert not same_bits(refit_one(river, 1, method, 3, 12, states[0], 3), whole.centers[1])  # the state matters


def test_retention_keeps_the_newest_two_and_the_first_after_each_milestone():
    entries = [(100, 1.9), (200, 2.0), (300, 3.5), (400, 4.1), (500, 6.0), (600, 7.9), (700, 8.0), (800, 9.0)]
    assert keep_set(entries, [2, 4, 8, 16]) == {200, 400, 700, 800}
    assert keep_set(entries[:3], [2, 4]) == {200, 300}
    assert keep_set(entries[:1], [2]) == {100}
    assert keep_set([], [2]) == set()
    assert keep_set([(5, 0.5), (9, 0.9), (12, 1.3), (20, 2.1)], [0.02, 1.0, 2.0]) == {5, 12, 20}


@pytest.mark.parametrize("name, method", [("m1_kmeans2d", "kmeans_2d"), ("m1_avg1d", "avg_1d")])
def test_real_deck_configs_hold_the_fixed_choices(name, method):
    cfg = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    assert make_game(cfg).params == Holdem(flops=("Ah7c2d", "KsKd9h"), board_cards=(3, 1, 1)).params
    assert cfg["buckets"] == {"method": method, "k": 200, "samples": 50, "seed": 0, "restarts": 3}
    assert cfg["solver"]["seed"] == 0 and cfg["solver"]["samples"] == 4 and cfg["solver"]["batch"] in (64, 256)
    assert not cfg["run"].get("eval_in_process", False)  # never the exact check inside a real-deck solve
