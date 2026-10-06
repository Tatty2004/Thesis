"""Brute-force references that never touch the compiled tree, plus OpenSpiel glue."""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from bombpot.core.game import CHANCE


def naive_value(game, table: dict) -> float:
    """Player 0's expected payoff by plain recursion over the Game protocol."""

    def rec(s):
        if game.is_terminal(s):
            return game.returns(s)[0]
        p = game.current_player(s)
        if p == CHANCE:
            return sum(q * rec(game.next_state(s, o)) for o, q in game.chance_outcomes(s))
        probs = table[game.infoset_key(s, p)]
        return sum(probs[a] * rec(game.next_state(s, a)) for a in game.legal_actions(s) if probs[a] > 0)

    return rec(game.initial_state())


def brute_force_best_response(game, tree, player: int, table: dict) -> float:
    """Best-response value by trying every pure strategy of `player` (tiny games only)."""
    keys = tree.keys[player]
    best = -np.inf
    for choice in itertools.product(*[tree.actions[player][i] for i in range(len(keys))]):
        t = dict(table)
        for key, a, acts in zip(keys, choice, tree.actions[player]):
            t[key] = {b: float(b == a) for b in acts}
        v = naive_value(game, t)
        best = max(best, v if player == 0 else -v)
    return best


def random_table(tree, seed: int) -> dict:
    """A random behavioural strategy for both players, as {key: {action: prob}}."""
    rng = np.random.default_rng(seed)
    table = {}
    for p in (0, 1):
        for key, acts in zip(tree.keys[p], tree.actions[p]):
            probs = rng.dirichlet(np.ones(len(acts)))
            table[key] = dict(zip(acts, probs))
    return table


# OpenSpiel ------------------------------------------------------------------------


def lockstep(os_game, game, outcome_map=lambda o: o) -> dict:
    """Walk OpenSpiel's tree and ours together and check they are the same game.

    Asserts identical legal actions (same count, same order), identical chance
    probabilities (OpenSpiel outcomes aggregated through `outcome_map`), and
    identical returns. Returns {OpenSpiel infostate string: (our key, OpenSpiel
    legal actions)}, checking that every OpenSpiel infoset maps to one key.
    """
    mapping = {}

    def rec(os_s, s):
        assert os_s.is_terminal() == game.is_terminal(s)
        if os_s.is_terminal():
            assert np.allclose(os_s.returns(), game.returns(s), atol=1e-12)
            return
        if os_s.is_chance_node():
            assert game.current_player(s) == CHANCE
            ours = dict(game.chance_outcomes(s))
            agg = defaultdict(float)
            for o, q in os_s.chance_outcomes():
                agg[outcome_map(o)] += q
            assert set(agg) == set(ours)
            assert all(abs(agg[k] - ours[k]) < 1e-12 for k in agg)
            for o, _ in os_s.chance_outcomes():
                rec(os_s.child(o), game.next_state(s, outcome_map(o)))
            return
        p = os_s.current_player()
        assert p == game.current_player(s)
        info = os_s.information_state_string(p)
        key = game.infoset_key(s, p)
        os_actions = os_s.legal_actions()
        ours = game.legal_actions(s)
        assert len(os_actions) == len(ours)
        assert mapping.setdefault(info, (key, tuple(os_actions))) == (key, tuple(os_actions))
        for a_os, a in zip(os_actions, ours):
            rec(os_s.child(a_os), game.next_state(s, a))

    rec(os_game.new_initial_state(), game.initial_state())
    return mapping


def to_openspiel_policy(os_game, mapping: dict, tree, table: dict):
    """Our strategy table as an OpenSpiel TabularPolicy."""
    from open_spiel.python import policy as os_policy

    pol = os_policy.TabularPolicy(os_game)
    actions = {}
    for p in (0, 1):
        for key, acts in zip(tree.keys[p], tree.actions[p]):
            actions[key] = acts
    for info, idx in pol.state_lookup.items():
        key, os_actions = mapping[info]
        probs = np.zeros(os_game.num_distinct_actions())
        for a_os, a in zip(os_actions, actions[key]):
            probs[a_os] = table[key][a]
        pol.action_probability_array[idx] = probs
    return pol


def from_openspiel_policy(pol, mapping: dict, tree) -> dict:
    """An OpenSpiel TabularPolicy as our strategy table (needs a one-to-one mapping)."""
    actions = {}
    for p in (0, 1):
        for key, acts in zip(tree.keys[p], tree.actions[p]):
            actions[key] = acts
    table = {}
    for info, idx in pol.state_lookup.items():
        key, os_actions = mapping[info]
        table[key] = {a: float(pol.action_probability_array[idx][a_os])
                      for a_os, a in zip(os_actions, actions[key])}
    return table
