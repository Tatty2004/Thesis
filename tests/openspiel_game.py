"""Our games as OpenSpiel Python games, so OpenSpiel's own algorithms can grade our results.

The wrapper only translates: OpenSpiel action ids 0, 1, 2 are "f", "c", "r", chance
outcomes are our (integer) outcomes, and the information state string is our infoset
key. The best responses, expected values and CFR iterations OpenSpiel computes on top
are its own code, walking the game tree its own way instead of our compiled arrays.
"""
from __future__ import annotations

import numpy as np
import pyspiel
from open_spiel.python import policy as os_policy

from toygames.games.base import CALL, CHANCE, FOLD, RAISE

ACTIONS = (FOLD, CALL, RAISE)
_GAMES: dict = {}  # id -> our game, so states refer to their game without copying it

_TYPE = pyspiel.GameType(
    short_name="python_toygames_wrapper",
    long_name="toygames game wrapped for OpenSpiel (tests only)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=False,
    provides_factored_observation_string=False)


class OpenSpielGame(pyspiel.Game):
    """`game` (any toygames Game with integer chance outcomes) as a pyspiel.Game."""

    def __init__(self, game, max_chance_outcomes: int = 64):
        info = pyspiel.GameInfo(num_distinct_actions=len(ACTIONS), max_chance_outcomes=max_chance_outcomes,
                                num_players=2, min_utility=-1000.0, max_utility=1000.0, utility_sum=0.0,
                                max_game_length=64)
        super().__init__(_TYPE, info, {})
        self.gid = id(game)
        _GAMES[self.gid] = game

    @property
    def toy(self):
        return _GAMES[self.gid]

    def new_initial_state(self):
        return OpenSpielState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return KeyObserver()


class OpenSpielState(pyspiel.State):

    def __init__(self, game):
        super().__init__(game)
        self._gid = game.gid
        self._s = _GAMES[self._gid].initial_state()

    def current_player(self):
        g = _GAMES[self._gid]
        if g.is_terminal(self._s):
            return pyspiel.PlayerId.TERMINAL
        p = g.current_player(self._s)
        return pyspiel.PlayerId.CHANCE if p == CHANCE else p

    def _legal_actions(self, player):
        return [ACTIONS.index(a) for a in _GAMES[self._gid].legal_actions(self._s)]

    def chance_outcomes(self):
        return [(int(o), float(q)) for o, q in _GAMES[self._gid].chance_outcomes(self._s)]

    def _apply_action(self, action):
        g = _GAMES[self._gid]
        self._s = g.next_state(self._s, action if g.current_player(self._s) == CHANCE else ACTIONS[action])

    def _action_to_string(self, player, action):
        return f"deal {action}" if player == pyspiel.PlayerId.CHANCE else ACTIONS[action]

    def is_terminal(self):
        return _GAMES[self._gid].is_terminal(self._s)

    def returns(self):
        g = _GAMES[self._gid]
        return list(g.returns(self._s)) if g.is_terminal(self._s) else [0.0, 0.0]

    def key(self, player: int) -> str:
        return _GAMES[self._gid].infoset_key(self._s, player)

    def __str__(self):
        return repr(self._s)


class KeyObserver:
    """Information state string = our infoset key (no tensors)."""

    def __init__(self):
        self.tensor = np.zeros(0, np.float32)
        self.dict = {}

    def set_from(self, state, player):
        pass

    def string_from(self, state, player):
        return state.key(player)


class KeyPolicy(os_policy.Policy):
    """Our strategy table {key: {action: prob}} as an OpenSpiel policy, looked up by the
    information state string, so OpenSpiel never has to enumerate the game up front."""

    def __init__(self, os_game, table: dict):
        super().__init__(os_game, [0, 1])
        self.table = table

    def action_probabilities(self, state, player_id=None):
        p = state.current_player() if player_id is None else player_id
        probs = self.table[state.information_state_string(p)]
        return {ACTIONS.index(a): q for a, q in probs.items()}
