"""Compile a Game into flat sequence-form arrays.

Each player gets their own tree of infosets and sequences. Sequence 0 is the empty
sequence, and the sequences of infoset I are first_seq[I] ... first_seq[I] +
num_actions[I] - 1. Infosets are sorted by level (how many of their own actions
come before them), so every level's infosets and sequences are contiguous and a
whole level can be processed with one numpy call.

A terminal history is stored as (player 0's last sequence, player 1's last
sequence, chance reach, payoff to player 0). The value of any strategy profile is
sum_z x0[seq0(z)] * x1[seq1(z)] * chance(z) * u0(z), where x_p is player p's
realization plan. That's all CFR, best response and the LP need.
"""
from __future__ import annotations

import pickle
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from toygames.games.base import CHANCE


class PerfectRecallError(ValueError):
    pass


@dataclass
class Tree:
    name: str
    n_infosets: list      # [p] -> number of infosets of player p
    n_seqs: list          # [p] -> number of sequences, the empty one included
    parent_seq: list      # [p] -> (n_infosets,) sequence leading to the infoset (0 = empty)
    first_seq: list       # [p] -> (n_infosets,) first sequence of the infoset
    num_actions: list     # [p] -> (n_infosets,)
    level_infosets: list  # [p] -> (n_levels + 1,) infoset bounds of each level
    level_seqs: list      # [p] -> (n_levels + 1,) sequence bounds of each level
    seq_infoset: list     # [p] -> (n_seqs,) owning infoset, -1 for the empty sequence
    seq_parent: list      # [p] -> (n_seqs,) parent sequence of the owning infoset
    term_seq: list        # [p] -> (n_terminals,) player p's last sequence
    term_cu0: np.ndarray  # (n_terminals,) chance reach * payoff to player 0
    keys: list            # [p] -> infoset keys
    actions: list         # [p] -> action labels of each infoset
    meta: list | None = None  # [p] -> {"hand", "street", "board", "public"} lists, card games only
    term_chance: np.ndarray | None = None
    term_u0: np.ndarray | None = None
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        self.term_cu = [self.term_cu0, -self.term_cu0]
        self._index = [None, None]

    @property
    def n_terminals(self) -> int:
        return len(self.term_cu0)

    def infoset_index(self, player: int) -> dict:
        if self._index[player] is None:
            self._index[player] = {k: i for i, k in enumerate(self.keys[player])}
        return self._index[player]

    def uniform(self) -> list[np.ndarray]:
        sigma = []
        for p in (0, 1):
            s = np.ones(self.n_seqs[p])
            s[1:] = np.repeat(1.0 / self.num_actions[p], self.num_actions[p])
            sigma.append(s)
        return sigma

    def summary(self) -> str:
        return (f"{self.name}: infosets {self.n_infosets[0]} + {self.n_infosets[1]} = "
                f"{sum(self.n_infosets)}, sequences {self.n_seqs[0]} + {self.n_seqs[1]}, "
                f"terminals {self.n_terminals}")

    def save(self, path) -> None:
        state = {k: v for k, v in self.__dict__.items() if k not in ("term_cu", "_index")}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "Tree":
        with open(path, "rb") as f:
            return cls(**pickle.load(f))


def compile_game(game, keep_meta: bool = True) -> Tree:
    """Walk the whole game once and build its Tree.

    Checks along the way: chance probabilities sum to 1 at every chance node,
    returns are zero-sum, and every infoset is reached through a single own
    sequence with a single action set (perfect recall).
    """
    start = time.perf_counter()
    card_game = keep_meta and hasattr(game, "private_hand")
    index = ({}, {})
    parent, first, nact, level, keys, acts = ([], []), ([], []), ([], []), ([], []), ([], []), ([], [])
    meta = ({"hand": [], "street": [], "board": [], "public": []},
            {"hand": [], "street": [], "board": [], "public": []})
    seq_owner = ([-1], [-1])  # infoset (in creation order) owning each sequence
    t_s0, t_s1, t_ch, t_u0 = array("q"), array("q"), array("d"), array("d")
    n_chance = n_decision = 0

    stack = [(game.initial_state(), 0, 0, 1.0)]
    while stack:
        s, q0, q1, pc = stack.pop()
        if game.is_terminal(s):
            r0, r1 = game.returns(s)
            if abs(r0 + r1) > 1e-9:
                raise ValueError(f"returns not zero-sum at {s}: {(r0, r1)}")
            t_s0.append(q0)
            t_s1.append(q1)
            t_ch.append(pc)
            t_u0.append(r0)
            continue
        p = game.current_player(s)
        if p == CHANCE:
            n_chance += 1
            outs = game.chance_outcomes(s)
            if abs(sum(q for _, q in outs) - 1.0) > 1e-9:
                raise ValueError(f"chance probabilities at {s} sum to {sum(q for _, q in outs)}")
            for o, q in reversed(outs):
                stack.append((game.next_state(s, o), q0, q1, pc * q))
            continue
        n_decision += 1
        key = game.infoset_key(s, p)
        la = tuple(game.legal_actions(s))
        q = q0 if p == 0 else q1
        i = index[p].get(key)
        if i is None:
            i = len(parent[p])
            index[p][key] = i
            parent[p].append(q)
            first[p].append(len(seq_owner[p]))
            nact[p].append(len(la))
            level[p].append(0 if q == 0 else level[p][seq_owner[p][q]] + 1)
            keys[p].append(key)
            acts[p].append(la)
            seq_owner[p].extend([i] * len(la))
            if card_game:
                meta[p]["hand"].append(game.private_hand(s, p))
                meta[p]["street"].append(game.street(s))
                meta[p]["board"].append(game.board(s))
                meta[p]["public"].append(game.public_key(s))
        elif parent[p][i] != q or acts[p][i] != la:
            raise PerfectRecallError(f"infoset {key!r} reached by two different own histories")
        f = first[p][i]
        for j in range(len(la) - 1, -1, -1):
            child = game.next_state(s, la[j])
            if p == 0:
                stack.append((child, f + j, q1, pc))
            else:
                stack.append((child, q0, f + j, pc))

    # Renumber infosets by level (stable) so each level is contiguous.
    arrays = {}
    term_seq_old = [np.frombuffer(t_s0, dtype=np.int64), np.frombuffer(t_s1, dtype=np.int64)]
    for p in (0, 1):
        lv = np.asarray(level[p], dtype=np.int64)
        na_old = np.asarray(nact[p], dtype=np.int64)
        first_old = np.asarray(first[p], dtype=np.int64)
        order = np.argsort(lv, kind="stable")
        na = na_old[order]
        first_new = 1 + np.concatenate(([0], np.cumsum(na)[:-1])) if len(na) else np.zeros(0, np.int64)
        n_seqs = 1 + int(na.sum())
        seq_map = np.zeros(n_seqs, dtype=np.int64)
        offsets = np.arange(n_seqs - 1) - np.repeat(first_new - 1, na)  # action index within infoset
        seq_map[np.repeat(first_old[order], na) + offsets] = np.repeat(first_new, na) + offsets
        parent_new = seq_map[np.asarray(parent[p], dtype=np.int64)[order]]
        seq_infoset = np.full(n_seqs, -1, dtype=np.int64)
        seq_infoset[1:] = np.repeat(np.arange(len(na)), na)
        seq_parent = np.zeros(n_seqs, dtype=np.int64)
        seq_parent[1:] = parent_new[seq_infoset[1:]]
        lv_sorted = lv[order]
        n_levels = int(lv_sorted.max()) + 1 if len(lv_sorted) else 0
        level_inf = np.searchsorted(lv_sorted, np.arange(n_levels + 1))
        level_seq = np.append(first_new, n_seqs)[level_inf]
        arrays[p] = dict(
            n_infosets=len(na), n_seqs=n_seqs, parent_seq=parent_new, first_seq=first_new,
            num_actions=na, level_infosets=level_inf, level_seqs=level_seq,
            seq_infoset=seq_infoset, seq_parent=seq_parent,
            term_seq=seq_map[term_seq_old[p]],
            keys=[keys[p][i] for i in order], actions=[acts[p][i] for i in order],
            meta={k: [v[i] for i in order] for k, v in meta[p].items()} if card_game else None,
        )

    chance = np.frombuffer(t_ch, dtype=np.float64).copy()
    u0 = np.frombuffer(t_u0, dtype=np.float64).copy()
    tree = Tree(
        name=game.name,
        **{k: [arrays[0][k], arrays[1][k]] for k in (
            "n_infosets", "n_seqs", "parent_seq", "first_seq", "num_actions", "level_infosets",
            "level_seqs", "seq_infoset", "seq_parent", "term_seq", "keys", "actions")},
        term_cu0=chance * u0,
        meta=[arrays[0]["meta"], arrays[1]["meta"]] if card_game else None,
        term_chance=chance,
        term_u0=u0,
        stats={"decision_nodes": n_decision, "chance_nodes": n_chance, "terminals": len(u0),
               "compile_seconds": time.perf_counter() - start},
    )
    return tree


def load_or_compile(game, cache_dir) -> Tree:
    """Compile `game`, reusing a pickled tree from `cache_dir` when there is one."""
    import hashlib

    path = Path(cache_dir) / f"{hashlib.sha1(game.name.encode()).hexdigest()[:16]}.pkl"
    if path.exists():
        tree = Tree.load(path)
        if tree.name == game.name:
            return tree
    tree = compile_game(game)
    tree.save(path)
    return tree


# Strategies ---------------------------------------------------------------------
# A behavioural strategy for player p is an array over p's sequences: sigma[s] is the
# probability of the last action of sequence s at its infoset (sigma[0] = 1).


def infoset_sums(tree: Tree, p: int, v: np.ndarray) -> np.ndarray:
    """Sum of v over the sequences of each infoset."""
    if tree.n_infosets[p] == 0:
        return np.zeros(0)
    return np.add.reduceat(v[1:], tree.first_seq[p] - 1)


def normalize(tree: Tree, p: int, w: np.ndarray) -> np.ndarray:
    """Behavioural strategy proportional to non-negative weights w (uniform where they sum to 0)."""
    na = tree.num_actions[p]
    total = np.repeat(infoset_sums(tree, p, w), na)
    sigma = np.ones(tree.n_seqs[p])
    positive = total > 0
    sigma[1:] = np.repeat(1.0 / na, na)
    sigma[1:][positive] = w[1:][positive] / total[positive]
    return sigma


def realization_plan(tree: Tree, p: int, sigma: np.ndarray) -> np.ndarray:
    """x[s] = probability that player p plays every action of sequence s."""
    x = np.empty(tree.n_seqs[p])
    x[0] = 1.0
    bounds = tree.level_seqs[p]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        x[lo:hi] = x[tree.seq_parent[p][lo:hi]] * sigma[lo:hi]
    return x


def behavioural_from_plan(tree: Tree, p: int, x: np.ndarray) -> np.ndarray:
    """Invert realization_plan (uniform at infosets the plan never reaches)."""
    return normalize(tree, p, np.maximum(x, 0.0))


def profile_value(tree: Tree, sigma) -> float:
    """Player 0's expected payoff under the profile (sigma0, sigma1)."""
    x0 = realization_plan(tree, 0, sigma[0])
    x1 = realization_plan(tree, 1, sigma[1])
    return float(np.dot(tree.term_cu0, x0[tree.term_seq[0]] * x1[tree.term_seq[1]]))


def strategy_table(tree: Tree, sigma) -> dict:
    """{infoset key: {action: probability}} for both players."""
    table = {}
    for p in (0, 1):
        for i, key in enumerate(tree.keys[p]):
            f = tree.first_seq[p][i]
            table[key] = {a: float(sigma[p][f + j]) for j, a in enumerate(tree.actions[p][i])}
    return table


def strategy_from_table(tree: Tree, table: dict) -> list[np.ndarray]:
    """Inverse of strategy_table; infosets missing from the table play uniformly."""
    sigma = tree.uniform()
    for p in (0, 1):
        for i, key in enumerate(tree.keys[p]):
            if key in table:
                f = tree.first_seq[p][i]
                for j, a in enumerate(tree.actions[p][i]):
                    sigma[p][f + j] = table[key][a]
    return sigma
