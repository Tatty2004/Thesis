"""The public tree of a limit-poker game, for solvers that work on ranges.

A public node is (street t, deal d, entering line l, betting node k):
  deal d       the public cards so far (one entry of `deals(t)`'s boards)
  line l       the betting on earlier streets, as a mixed-radix index over each street's
               continuing lines (lines of street t + 1 = lines of t x continuing lines of t)
  node k       a node of street t's betting template, which is the same for every deal and
               every line because limit betting ignores the cards
Every (private hand, public node) pair is one infoset of the full game.

Chance enters only through W[t][d, i, j] = P(hand0 = i, hand1 = j, public cards = deal d),
so a counterfactual value is a matrix-vector product: player 0's value at a fold is
payoff x (W @ reach1), and at a showdown pot x (S @ reach1) - contribution x (W @ reach1)
with S = W x player 0's average board share. Each Street supplies these two products
(`fold`, `share`). build_public_tree makes them dense, from `CardGame.deals` and
`showdown`; a big game supplies a faster Street with the same methods.

Only `LimitPoker` games are supported: the betting template is read off the engine itself.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from bombpot.core.game import CHANCE
from bombpot.core.limit import CALL


@dataclass
class Template:
    """One street's betting. Contributions are relative to the street's start, when both
    players have the same amount in."""
    players: list   # [k] acting player at decision node k (nodes in creation order: parents first)
    actions: list   # [k] legal actions at node k
    hist: list      # [k] the street's betting string before node k acts
    children: list  # [k][a] ("node", k2) | ("fold", folder, chips the folder added) | ("close", c)
    closes: list    # [c] (betting string, chips each player added) for each line that ends the street calmly

    @property
    def num_nodes(self) -> int:
        return len(self.players)

    @property
    def num_closes(self) -> int:
        return len(self.closes)


class Street:
    """One street's deals and the chance-weighted operators the range solver needs.

    For player 0's reach r1 over player 1's hands (and symmetrically for player 1):
      fold(r1, 0)[d, l, i]  = sum_j W[d, i, j] r1[d, l, j]
      share(r1, 0)[d, l, i] = sum_j S[d, i, j] r1[d, l, j]
    with W = P(hand0, hand1, deal) and S = W x player 0's average share of the boards
    (only needed on the last street)."""
    deals: list          # [d] public cards (the game's board tuple) at this street
    parent: np.ndarray   # [d] index of the deal's prefix on the previous street (-1 on street 0)
    index: dict          # deal -> d

    def fold(self, reach: np.ndarray, player: int) -> np.ndarray:
        raise NotImplementedError

    def share(self, reach: np.ndarray, player: int) -> np.ndarray:
        raise NotImplementedError

    def possible(self, player: int) -> np.ndarray:
        """(D, H): whether the player can hold each hand at each deal."""
        raise NotImplementedError


class DenseStreet(Street):
    """W and S as dense (D, H, H) arrays: exact and simple, for small games."""

    def __init__(self, deals, parent, W, S=None):
        self.deals, self.parent, self.W, self.S = deals, parent, W, S
        self.index = {b: i for i, b in enumerate(deals)}

    @staticmethod
    def _apply(M, reach, player):
        if player == 0:
            return np.einsum("dij,dlj->dli", M, reach)
        return np.einsum("dij,dli->dlj", M, reach)

    def fold(self, reach, player):
        return self._apply(self.W, reach, player)

    def share(self, reach, player):
        return self._apply(self.S, reach, player)

    def possible(self, player):
        return (self.W.sum(axis=2) if player == 0 else self.W.sum(axis=1)) > 0


@dataclass
class PublicTree:
    name: str
    hands: list           # private hands (one list for both players)
    streets: list         # [t] Street (DenseStreet for small games)
    templates: list       # [t] Template
    contrib: list         # [t] (L_t,) chips each player has in when street t's betting starts

    @property
    def num_streets(self) -> int:
        return len(self.streets)

    def num_lines(self, t: int) -> int:
        return len(self.contrib[t])

    def shape(self, t: int) -> tuple:
        return (len(self.streets[t].deals), self.num_lines(t), len(self.hands))


def _street_start(game, t: int):
    """A state where street t's betting begins (earlier streets checked down)."""
    s = game.initial_state()
    while True:
        p = game.current_player(s)
        if p == CHANCE:
            s = game.next_state(s, game.chance_outcomes(s)[0][0])
        elif game.street(s) == t:
            return s
        else:
            s = game.next_state(s, CALL)


def _template(game, t: int) -> Template:
    start = _street_start(game, t)
    base = start.contrib[0]
    players, actions, hist, children, closes = [], [], [], [], []
    queue = [start]
    while queue:
        s = queue.pop(0)
        k = len(players)
        players.append(game.current_player(s))
        acts = tuple(game.legal_actions(s))
        actions.append(acts)
        hist.append(s.hist[t])
        row = []
        for a in acts:
            c = game.next_state(s, a)
            if c.folder >= 0:
                row.append(("fold", c.folder, c.contrib[c.folder] - base))
            elif game.is_terminal(c) or c.step != s.step:  # showdown, or the street is over
                row.append(("close", len(closes)))
                closes.append((c.hist[t], c.contrib[0] - base))
            else:
                row.append(("node", k + 1 + len(queue)))
                queue.append(c)
        children.append(row)
    return Template(players, actions, hist, children, closes)


def build_public_tree(game) -> PublicTree:
    """The public tree of a LimitPoker game with dense chance matrices (small games only)."""
    T = game.num_streets
    joint = []
    hands = set()
    for t in range(T):
        d = defaultdict(float)
        for board, h0, h1, q in game.deals(t):
            d[(board, h0, h1)] += q
            hands.update((h0, h1))
        joint.append(d)
    hands = sorted(hands)
    hidx = {h: i for i, h in enumerate(hands)}
    H = len(hands)

    streets = []
    for t in range(T):
        deals = sorted({b for b, _, _ in joint[t]})
        index = {b: i for i, b in enumerate(deals)}
        W = np.zeros((len(deals), H, H))
        for (b, h0, h1), q in joint[t].items():
            W[index[b], hidx[h0], hidx[h1]] += q
        if t == 0:
            parent = np.full(len(deals), -1, dtype=np.int64)
        else:
            prev = streets[t - 1]
            n = len(prev.deals[0])
            parent = np.array([prev.index[b[:n]] for b in deals], dtype=np.int64)
        streets.append(DenseStreet(deals, parent, W))
    last = streets[-1]
    S = np.zeros_like(last.W)
    for (b, h0, h1), q in joint[-1].items():
        S[last.index[b], hidx[h0], hidx[h1]] += q * float(np.mean(game.showdown(b, h0, h1)))
    last.S = S

    templates = [_template(game, t) for t in range(T)]
    contrib = [np.array([_street_start(game, 0).contrib[0]], dtype=float)]
    for t in range(T - 1):
        extra = np.array([e for _, e in templates[t].closes])
        contrib.append((contrib[t][:, None] + extra[None, :]).ravel())
    return PublicTree(game.name, hands, streets, templates, contrib)


# Coordinates of the compiled tree's infosets -----------------------------------------------


def tree_coordinates(pt: PublicTree, tree, player: int) -> np.ndarray:
    """(t, d, l, k, i) of every infoset of `player` in a compiled Tree of the same game.

    Uses the tree's card-game metadata, and the LimitPoker key convention that the public
    key ends with ":" followed by each street's betting joined by "/"."""
    meta = tree.meta[player]
    hidx = {h: i for i, h in enumerate(pt.hands)}
    node_of = [{h: k for k, h in enumerate(tp.hist)} for tp in pt.templates]
    close_of = [{h: c for c, (h, _) in enumerate(tp.closes)} for tp in pt.templates]
    out = np.empty((tree.n_infosets[player], 5), dtype=np.int64)
    for n in range(tree.n_infosets[player]):
        t = meta["street"][n]
        bets = meta["public"][n].rsplit(":", 1)[1].split("/")
        line = 0
        for u in range(t):
            line = line * pt.templates[u].num_closes + close_of[u][bets[u]]
        out[n] = (t, pt.streets[t].index[meta["board"][n]], line, node_of[t][bets[t]], hidx[meta["hand"][n]])
    return out


def to_tree_sigma(pt: PublicTree, tree, hand_sigma) -> list:
    """A per-hand range strategy (hand_sigma[t][k] shaped (D, L, H, actions)) as the tree
    solver's per-sequence arrays."""
    out = []
    for p in (0, 1):
        co = tree_coordinates(pt, tree, p)
        s = np.ones(tree.n_seqs[p])
        for n, (t, d, l, k, i) in enumerate(co):
            f = tree.first_seq[p][n]
            s[f:f + tree.num_actions[p][n]] = hand_sigma[t][k][d, l, i]
        out.append(s)
    return out


def from_tree_sigma(pt: PublicTree, tree, sigma) -> list:
    """The tree solver's per-sequence strategy as per-hand range arrays (uniform at hands the
    tree has no infoset for, which can't be held there)."""
    out = [[np.full(pt.shape(t) + (len(a),), 1.0 / len(a)) for a in tp.actions]
           for t, tp in enumerate(pt.templates)]
    for p in (0, 1):
        co = tree_coordinates(pt, tree, p)
        for n, (t, d, l, k, i) in enumerate(co):
            f = tree.first_seq[p][n]
            out[t][k][d, l, i] = sigma[p][f:f + tree.num_actions[p][n]]
    return out
