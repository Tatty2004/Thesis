"""R4: the range solver on real-sized Hold'em, checked by brute force and by symmetry.

- On the 52-card deck, one turn deal's values (both players, every hand, every entering
  line) are recomputed by brute force: every pair of hands, every betting path of the
  street, payoffs from the hand evaluator. Nothing is shared with the solver's sweeps.
- Swapping the two boards, and relabeling suits that neither flop uses, map the game
  to itself, so any strategy and its image must have the same best responses and value.
"""
import numpy as np
import pytest

from bombpot.core.solvers.range_cfr import RangeCFR
from bombpot.holdem import Holdem
from bombpot.holdem.cards import SUIT_CHARS
from bombpot.holdem.public import holdem_public_tree


def random_sigma(pt, seed):
    rng = np.random.default_rng(seed)
    out = []
    for t, tp in enumerate(pt.templates):
        row = []
        for a in tp.actions:
            x = rng.random(pt.shape(t) + (len(a),)) ** 3  # lopsided, so mistakes show
            row.append(x / x.sum(-1, keepdims=True))
        out.append(row)
    return out


def brute_force_values(game, pt, sigma, reach, d, line, player):
    """Player's counterfactual value of every hand at the last street's root, deal d."""
    st, tp = pt.streets[-1], pt.templates[-1]
    hands = pt.hands
    H = len(hands)
    key = st.deals[d]
    boards = [key[j::game.num_boards] for j in range(game.num_boards)]
    public = set(key)
    valid = np.array([not public & set(h) for h in hands])
    disjoint = np.array([[not set(a) & set(b) for b in hands] for a in hands])
    W = st.kappa[d] * disjoint * valid[:, None] * valid[None, :]
    strength = np.array([[game.hand_value(h + b) if v else 0 for h, v in zip(hands, valid)] for b in boards])
    share0 = np.mean([(s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]) for s in strength], axis=0)
    o = 1 - player
    root_o = reach[-1][0][o][d, line]
    contrib = pt.contrib[-1][line]
    out = np.zeros(H)

    def walk(k, path):  # path: per player, the probability of their actions so far, per hand
        for a, child in enumerate(tp.children[k]):
            q = tp.players[k]
            nxt = [path[0], path[1]]
            nxt[q] = path[q] * sigma[-1][k][d, line, :, a]
            if child[0] == "node":
                walk(child[1], nxt)
                continue
            if child[0] == "fold":
                lost = contrib + child[2]
                u0 = np.full((H, H), -lost if child[1] == 0 else lost)
            else:
                chips = contrib + tp.closes[child[1]][1]
                u0 = chips * (2 * share0 - 1)
            if player == 0:
                out[:] += nxt[0] * ((W * u0) @ (root_o * nxt[1]))
            else:
                out[:] += nxt[1] * ((W * -u0).T @ (root_o * nxt[0]))

    walk(0, [np.ones(H), np.ones(H)])
    return out


@pytest.mark.slow
def test_real_deck_values_match_brute_force():
    game = Holdem(flops=("Ah7c2d", "KsKd9h"))
    pt = holdem_public_tree(game)
    solver = RangeCFR(pt)
    sigma = random_sigma(pt, 0)
    reach = solver.reach(sigma)
    for player in (0, 1):
        values, _ = solver.values(player, sigma, reach)
        for d in (0, 777, 2069):
            for line in (0, 3):
                want = brute_force_values(game, pt, sigma, reach, d, line, player)
                np.testing.assert_allclose(values[-1][0][d, line], want, rtol=1e-11, atol=1e-16)


# Symmetries ------------------------------------------------------------------------------

DECK = dict(num_ranks=9, num_suits=4, board_cards=(3, 1))  # 2 to T: 435 hands, 870 turn deals


def permuted(pt_from, pt_to, sigma, card_map):
    """sigma moved through a relabeling of the cards (and so of hands and deals)."""
    hidx = {h: i for i, h in enumerate(pt_to.hands)}
    hand_perm = np.array([hidx[tuple(sorted(card_map[c] for c in h))] for h in pt_from.hands])
    out = []
    for t, (sf, stt) in enumerate(zip(pt_from.streets, pt_to.streets)):
        deal_perm = np.array([stt.index[tuple(card_map[c] for c in key)] for key in sf.deals])
        row = []
        for s in sigma[t]:
            x = np.empty_like(s)
            x[np.ix_(deal_perm, np.arange(s.shape[1]), hand_perm)] = s
            row.append(x)
        out.append(row)
    return out


def assert_same_grades(pt_a, sig_a, pt_b, sig_b):
    a, b = RangeCFR(pt_a).exploitability(sig_a), RangeCFR(pt_b).exploitability(sig_b)
    for key in a:
        assert b[key] == pytest.approx(a[key], abs=1e-12)


def test_swapping_boards_changes_nothing():
    g_ab = Holdem(flops=("2c5d9h", "8s8d3c"), **DECK)
    g_ba = Holdem(flops=("8s8d3c", "2c5d9h"), **DECK)
    pt_ab, pt_ba = holdem_public_tree(g_ab), holdem_public_tree(g_ba)
    swap = {}
    for t, (sa, sb) in enumerate(zip(pt_ab.streets, pt_ba.streets)):
        for key in sa.deals:  # (a1, b1, a2, b2, ...) -> (b1, a1, b2, a2, ...)
            swap[key] = tuple(c for pair in zip(key[1::2], key[0::2]) for c in pair)
    sigma = random_sigma(pt_ab, 1)
    hidx = {h: i for i, h in enumerate(pt_ba.hands)}
    assert [hidx[h] for h in pt_ab.hands] == list(range(len(pt_ab.hands)))
    moved = []
    for t, (sa, sb) in enumerate(zip(pt_ab.streets, pt_ba.streets)):
        perm = np.array([sb.index[swap[key]] for key in sa.deals])
        row = []
        for s in sigma[t]:
            x = np.empty_like(s)
            x[perm] = s
            row.append(x)
        moved.append(row)
    assert_same_grades(pt_ab, sigma, pt_ba, moved)


def test_relabeling_unused_suits_changes_nothing():
    # Neither flop has a diamond or a spade, so swapping those suits maps the game to itself.
    game = Holdem(flops=("2c5h9h", "8c8h3c"), **DECK)
    pt = holdem_public_tree(game)
    d, s = SUIT_CHARS.index("d"), SUIT_CHARS.index("s")
    card_map = {}
    for c in game.deck.cards:
        suit = game.deck.suit(c)
        card_map[c] = game.deck.card(game.deck.rank(c), {d: s, s: d}.get(suit, suit))
    sigma = random_sigma(pt, 2)
    assert_same_grades(pt, sigma, pt, permuted(pt, pt, sigma, card_map))
    # A strategy that ignores the symmetry is a different strategy, so the check has teeth:
    assert not all(np.array_equal(a, b) for a, b in zip(sigma[1], permuted(pt, pt, sigma, card_map)[1]))
