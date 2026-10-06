"""Board-Leduc: an independent reference implementation, for cross-checking the main one.

Written from the plain-language rules alone and deliberately naive: every card is a real
suited card, every deal enumerates the cards still available, and nothing is merged by
symmetry or cached. Standard library only.

A card is a (rank, suit) pair: rank 0..N-1 (0 is lowest), suit 0 or 1.
A state is a tuple (hands, boards, history) holding only what has happened so far:
    hands    private cards dealt so far: (), (p0's card,), then (p0's card, p1's card)
    boards   one tuple of cards per board (board A first), each in deal order
    history  one string of actions per betting round that has begun, e.g. ("rc", "")
Whose turn it is, the legal actions and the chips put in are all derived from it.

Run this file to walk small trees and check hand-worked examples.
"""

CHANCE = -1
TERMINAL = -2


class ReferenceBoardLeduc:
    """Heads-up Board-Leduc: 1 or 2 boards, 1 or 2 streets, three deck modes."""

    def __init__(self, num_ranks=3, num_boards=2, streets=1, deck_mode="shared",
                 ante=1.0, bet_sizes=(2.0, 4.0), max_raises=2):
        if num_ranks < 3 or num_boards not in (1, 2) or streets not in (1, 2):
            raise ValueError("need num_ranks >= 3, num_boards in (1, 2), streets in (1, 2)")
        if deck_mode not in ("shared", "independent", "identical"):
            raise ValueError(f"unknown deck_mode {deck_mode!r}")
        if isinstance(max_raises, int):
            max_raises = [max_raises] * streets              # the same cap on every street
        if len(bet_sizes) < streets or len(max_raises) < streets:
            raise ValueError("need a bet size and a raise cap for every street")
        self.num_ranks, self.num_boards, self.streets = num_ranks, num_boards, streets
        self.deck_mode = deck_mode
        self.ante = float(ante)
        self.bet_sizes = tuple(float(size) for size in bet_sizes[:streets])
        self.max_raises = tuple(int(cap) for cap in max_raises[:streets])
        self.deck = tuple((rank, suit) for rank in range(num_ranks) for suit in (0, 1))
        self.name = (f"ReferenceBoardLeduc(num_ranks={num_ranks}, num_boards={num_boards}, "
                     f"streets={streets}, deck_mode={deck_mode!r}, ante={self.ante}, "
                     f"bet_sizes={self.bet_sizes}, max_raises={self.max_raises})")

    def initial_state(self):
        """Antes posted (they are counted in _put_in), nothing dealt."""
        return ((), ((),) * self.num_boards, ())

    def current_player(self, state):
        hands, boards, history = state
        if len(hands) < 2 or not history:
            return CHANCE                  # private cards or street-1 board cards still to come
        actions = history[-1]              # the latest betting round
        if actions.endswith("f"):
            return TERMINAL                # a fold ends the hand at once
        if actions == "cc" or actions.endswith("rc"):    # both checked, or a bet was called
            return TERMINAL if len(history) == self.streets else CHANCE   # else next street
        return len(actions) % 2            # player 0 acts first every street; turns alternate

    def is_terminal(self, state):
        return self.current_player(state) == TERMINAL

    def legal_actions(self, state):
        if self.current_player(state) not in (0, 1):
            raise ValueError("not a decision node")
        history = state[2]
        actions = history[-1]
        # Both players have put in the same amount when a street starts. A check or a call keeps
        # it that way and a bet or raise breaks it, so a player faces a bet exactly after an "r".
        legal = ("f", "c") if actions.endswith("r") else ("c",)
        if actions.count("r") < self.max_raises[len(history) - 1]:   # bets + raises this street
            legal += ("r",)
        return legal

    def _board_to_deal(self, state):
        """Index of the board that gets the next card: street by street, board A before B."""
        hands, boards, history = state
        street = len(history) + 1          # while boards are dealt, history has finished rounds
        return next(index for index, board in enumerate(boards) if len(board) < street)

    def _cards_for_next_deal(self, state):
        """Every card the next deal can produce, per the deck mode. Each is equally likely."""
        if self.current_player(state) != CHANCE:
            raise ValueError("not a chance node")
        hands, boards, history = state
        if len(hands) < 2:
            gone = hands                                        # private cards: from the deck
        elif self.deck_mode == "shared":                        # one deck for everything
            gone = hands + tuple(card for board in boards for card in board)
        elif self.deck_mode == "independent":                   # this board's own deck copy
            gone = hands + boards[self._board_to_deal(state)]
        else:                                                   # identical: deal board A only
            gone = hands + boards[0]
        return [card for card in self.deck if card not in gone]

    def chance_outcomes(self, state):
        cards = self._cards_for_next_deal(state)
        return [(card, 1.0 / len(cards)) for card in cards]

    def next_state(self, state, action_or_outcome):
        hands, boards, history = state
        if self.current_player(state) != CHANCE:
            if action_or_outcome not in self.legal_actions(state):
                raise ValueError(f"illegal action {action_or_outcome!r}")
            return (hands, boards, history[:-1] + (history[-1] + action_or_outcome,))
        card = action_or_outcome
        if card not in self._cards_for_next_deal(state):
            raise ValueError(f"card {card!r} cannot be dealt here")
        if len(hands) < 2:
            return (hands + (card,), boards, history)           # p0's card, then p1's
        if self.deck_mode == "identical":
            boards = tuple(board + (card,) for board in boards)     # board B copies board A
        else:
            i = self._board_to_deal(state)
            boards = boards[:i] + (boards[i] + (card,),) + boards[i + 1:]
        if all(len(board) == len(history) + 1 for board in boards):
            history += ("",)               # every board has this street's card: betting begins
        return (hands, boards, history)

    def _put_in(self, history):
        """Chips each player has put in: the ante plus every bet, raise and call."""
        put_in = [self.ante, self.ante]
        for street, actions in enumerate(history):
            for i, action in enumerate(actions):
                me, other = i % 2, 1 - i % 2        # player 0 acts first; turns alternate
                if action == "r":                   # match the opponent, then add the bet size
                    put_in[me] = put_in[other] + self.bet_sizes[street]
                elif action == "c":                 # a call matches; a check changes nothing
                    put_in[me] = put_in[other]
        return put_in

    @staticmethod
    def _hand(card, board):
        """Hand on one board as (1 for a pair or 0 for high card, own rank); bigger wins."""
        rank = card[0]
        pair = any(board_card[0] == rank for board_card in board)
        return (1 if pair else 0, rank)

    def returns(self, state):
        if self.current_player(state) != TERMINAL:
            raise ValueError("not a terminal")
        hands, boards, history = state
        put_in = self._put_in(history)
        if history[-1].endswith("f"):
            folder = (len(history[-1]) - 1) % 2     # whoever made the last action
            lost = put_in[folder]                   # the folder loses it, the other wins it
            return (-lost, lost) if folder == 0 else (lost, -lost)
        share = (put_in[0] + put_in[1]) / self.num_boards     # each board's share of the pot
        received = [0.0, 0.0]
        for board in boards:
            hand0, hand1 = self._hand(hands[0], board), self._hand(hands[1], board)
            if hand0 > hand1:
                received[0] += share
            elif hand1 > hand0:
                received[1] += share
            else:                                   # equal hands split this board's share
                received[0] += share / 2
                received[1] += share / 2
        return (received[0] - put_in[0], received[1] - put_in[1])

    def infoset_key(self, state, player):
        """Player, own rank, each board's ranks in deal order, betting per street. No suits."""
        hands, boards, history = state
        board_ranks = "|".join(",".join(str(card[0]) for card in board) for board in boards)
        return f"P{player} own={hands[player][0]} boards={board_ranks} bets={'/'.join(history)}"

    def describe(self, state):
        hands, boards, history = state
        return (tuple(card[0] for card in hands),
                tuple(tuple(card[0] for card in board) for board in boards),
                history)


def _walk(game, state, reach, stats):
    """Visit and check every node below `state`. Players follow a fixed strategy that bets more
    with a higher rank or a pair on board A, so the value depends on the cards."""
    player = game.current_player(state)
    if player == CHANCE:
        outcomes = game.chance_outcomes(state)
        cards = [card for card, _ in outcomes]
        assert cards and len(set(cards)) == len(cards) and all(p > 0 for _, p in outcomes)
        assert abs(sum(p for _, p in outcomes) - 1.0) < 1e-12
        for card, p in outcomes:
            _walk(game, game.next_state(state, card), reach * p, stats)
        return
    hands, boards, history = game.describe(state)
    assert len(hands) == 2 and all(len(board) == len(history) for board in boards)
    if player == TERMINAL:
        r0, r1 = game.returns(state)
        assert abs(r0 + r1) < 1e-9, (hands, boards, history, r0, r1)
        stats["terminals"] += 1
        stats["value"] += reach * r0
        return
    actions = game.legal_actions(state)
    assert actions
    stats["infosets"].add(game.infoset_key(state, player))
    own = hands[player]
    weights = {"f": 1.0, "c": 2.0, "r": 1.0 + own + 2 * (own in boards[0])}
    total = sum(weights[action] for action in actions)
    for action in actions:
        _walk(game, game.next_state(state, action), reach * weights[action] / total, stats)


def _play(game, *moves):
    """The state reached from the start by these cards and actions, in order."""
    state = game.initial_state()
    for move in moves:
        state = game.next_state(state, move)
    return state


if __name__ == "__main__":
    import time

    # Whole-tree walks for N = 3, sizes worked out by hand. Views (own rank, board ranks): 9 =
    # own x 1 card; 24 = 3 distinct cards; 27 = own x A x B, independent; 90 = 5 distinct cards
    # (rank counts 2, 2, 1); 192 = 3 own ranks x 8 x 8 ordered rank pairs, independent. A street
    # with cap 2 has 6 decision nodes and 9 lines (cc crf crc crrf crrc rf rc rrf rrc: 4 folds,
    # 5 go on); cap 1 has 4 and 5 (2 folds); cap 3 has 8 and 13. Suited deals: 120 = 6 x 5 x 4.
    walks = [  # (parameters, expected infosets, expected terminals)
        (dict(num_boards=1), 9 * 6, 120 * 9),
        (dict(deck_mode="shared"), 24 * 6, 360 * 9),
        (dict(deck_mode="independent"), 27 * 6, 480 * 9),
        (dict(deck_mode="identical"), 9 * 6, 120 * 9),
        (dict(num_boards=1, streets=2), 9 * 6 + 24 * 30, 120 * 4 + 360 * 5 * 9),
        (dict(streets=2), 24 * 6 + 90 * 30, 360 * 4 + 720 * 5 * 9),
        (dict(streets=2, deck_mode="independent"), 27 * 6 + 192 * 30, 480 * 4 + 4320 * 5 * 9),
        (dict(streets=2, deck_mode="identical"), 9 * 6 + 24 * 30, 120 * 4 + 360 * 5 * 9),
        (dict(num_boards=1, streets=2, ante=0.1, bet_sizes=(0.3, 0.7), max_raises=(1, 3)),
         9 * 4 + 24 * 3 * 8, 120 * 2 + 360 * 3 * 13),
    ]
    values = []
    for params, infosets, terminals in walks:
        game, start = ReferenceBoardLeduc(**params), time.time()
        stats = {"terminals": 0, "value": 0.0, "infosets": set()}
        _walk(game, game.initial_state(), 1.0, stats)
        found = (len(stats["infosets"]), stats["terminals"])
        assert found == (infosets, terminals), (params, found)
        values.append(stats["value"])
        print(f"{game.name}: {terminals} terminals, {infosets} infosets, p0 value under "
              f"the test strategy {stats['value']:+.6f} ({time.time() - start:.1f} s)")
    # identical with two boards must play exactly like one board
    assert abs(values[3] - values[0]) < 1e-12 and abs(values[7] - values[4]) < 1e-12

    game = ReferenceBoardLeduc()                    # N = 3, two boards, one street, shared
    J0, J1, Q0, Q1, K0, K1 = game.deck              # ranks 0, 1, 2; suits 0, 1
    four = ReferenceBoardLeduc(num_ranks=4, streets=2)
    two_streets = ((2, 0), (1, 0), (0, 0), (3, 0), "r", "c", (2, 1), (0, 1), "c", "r", "c")
    examples = [  # (game, cards and actions in order, expected returns)
        (game, (K0, J0, Q0, Q1, "r", "c"), (3.0, -3.0)),       # K high beats J high on A and B
        (game, (K0, J0, J1, Q0, "c", "r", "c"), (0.0, 0.0)),   # J pair wins A, K high wins B
        (game, (Q0, Q1, K0, J0, "r", "r", "c"), (0.0, 0.0)),   # same rank: both boards tie
        (game, (K0, J0, Q0, Q1, "c", "r", "f"), (-1.0, 1.0)),  # p0 folds, losing its ante
        (game, (K0, J0, Q0, Q1, "r", "f"), (1.0, -1.0)),       # p1 folds, losing its ante
        (ReferenceBoardLeduc(deck_mode="independent"),         # J1 on both boards (own decks),
         (K0, J0, J1, J1, "r", "c"), (-3.0, 3.0)),             # so the J pair scoops
        (ReferenceBoardLeduc(num_boards=1, streets=2),         # board J J pairs nobody: K wins
         (K0, Q0, J0, "c", "c", J1, "c", "c"), (1.0, -1.0)),
        (four, two_streets, (7.0, -7.0)),   # 3 then 7 each; p0 pairs on A, 2 beats 1 high on B
        (four, two_streets[:4] + ("r", "r", "c", (2, 1), (0, 1), "r", "r", "f"),
         (-9.0, 9.0)),                      # 5 each after street 1; p0 bets to 9 and folds
    ]
    for g, moves, expected in examples:
        state = _play(g, *moves)
        assert g.is_terminal(state) and g.returns(state) == expected, (moves, g.returns(state))
    assert four.describe(_play(four, *two_streets)) == ((2, 1), ((0, 2), (3, 0)), ("rc", "crc"))
    assert four.describe(_play(four, *two_streets[:8])) == ((2, 1), ((0, 2), (3, 0)), ("rc", ""))
    same = ReferenceBoardLeduc(deck_mode="identical")              # board B copies board A
    assert same.describe(_play(same, K0, J0, J1)) == ((2, 0), ((0,), (0,)), ("",))

    for actions, legal in [("", "cr"), ("c", "cr"), ("r", "fcr"), ("cr", "fcr"), ("crr", "fc")]:
        assert game.legal_actions(_play(game, K0, J0, Q0, Q1, *actions)) == tuple(legal)
    capped = ReferenceBoardLeduc(num_boards=1, streets=2, max_raises=(1, 0))  # a cap per street
    assert capped.legal_actions(_play(capped, K0, J0, Q0, "r")) == ("f", "c")
    assert capped.legal_actions(_play(capped, K0, J0, Q0, "r", "c", Q1)) == ("c",)

    key = game.infoset_key(_play(game, K0, J0, Q0, Q1), 0)
    assert key == game.infoset_key(_play(game, K1, J1, Q1, Q0), 0)    # only suits differ
    assert key == game.infoset_key(_play(game, K0, K1, Q0, Q1), 0)    # only p1's card differs
    assert key != game.infoset_key(_play(game, K0, J0, Q0, J1), 0)    # board B differs
    print(f"{len(examples)} hand-worked showdowns and folds, describe, legal actions and "
          "infoset keys: all passed")
