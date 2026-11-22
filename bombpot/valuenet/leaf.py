"""NetLeaf: the value network as the turn-end value provider of range_cfr.backward
(PLAN_valuenet.md, section C):

    leaf.values(p, t, closing) -> agg

closing = (reach0, reach1), each (D, L, closes, H): both players' reach at every turn deal d,
entering line l and closing line c. agg is (D, L, closes, H): player p's counterfactual value
of each hand in the solver's units (chance weight x opponent reach x payoff, as a showdown
terminal). With line classes, `closing` is a list of such tuples (one per class), pot_by_line
and stack_by_line are lists of per-class arrays, and the result is a list.

For every (d, l, c):
  1. Mask both reaches with possible_by_deal[d] (hands that can be held at that deal) and
     normalize each to sum 1. A reach with no mass becomes uniform over the possible hands:
     the network still needs a valid input range, and the result is multiplied by the
     opponent's mass anyway (0 when the opponent can't be here; player p's own mass doesn't
     enter, so p's values stay defined at hands it never plays, as CFR needs).
  2. Histogram both normalized ranges over deal d's equity bins (equity_by_deal[d], (H, 2)).
  3. Run the network (all (d, l, c) in one batch, split into chunks of `batch_size`) and map
     player p's per-bin output (pots) back to hands.
  4. Rescale:  agg[d, l, c, i] = cfv[i] x pot[l, c] x sum_j reach_opp[d, l, c, j] x kappa[d],
     the sum over the opponent's possible hands, unnormalized. Hands impossible at d get 0.
Counterfactual values are linear in the opponent's reach, which is why step 4 is exact for
an exact network. kappa_by_deal[d] is the chance weight per hand pair at deal d (the caller's
W[d, i, j] for compatible hands); the leaf returns equilibrium values in every mode.

A network with per-hand inputs (model.HandSetNet, features.method "hand_set") needs the hand
list too (`hands`): every hand is its own bin, step 2 is the identity, and the network also
gets features.hand_features of each deal (its possible flag taken from possible_by_deal) and
the hands' cards. Its batches are at most `model.chunk` samples.

The rows can be shared out over several devices (`devices`), a copy of the network on each.

The caller (experiments/) supplies the equities, boards and possible hands from the game's
code; this layer never imports a game.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from bombpot.valuenet.features import bin_index, encode_boards, hand_cards, hand_features, pot_features


class NetLeaf:

    def __init__(self, model, equity_by_deal, pot_by_line, stack_by_line, kappa_by_deal, device="cpu", *,
                 boards_by_deal=None, possible_by_deal=None, batch_size: int = 32768, hands=None,
                 devices=None, amp: bool = False, block: int = 8192):
        """model: a ValueNet (anything with .features and forward(board, scalars, h0, h1) ->
        (v0, v1) per bin, in pots). equity_by_deal (D, H, 2). pot_by_line, stack_by_line
        (L, closes) chips (pot, and chips behind each) at each turn-closing line, or a list of
        such per line class. kappa_by_deal (D,). boards_by_deal (D, 2, k) card ids, required
        unless the model's board encoding is "none". possible_by_deal (D, H) bool, default all
        hands (then the caller must give impossible hands zero reach). hands (H, 2) card ids,
        required by a network with per-hand inputs.

        devices: several devices (default [device]); each gets a copy of the network and a
        contiguous share of the rows, on its own thread. amp: mixed precision for the network on
        CUDA (everything around it stays float64). block: rows moved to a device at a time (the
        reaches stay on the host otherwise)."""
        devs = [torch.device(x) for x in (devices or [device])]
        self.cfg, self.device, self.batch_size = model.features, devs[0], batch_size
        self.amp, self.block = amp, block
        params = list(model.parameters())
        self.dtype = params[0].dtype if params else torch.float32  # the network's inputs
        equity = np.asarray(equity_by_deal, dtype=np.float64)
        D, H, _ = equity.shape
        self.D, self.H = D, H
        bins = bin_index(equity, self.cfg)
        possible = np.ones((D, H), bool) if possible_by_deal is None else np.asarray(possible_by_deal, bool)
        if self.cfg.board == "none":
            boards_by_deal = np.zeros((D, 2, self.cfg.board_cards), np.int64)
        elif boards_by_deal is None:
            raise ValueError("boards_by_deal is required for board encoding " + repr(self.cfg.board))
        board = encode_boards(np.asarray(boards_by_deal), self.cfg)
        self.kappa = np.asarray(kappa_by_deal, dtype=np.float64)
        hx = hc = None
        if getattr(model, "per_hand", False):
            if hands is None:
                raise ValueError("a network with per-hand inputs needs hands and boards_by_deal")
            hx = hand_features(equity, hands, np.asarray(boards_by_deal), self.cfg)
            hx[..., 2] = possible
            hc = hand_cards(hands)
            self.batch_size = min(batch_size, getattr(model, "chunk", batch_size))
        self.replicas = []
        for k, dev in enumerate(devs):
            rep = {"device": dev, "model": (model if k == 0 else copy.deepcopy(model)).to(dev).eval(),
                   "bins": torch.as_tensor(bins, device=dev), "possible": torch.as_tensor(possible, device=dev),
                   "board": torch.as_tensor(board, device=dev), "kappa": torch.as_tensor(self.kappa, device=dev),
                   "hx": None if hx is None else torch.as_tensor(hx, device=dev),
                   "hc": None if hc is None else torch.as_tensor(hc, device=dev)}
            self.replicas.append(rep)
        self.model = self.replicas[0]["model"]
        self.pot, self.stack = pot_by_line, stack_by_line

    def values(self, p: int, t: int, closing):
        if isinstance(closing, list):
            return [self._values((p,), c, np.asarray(pot, float), np.asarray(st, float))[0]
                    for c, pot, st in zip(closing, self.pot, self.stack)]
        return self._values((p,), closing, np.asarray(self.pot, float), np.asarray(self.stack, float))[0]

    def values_both(self, t: int, closing):
        """(values(0, t, closing), values(1, t, closing)) from one network pass per row."""
        if isinstance(closing, list):
            per = [self._values((0, 1), c, np.asarray(pot, float), np.asarray(st, float))
                   for c, pot, st in zip(closing, self.pot, self.stack)]
            return [x[0] for x in per], [x[1] for x in per]
        return tuple(self._values((0, 1), closing, np.asarray(self.pot, float), np.asarray(self.stack, float)))

    def _values(self, ps, closing, pot: np.ndarray, stack: np.ndarray) -> list:
        r0, r1 = (np.asarray(r, dtype=np.float64) for r in closing)
        D, L, C, H = r0.shape
        assert (D, H) == (self.D, self.H) and pot.shape == stack.shape == (L, C), "shapes disagree"
        rows = D * L * C
        flat = [np.ascontiguousarray(r.reshape(rows, H)) for r in (r0, r1)]
        scal = pot_features(pot, stack).reshape(L * C, -1)
        outs = [np.empty((rows, H)) for _ in ps]
        K = len(self.replicas)
        cuts = [rows * k // K for k in range(K + 1)]
        if K == 1:
            self._rows(self.replicas[0], ps, flat, scal, pot.reshape(-1), outs, 0, rows, L * C)
        else:
            with ThreadPoolExecutor(K) as pool:
                jobs = [pool.submit(self._rows, rep, ps, flat, scal, pot.reshape(-1), outs, cuts[k], cuts[k + 1], L * C)
                        for k, rep in enumerate(self.replicas)]
                for j in jobs:
                    j.result()
        return [out.reshape(D, L, C, H) for out in outs]

    def _rows(self, rep, ps, flat, scal_np, pot_np, outs, lo: int, hi: int, LC: int) -> None:
        """Rows [lo, hi) of `outs[i]` (player ps[i]'s values) on one replica."""
        dev, model, B = rep["device"], rep["model"], self.cfg.num_bins
        H = self.H
        with torch.no_grad():
            scal = torch.as_tensor(scal_np, device=dev)
            potv = torch.as_tensor(pot_np, device=dev)
            for A in range(lo, hi, self.block):
                Z = min(A + self.block, hi)
                blk = [torch.from_numpy(r[A:Z]).to(dev) for r in flat]
                res = [torch.empty(Z - A, H, dtype=torch.float64, device=dev) for _ in ps]
                for a in range(A, Z, self.batch_size):
                    n = torch.arange(a, min(a + self.batch_size, Z), device=dev)
                    d, lc = n // LC, n % LC
                    poss = rep["possible"][d]
                    bins = rep["bins"][d]
                    hists, mass = [], []
                    for r in blk:
                        x = torch.where(poss, r[n - A], 0.0)
                        m = x.sum(-1, keepdim=True)
                        uni = poss.double() / poss.double().sum(-1, keepdim=True).clamp_min(1.0)
                        x = torch.where(m > 0, x / torch.where(m > 0, m, 1.0), uni)
                        h = torch.zeros(len(n), B, dtype=torch.float64, device=dev).scatter_add_(1, bins, x)
                        hists.append(h.to(self.dtype))
                        mass.append(m[:, 0])
                    board = rep["board"][d] if self.cfg.board == "ids" else rep["board"][d].to(self.dtype)
                    hand = {} if rep["hx"] is None else {"hx": rep["hx"][d], "hc": rep["hc"].expand(len(n), H, 2)}
                    with torch.autocast("cuda", enabled=self.amp and dev.type == "cuda"):
                        both = model(board, scal[lc].to(self.dtype), hists[0], hists[1], **hand)
                    for i, p in enumerate(ps):
                        cfv = torch.gather(both[p].double(), 1, bins)
                        scale = potv[lc] * mass[1 - p] * rep["kappa"][d]
                        res[i][a - A:a - A + len(n)] = torch.where(poss, cfv * scale[:, None], 0.0)
                for out, r in zip(outs, res):
                    out[A:Z] = r.cpu().numpy()
