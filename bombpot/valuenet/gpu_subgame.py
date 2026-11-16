"""A last-street subgame solved on a GPU (or on the CPU through torch): the algorithm of
core/solvers/fused.py (alternating CFR per deal, the same discounts, the same averaging weight,
the same stopping rule), written as whole-array torch operations over (deal, hand).

What differs from the compiled kernel is only how the fold and showdown sums are added up.
The kernel sweeps each deal's hands in strength order and keeps running totals per card. Here
the same sums come from prefix sums:

- fold:      sum over the opponent hands j that share no card with i of x[j]
             = all - (hands with i's first card) - (hands with i's second card) + x[i]
- showdown:  the same inclusion-exclusion on "weaker than i" and "tied with i" on each board.
             "Weaker, any cards" is a prefix sum of x in strength order, read at the start of
             i's tie group. "Weaker, sharing card c" is a prefix sum over the hands that hold c,
             sorted by strength, read at the start of i's tie group within that list. The lists
             of all cards sit side by side in one array of 2H entries (each hand is in two
             lists), so one cumsum per board serves every hand.

The sums have the same terms as the kernel's but another order of addition, so in float64 the
two solvers agree to rounding (about 1e-12 relative per iteration), not bit for bit; see
tests/valuenet/test_gpu_subgame.py. float32 is the fast setting on a GPU.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from bombpot.core.solvers.fused import CLOSE, FOLD, NODE, _compact_ids, _discounts, compile_template
from bombpot.core.solvers.range_cfr import VARIANTS
from bombpot.core.solvers.subgame import Subgame


def sweep_index(street) -> dict:
    """The index arrays of a CardStreet's fold and showdown sums (numpy, int64):

    order     (D, B, H)     hands in strength order (the street's)
    gs, ge    (D, B, H)     per hand: start and end of its tie group in `order`
    pair_hand (D, B, P)     P = 2H: the hands holding card 0 sorted by strength, then card 1, ...
    cgs, cge  (D, B, 2H)    per hand and card slot: start and end of its tie group in `pair_hand`
    seg       (2H,)         per hand and card slot: where that card's list starts in `pair_hand`
    cards     (H, C) float  1 where the hand holds the card
    """
    h = street.hands
    H = h.subs.shape[0]
    if h.subs.shape[1] != 4 or not np.array_equal(np.asarray(h.signs, dtype=float), [1.0, -1.0, -1.0, 1.0]):
        raise ValueError("two-card hands only (subsets: empty, each card, the hand)")
    mid, C = _compact_ids(h.subs[:, 1:-1])  # (H, 2) the hand's cards, numbered 0..C-1
    order = np.asarray(street.order, dtype=np.int64)
    group_end = np.asarray(street.group_end, dtype=np.int64)
    D, B, _ = order.shape
    pos = np.arange(H)
    start = np.ones((D, B, H), bool)  # a tie group starts where the one before ends
    start[..., 1:] = group_end[..., :-1] == pos[1:]
    gid_pos = np.cumsum(start, axis=2) - 1
    gs_pos = np.maximum.accumulate(np.where(start, pos, 0), axis=2)
    where = np.empty_like(order)  # each hand's position in the order
    np.put_along_axis(where, order, np.broadcast_to(pos, order.shape), axis=2)
    gs = np.take_along_axis(gs_pos, where, axis=2)
    ge = np.take_along_axis(group_end, where, axis=2)
    gid = np.take_along_axis(gid_pos, where, axis=2)  # (D, B, H) each hand's tie group

    counts = np.bincount(mid.ravel(), minlength=C)
    n = int(counts[0])
    if (counts != n).any():
        raise ValueError("every card must be in the same number of hands")
    flat = np.argsort(mid.ravel(), kind="stable")  # (hand, slot) pairs grouped by card
    lh, slot = (flat // 2).reshape(C, n), (flat % 2).reshape(C, n)
    g = gid[:, :, lh]  # (D, B, C, n)
    perm = np.argsort(g, axis=3, kind="stable")
    g = np.take_along_axis(g, perm, axis=3)
    lhb, slb = np.broadcast_to(lh, g.shape), np.broadcast_to(slot, g.shape)
    pair_hand = np.take_along_axis(lhb, perm, axis=3)
    pair_slot = np.take_along_axis(slb, perm, axis=3)
    u = np.arange(n)
    first = np.ones(g.shape, bool)
    first[..., 1:] = g[..., 1:] != g[..., :-1]
    last = np.ones(g.shape, bool)
    last[..., :-1] = first[..., 1:]
    rs = np.maximum.accumulate(np.where(first, u, 0), axis=3)
    re = np.minimum.accumulate(np.where(last, u + 1, n)[..., ::-1], axis=3)[..., ::-1]
    seg0 = (np.arange(C) * n)[:, None]
    cgs = np.empty((D, B, H, 2), np.int64)
    cge = np.empty((D, B, H, 2), np.int64)
    dd, bb = np.arange(D)[:, None, None, None], np.arange(B)[None, :, None, None]
    cgs[dd, bb, pair_hand, pair_slot] = seg0 + rs
    cge[dd, bb, pair_hand, pair_slot] = seg0 + re
    cards = np.zeros((H, C))
    cards[np.arange(H)[:, None], mid] = 1.0
    return {"order": order, "gs": gs, "ge": ge, "pair_hand": pair_hand.reshape(D, B, C * n),
            "cgs": cgs.reshape(D, B, 2 * H), "cge": cge.reshape(D, B, 2 * H), "seg": (mid * n).reshape(2 * H),
            "cards": cards, "valid": np.asarray(street.valid, bool), "kappa": np.asarray(street.kappa, dtype=float)}


class StreetSums:
    """The fold and showdown sums of one street on a device, for batches x (T, D, H) of
    opponent reaches: core/removal's `fold` and `beats` (with the street's chance weight)."""

    def __init__(self, street, device="cuda", dtype=torch.float32, index: dict | None = None):
        ix = sweep_index(street) if index is None else index
        self.device, self.dtype = torch.device(device), dtype
        for k in ("order", "gs", "ge", "pair_hand", "cgs", "cge", "seg"):
            setattr(self, k, torch.as_tensor(ix[k], device=self.device))
        self.cards = torch.as_tensor(ix["cards"], device=self.device, dtype=dtype)
        self.valid = torch.as_tensor(ix["valid"], device=self.device)
        self.validf = self.valid.to(dtype)
        self.kv = torch.as_tensor(ix["kappa"], device=self.device, dtype=dtype)[:, None] * self.validf  # (D, H)
        self.D, self.B, self.H = self.order.shape

    @staticmethod
    def _prefix(x):
        """(..., n) -> (..., n + 1): sums of the first 0..n entries."""
        return torch.nn.functional.pad(torch.cumsum(x, dim=-1), (1, 0))

    def fold(self, x):
        """(T, D, H) reaches -> (T, D, H): kappa x the reach of the hands sharing no card with i."""
        x = x * self.validf
        tot = x @ self.cards  # (T, D, C) reach per card
        return (x.sum(-1, keepdim=True) - tot @ self.cards.T + x) * self.kv

    def beats(self, x):
        """(T, D, H) reaches -> (T, D, H): kappa x the mean over boards of (the reach i beats +
        half the reach i ties with), among the hands sharing no card with i."""
        x = x * self.validf
        T = x.shape[0]
        out = torch.zeros_like(x)

        def take(a, idx):
            return torch.gather(a, 2, idx.expand(T, -1, -1))

        for b in range(self.B):
            cs = self._prefix(take(x, self.order[:, b]))
            w_all = take(cs, self.gs[:, b])
            t_all = take(cs, self.ge[:, b]) - w_all
            cp = self._prefix(take(x, self.pair_hand[:, b]))
            lo = take(cp, self.cgs[:, b])
            w_card = (lo - cp[:, :, self.seg]).view(T, self.D, self.H, 2).sum(-1)
            t_card = (take(cp, self.cge[:, b]) - lo).view(T, self.D, self.H, 2).sum(-1)
            out += (w_all - w_card) + 0.5 * (t_all - t_card + x)
        return out * (self.kv / self.B)


class GpuSubgame:
    """One one-line subgame on a street, solved per deal (FusedSubgames with S = 1).

    street    a CardStreet (its kappa are the chance weights), or a StreetSums of it
    template  a one-line Template (line_template(class template, l))
    contrib   chips each player has in at the root
    reach0/1  (H,) root ranges
    """

    def __init__(self, street, template, contrib, reach0, reach1, variant: str = "dcfr", alpha: float = 1.5,
                 beta: float = 0.0, gamma: float = 2.0, dtype=torch.float32, device="cuda"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.sums = street if isinstance(street, StreetSums) else StreetSums(street, device, dtype)
        self.device, self.dtype = self.sums.device, self.sums.dtype
        self.variant, self.alpha, self.beta, self.gamma = variant, alpha, beta, gamma
        self.player, self.nact, self.kind, self.target, self.coef = compile_template(template, contrib)
        self.K = len(self.player)
        D, H = self.sums.D, self.sums.H
        self.D, self.H = D, H
        self.roots = (np.asarray(reach0, dtype=float), np.asarray(reach1, dtype=float))
        self.root = [torch.as_tensor(r, device=self.device, dtype=self.dtype).expand(D, H) for r in self.roots]
        z = lambda k: torch.zeros((D, H, int(self.nact[k])), device=self.device, dtype=self.dtype)
        self.regret = [z(k) for k in range(self.K)]
        self.cum = [z(k) for k in range(self.K)]
        self.sigma = [torch.full_like(self.regret[k], 1.0 / int(self.nact[k])) for k in range(self.K)]
        self.flush = -1e-30 if dtype == torch.float32 else None  # as FusedSubgames (denormals)
        self.t = 0
        self.seconds = 0.0

    # One pass -------------------------------------------------------------------------------

    def _forward(self, sigma, root):
        """reach[q][k] (D, H) at every node from the root reaches (range_cfr.class_reach)."""
        reach = [[None] * self.K, [None] * self.K]
        reach[0][0], reach[1][0] = root
        for k in range(self.K):
            q = int(self.player[k])
            for a in range(int(self.nact[k])):
                if self.kind[k, a] == NODE:
                    j = int(self.target[k, a])
                    reach[q][j] = reach[q][k] * sigma[k][:, :, a]
                    reach[1 - q][j] = reach[1 - q][k]
        return reach

    def _terminal_values(self, p, sigma, reach):
        """{(k, a): (D, H)} player p's value of every fold and showdown child against the
        opponent's reach there. All the sums of the pass go through the device in two calls."""
        o = 1 - p
        xs, slot_of, need_b = [], {}, []
        for k in range(self.K):
            q = int(self.player[k])
            for a in range(int(self.nact[k])):
                kd = self.kind[k, a]
                if kd == NODE:
                    continue
                key = (k, -1) if q == p else (k, a)  # at p's own node the opponent's reach is the node's
                if key not in slot_of:
                    slot_of[key] = len(xs)
                    xs.append(reach[o][k] if q == p else reach[o][k] * sigma[k][:, :, a])
                    need_b.append(False)
                if kd == CLOSE:
                    need_b[slot_of[key]] = True
        if not xs:
            return {}
        x = torch.stack(xs)
        f = self.sums.fold(x)
        rows = [i for i, nb in enumerate(need_b) if nb]
        bt = self.sums.beats(x[rows]) if rows else None
        row_of = {i: r for r, i in enumerate(rows)}
        out = {}
        for k in range(self.K):
            q = int(self.player[k])
            for a in range(int(self.nact[k])):
                kd = self.kind[k, a]
                if kd == NODE:
                    continue
                i = slot_of[(k, -1) if q == p else (k, a)]
                c = float(self.coef[p, k, a])
                if kd == FOLD:
                    out[(k, a)] = c * f[i]
                elif p == 0:
                    out[(k, a)] = c * (2.0 * bt[row_of[i]] - f[i])
                else:
                    out[(k, a)] = c * (2.0 * (f[i] - bt[row_of[i]]) - f[i])
        return out

    def _children(self, k, V, tv):
        return [V[int(self.target[k, a])] if self.kind[k, a] == NODE else tv[(k, a)] for a in range(int(self.nact[k]))]

    def iterate(self, n: int) -> None:
        """n alternating iterations (fused._iterate_kernel)."""
        start = time.perf_counter()
        W = _discounts(self.variant, self.t + 1 + np.arange(n), self.alpha, self.beta, self.gamma)
        ones = torch.ones((self.D, self.H), device=self.device, dtype=self.dtype)
        mode = VARIANTS[self.variant]
        for it in range(n):
            w0, ps, ng = float(W[0][it]), float(W[1][it]), float(W[2][it])
            for p in range(2):
                reach = self._forward(self.sigma, self.root)
                within = self._forward(self.sigma, (ones, ones))[p]  # the averaging weight (SubgameCFR.own_reach)
                tv = self._terminal_values(p, self.sigma, reach)
                V = [None] * self.K
                inst = {}
                for k in range(self.K - 1, -1, -1):
                    cv = torch.stack(self._children(k, V, tv), dim=-1)  # (D, H, A)
                    if int(self.player[k]) != p:
                        V[k] = cv.sum(-1)
                    else:
                        V[k] = (self.sigma[k] * cv).sum(-1)
                        inst[k] = cv - V[k][:, :, None]
                for k, d in inst.items():
                    self.cum[k] += (w0 * within[k])[:, :, None] * self.sigma[k]
                    r = self.regret[k] + d
                    if mode == 1:
                        r = r.clamp_(min=0.0)
                    elif mode == 2:
                        r = torch.where(r > 0, r * ps, r * ng)
                        if self.flush is not None:
                            r = torch.where((r < 0) & (r > self.flush), torch.zeros_like(r), r)
                    self.regret[k] = r
                    rp = r.clamp(min=0.0)
                    tot = rp.sum(-1, keepdim=True)
                    self.sigma[k] = torch.where(tot > 0, rp / torch.where(tot > 0, tot, torch.ones_like(tot)),
                                                torch.full_like(rp, 1.0 / rp.shape[-1]))
        self.t += n
        self._sync()
        self.seconds += time.perf_counter() - start

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def average_strategy(self) -> list:
        out = []
        for c in self.cum:
            tot = c.sum(-1, keepdim=True)
            out.append(torch.where(tot > 0, c / torch.where(tot > 0, tot, torch.ones_like(tot)),
                                   torch.full_like(c, 1.0 / c.shape[-1])))
        return out

    def mass(self) -> float:
        if not hasattr(self, "_mass"):
            f = self.sums.fold(self.root[1][None])[0].double().sum(0)  # (H,)
            self._mass = float((f.cpu().numpy() * self.roots[0]).sum())
        return self._mass

    def evaluate(self) -> dict:
        """Root values of the average strategy and of a best response to it, summed over deals
        (fused._evaluate_kernel and FusedSubgames.evaluate)."""
        start = time.perf_counter()
        avg = self.average_strategy()
        reach = self._forward(avg, self.root)
        values, brs = [], []
        for p in range(2):
            tv = self._terminal_values(p, avg, reach)
            Vc, Vb = [None] * self.K, [None] * self.K
            for k in range(self.K - 1, -1, -1):
                cc = torch.stack(self._children(k, Vc, tv), dim=-1)
                cb = torch.stack(self._children(k, Vb, tv), dim=-1)
                if int(self.player[k]) == p:
                    Vc[k] = (avg[k] * cc).sum(-1)
                    Vb[k] = cb.max(-1).values
                else:
                    Vc[k], Vb[k] = cc.sum(-1), cb.sum(-1)
            values.append(Vc[0].double().sum(0).cpu().numpy())
            brs.append(Vb[0].double().sum(0).cpu().numpy())
        raw = float(((brs[0] * self.roots[0]).sum() + (brs[1] * self.roots[1]).sum()) / 2)
        mass = self.mass()
        self._sync()
        self.seconds += time.perf_counter() - start
        return {"values": values, "br_values": brs, "exploitability_raw": raw, "mass": mass,
                "exploitability": raw / mass if mass > 0 else 0.0}

    def solve(self, iters: int, tol: float | None = None, check_every: int = 10) -> dict:
        """Iterate until the exploitability is <= tol (checked every `check_every` iterations)
        or `iters` iterations (FusedSubgames.solve)."""
        ev = None
        while self.t < iters:
            n = iters - self.t if tol is None else min(check_every - self.t % check_every, iters - self.t)
            self.iterate(n)
            ev = self.evaluate()
            if tol is not None and ev["exploitability"] <= tol:
                break
        if ev is None:
            ev = self.evaluate()
        ev["iterations"] = self.t
        return ev


def solve_subgame_gpu(street, template, contrib, reach0, reach1, variant: str = "dcfr", iters: int = 1000,
                      tol: float | None = None, check_every: int = 10, dtype=torch.float32, device="cuda") -> Subgame:
    """fused.solve_subgame_fused on a device (strategy left empty)."""
    g = GpuSubgame(street, template, contrib, reach0, reach1, variant, dtype=dtype, device=device)
    ev = g.solve(iters, tol, check_every)
    return Subgame(ev["values"], ev["br_values"], [], ev["iterations"], ev["exploitability"],
                   ev["exploitability_raw"], ev["mass"], g.seconds, None)


# Safe re-solving (core/solvers/resolve.py's gadget) -------------------------------------------


class GpuGadget(GpuSubgame):
    """The re-solving gadget of one one-line subgame (resolve._solve with `cfv`): player p
    re-solves from its root reach; the opponent o chooses per hand between F (enter, with reach
    equal to its F probability) and T (take cfv[h]). The gadget's own numbers (2 per hand) are
    kept in float64.

    reach_p (H,) p's root reach; cfv (H,) o's T values, in the units of this street's root values
    (chance weight x p's reach x payoff, summed over the deals), as resolve's opponent_cfv.

    The average strategies weigh each iteration by the acting player's own reach from the root
    of the gadget game: o's by its F probability times its reach within the subgame (as
    resolve._solve), p's by its reach within the subgame (its root reach is the same in every
    iteration, so that is resolve's average wherever p's root reach is positive, and p's
    strategy stays defined for the hands it doesn't hold)."""

    def __init__(self, street, template, contrib, reach_p, cfv, player: int, variant: str = "dcfr",
                 alpha: float = 1.5, beta: float = 0.0, gamma: float = 2.0, dtype=torch.float32, device="cuda"):
        H = len(reach_p)
        r = [np.ones(H), np.ones(H)]
        r[player] = np.asarray(reach_p, dtype=float)
        super().__init__(street, template, contrib, r[0], r[1], variant, alpha, beta, gamma, dtype, device)
        self.p, self.o = player, 1 - player
        f64 = torch.float64
        self.cfv = torch.as_tensor(np.asarray(cfv, dtype=float), device=self.device, dtype=f64)
        self.rp = torch.as_tensor(r[player], device=self.device, dtype=f64)
        self.g_regret = torch.zeros((H, 2), device=self.device, dtype=f64)
        self.g_cum = torch.zeros((H, 2), device=self.device, dtype=f64)
        self.g_sig = torch.full((H, 2), 0.5, device=self.device, dtype=f64)

    def _root(self, follow):
        """Both root reaches with o's = `follow` (H,), its F probabilities."""
        root = [None, None]
        root[self.p] = self.root[self.p]
        root[self.o] = follow.to(self.dtype).expand(self.D, self.H)
        return root

    def iterate(self, n: int) -> None:
        start = time.perf_counter()
        W = _discounts(self.variant, self.t + 1 + np.arange(n), self.alpha, self.beta, self.gamma)
        ones = torch.ones((self.D, self.H), device=self.device, dtype=self.dtype)
        mode = VARIANTS[self.variant]

        def discounted(r, ps, ng, flush):
            if mode == 1:
                return r.clamp_(min=0.0)
            if mode == 2:
                r = torch.where(r > 0, r * ps, r * ng)
                if flush is not None:
                    r = torch.where((r < 0) & (r > flush), torch.zeros_like(r), r)
            return r

        def matching(r):
            rp = r.clamp(min=0.0)
            tot = rp.sum(-1, keepdim=True)
            return torch.where(tot > 0, rp / torch.where(tot > 0, tot, torch.ones_like(tot)),
                               torch.full_like(rp, 1.0 / rp.shape[-1]))

        for it in range(n):
            w0, ps, ng = float(W[0][it]), float(W[1][it]), float(W[2][it])
            for q in range(2):
                follow = self.g_sig[:, 0]
                reach = self._forward(self.sigma, self._root(follow))
                within = self._forward(self.sigma, (ones, ones))[q]
                own = follow.to(self.dtype)[None, :] if q == self.o else None
                tv = self._terminal_values(q, self.sigma, reach)
                V = [None] * self.K
                inst = {}
                for k in range(self.K - 1, -1, -1):
                    cv = torch.stack(self._children(k, V, tv), dim=-1)
                    if int(self.player[k]) != q:
                        V[k] = cv.sum(-1)
                    else:
                        V[k] = (self.sigma[k] * cv).sum(-1)
                        inst[k] = cv - V[k][:, :, None]
                for k, d in inst.items():
                    wk = within[k] if own is None else within[k] * own
                    self.cum[k] += (w0 * wk)[:, :, None] * self.sigma[k]
                    self.regret[k] = discounted(self.regret[k] + d, ps, ng, self.flush)
                    self.sigma[k] = matching(self.regret[k])
                if q == self.o:  # the gadget node: F pays o its root value, T pays cfv
                    w = V[0].double().sum(0)
                    value = self.g_sig[:, 0] * w + self.g_sig[:, 1] * self.cfv
                    self.g_cum += w0 * self.g_sig
                    self.g_regret = discounted(self.g_regret + torch.stack([w - value, self.cfv - value], -1), ps, ng, None)
                    self.g_sig = matching(self.g_regret)
        self.t += n
        self._sync()
        self.seconds += time.perf_counter() - start

    def gadget_average(self):
        tot = self.g_cum.sum(-1, keepdim=True)
        return torch.where(tot > 0, self.g_cum / torch.where(tot > 0, tot, torch.ones_like(tot)),
                           torch.full_like(self.g_cum, 0.5))

    def mass(self) -> float:
        if not hasattr(self, "_mass"):
            self._mass = float(self.sums.fold(self.root[self.p][None])[0].double().sum())
        return self._mass

    def evaluate(self) -> dict:
        """The gadget game's NashConv for the average profile (resolve._solve's evaluate):
        br_values[o] is o's best-response root value against p's average strategy, what the
        street above sees when p plays it."""
        start = time.perf_counter()
        avg = self.average_strategy()
        g_avg = self.gadget_average()
        reach = self._forward(avg, self._root(g_avg[:, 0]))
        brs = []
        for q in range(2):
            tv = self._terminal_values(q, avg, reach)
            Vb = [None] * self.K
            for k in range(self.K - 1, -1, -1):
                cb = torch.stack(self._children(k, Vb, tv), dim=-1)
                Vb[k] = cb.max(-1).values if int(self.player[k]) == q else cb.sum(-1)
            brs.append(Vb[0].double().sum(0))
        best_o = torch.maximum(brs[self.o], self.cfv).sum()
        best_p = (self.rp * brs[self.p]).sum() - (g_avg[:, 1] * self.cfv).sum()
        gap = float(best_o + best_p)
        mass = self.mass()
        self._sync()
        self.seconds += time.perf_counter() - start
        return {"br_values": [b.cpu().numpy() for b in brs], "gap": gap, "mass": mass,
                "exploitability": gap / mass if mass > 0 else 0.0, "gadget": g_avg.cpu().numpy()}


def resolve_gpu(street, template, contrib, reach_p, cfv, player: int, variant: str = "dcfr", iters: int = 1000,
                tol: float | None = None, check_every: int = 10, dtype=torch.float32, device="cuda",
                keep_strategy: bool = False) -> dict:
    """resolve.resolve (safe) for one one-line subgame on a device. Returns GpuGadget.evaluate's
    dict with `iterations`, `seconds` and, with keep_strategy, `strategy` ([k] (D, H, A) numpy:
    use p's nodes). `tol` bounds gap / mass (the gadget's NashConv over its mass, chips), as
    resolve's."""
    g = GpuGadget(street, template, contrib, reach_p, cfv, player, variant, dtype=dtype, device=device)
    ev = g.solve(iters, tol, check_every)
    ev["seconds"] = g.seconds
    if keep_strategy:
        ev["strategy"] = [s.double().cpu().numpy() for s in g.average_strategy()]
    return ev
