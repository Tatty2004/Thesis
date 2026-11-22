"""A local strategy viewer for a real-deck value-net solve (real_trunk's output directory).

    python -m bombpot.experiments.valuenet.viewer_server --trunk <dir> [--device mps|cuda|cpu]

then open http://localhost:8765. The flop and the turn are read from the stored strategy
(strategy.npz). A river is not stored: the first visit to a (turn deal, river line) re-solves it
for both players exactly as the grade does (real_grade.resolve_deal: the safe re-solve gadget
from the player's reach under the stored strategy, the opponent held to its averaged leaf values,
leafcfv.npy), over all 1,892 river deals at once; the result is kept in <trunk>/river_cache so a
second visit is instant.

Listens on localhost only. Needs torch.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from bombpot.core.solvers.subgame import line_template
from bombpot.experiments.valuenet import real_grade as rg

RANKS, SUITS = "23456789TJQKA", "cdhs"
TOKEN = re.compile(r"r[\d.]+|c|f|a")


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def parse_card(s: str) -> int:
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def npz_memmaps(path: Path) -> dict:
    """The arrays of an uncompressed .npz, memory-mapped in place."""
    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as f:
        for info in z.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{path} is compressed")
            f.seek(info.header_offset)
            h = f.read(30)
            start = info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little")
            f.seek(start)
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                     else np.lib.format.read_array_header_2_0)(f)
            out[info.filename[:-4]] = np.memmap(path, dtype=dtype, mode="r", offset=f.tell(), shape=shape,
                                                order="F" if fortran else "C")
    return out


def chips(hist: str, contrib: float, stack: float) -> list:
    """Chips each player has added on this street after the betting string `hist` (the engine's
    pot-fraction sizes: a raise calls, then adds the fraction of the pot after the call)."""
    inn, p = [0.0, 0.0], 0
    for a in TOKEN.findall(hist):
        if a == "c":
            inn[p] = max(inn)
        elif a == "a":
            inn[p] = stack
        elif a[0] == "r":
            call = max(inn)
            inn[p] = min(call + float(a[1:]) * (2 * contrib + 2 * call), stack)
        p = 1 - p
    return inn


def label(a: str, hist: str, contrib: float, stack: float) -> str:
    inn = chips(hist, contrib, stack)
    p = len(TOKEN.findall(hist)) % 2
    facing = max(inn) - inn[p]
    if a == "f":
        return "Fold"
    if a == "c":
        return f"Call {facing:g}" if facing > 0 else "Check"
    to = chips(hist + a, contrib, stack)[p]
    kind = "All-in" if a == "a" or to >= stack else ("Raise" if max(inn) > 0 else "Bet")
    return f"{kind} {to - inn[p]:g}" if kind == "Bet" else f"{kind} to {to:g}"


class Viewer:
    def __init__(self, trunk: Path, device: str, tol: float, iters: int, keep: int = 6):
        self.trunk, self.device, self.tol, self.iters, self.keep = trunk, device, tol, iters, keep
        self.info = json.loads((trunk / "trunk.json").read_text())
        self.game, self.upper, self.river = rg.build(trunk)
        up = self.upper
        self.hands = np.array(up.hands)  # (H, 2) cards
        self.H = len(self.hands)
        self.sigma = npz_memmaps(trunk / "strategy.npz")
        cfv = trunk / "leafcfv32.npy"
        self.cfv = np.load(cfv if cfv.exists() else trunk / "leafcfv.npy", mmap_mode="r")
        self.flops = [[parse_card(f[i:i + 2]) for i in (0, 2, 4)] for f in self.info["args"]["flops"].split(",")]
        self.turn_of = {(int(a), int(b)): d for d, (a, b) in enumerate(self.river.turn_cards)}
        # (turn line, close) -> (river class, line in class, index among all river lines)
        self.river_of, i = {}, 0
        for c, rc in enumerate(up.classes[2]):
            for l in range(rc.num_lines):
                self.river_of[(int(rc.parent_line[l]), int(rc.parent_close[l]))] = (c, l, i)
                i += 1
        tc = up.classes[1][0]
        self.turn_line = {int(j): l for l, j in enumerate(tc.parent_close)}
        self.cache, self.sums, self.quick = {}, {}, {}
        self.progress = {"active": False}
        self.lock = threading.Lock()   # exact re-solves
        self.qlock = threading.Lock()  # quick solves: never wait behind an exact one
        (trunk / "river_cache").mkdir(exist_ok=True)
        grade = trunk / "grade.json"
        self.grade = json.loads(grade.read_text()) if grade.exists() else None

    # The stored streets --------------------------------------------------------------------

    def row(self, t: int, k: int, d: int, l: int) -> np.ndarray:
        """(H, A) the stored strategy at node k, as the grade reads it (real_grade.rounded)."""
        x = np.asarray(self.sigma[f"t{t}_c0_k{k}"][d, l], dtype=np.float64)
        return x / x.sum(-1, keepdims=True)

    def possible(self, public) -> np.ndarray:
        return ~np.isin(self.hands, list(public)).any(axis=1)

    # The river -----------------------------------------------------------------------------

    def river_child(self, d: int, ra: int, rb: int) -> int:
        rem = list(self.river.remaining[d])
        i, j = rem.index(ra), rem.index(rb)
        return i * (self.river.left - 1) + j - (j > i)

    def solve_river(self, d: int, c: int, l: int, i: int, reach) -> dict:
        """Both players' re-solves of river line i of turn deal d: {"k<node>": (R, H, A) float16
        for the node's acting player, "stats": (2, 3) iterations, gap / mass, seconds}."""
        key = (d, i)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            path = self.trunk / "river_cache" / f"d{d:04d}_l{i:03d}.npz"
            if path.exists():
                with np.load(path) as f:
                    out = {k: f[k] for k in f.files}
            else:
                out = self._solve(d, c, l, i, reach)
                tmp = path.with_suffix(".tmp.npz")
                np.savez(tmp, **out)
                tmp.replace(path)
            self.cache[key] = out
            while len(self.cache) > self.keep:
                self.cache.pop(next(iter(self.cache)))
            return out

    def _solve(self, d: int, c: int, l: int, i: int, reach) -> dict:
        import torch

        from bombpot.valuenet.gpu_subgame import GpuGadget, StreetSums

        up, river = self.upper, self.river
        if d not in self.sums:
            self.sums.clear()
            st = river.street(*river.all_children([d]))
            st.kappa = np.full(st.num_deals, river.scale)
            self.sums[d] = StreetSums(st, self.device, torch.float32)
        sums = self.sums[d]
        turn = up.streets[1]
        kt, valid = float(turn.kappa[d]), np.asarray(turn.valid[d], bool)
        rc = up.classes[2][c]
        tp, contrib = line_template(rc.template, l), float(rc.contrib[l])
        out, stats = {}, np.zeros((2, 3))
        t0 = time.perf_counter()
        self.progress = {"active": True, "part": 0, "parts": 2, "fraction": 0.0, "iterations": 0, "seconds": 0.0,
                         "text": "building the river's card tables"}
        for p in (0, 1):
            o = 1 - p
            rp = np.where(valid, reach[p], 0.0)
            m = float(rp.sum())
            if m <= 0:
                continue
            g = GpuGadget(sums, tp, contrib, rp / m, np.where(valid, self.cfv[o, d, i], 0.0) / (kt * m), p,
                          dtype=torch.float32, device=self.device)
            ev = self._run(g, 2.0 * self.tol * 2.0 * contrib, p, 2, t0)
            stats[p] = (ev["iterations"], ev["exploitability"], g.seconds)
            for k, s in enumerate(g.average_strategy()):
                if tp.players[k] == p:
                    out[f"k{k}"] = s.cpu().numpy().astype(np.float16)
            del g
        out["stats"] = stats
        self.progress = {"active": False}
        return out

    def _run(self, g, tol: float, part: int, parts: int, t0: float) -> dict:
        """g.solve(self.iters, tol, 10), reporting progress: how far the gap has come from its
        first value to the tolerance, on a log scale."""
        first, ev = None, None
        frac = 0.0
        while g.t < self.iters:
            for _ in range(5):  # the grade's check every 10 iterations, reported every 2
                g.iterate(2)
                self.progress = {"active": True, "part": part, "parts": parts,
                                 "fraction": float(frac if first is not None else 0.04 * g.t),
                                 "iterations": g.t, "seconds": time.perf_counter() - t0,
                                 "text": f"P{part}'s solve: iteration {g.t}"}
            ev = g.evaluate()
            gap = max(float(ev["exploitability"]), 1e-30)
            first = gap if first is None else first
            frac = 1.0 if gap <= tol else (0.0 if first <= tol else
                                           min(1.0, max(0.0, np.log(first / gap) / np.log(first / tol))))
            self.progress = {"active": True, "part": part, "parts": parts, "fraction": float(frac),
                             "iterations": g.t, "seconds": time.perf_counter() - t0,
                             "text": f"P{part}'s solve: iteration {g.t}, gap {gap:.3g} chips, target {tol:.3g}"}
            if gap <= tol:
                break
        ev["iterations"] = g.t
        return ev

    def solve_quick(self, d: int, c: int, l: int, i: int, r: int, reach) -> dict:
        """One river deal alone, solved from both players' ranges (no gadget, so without the
        re-solve's guarantee; not what the grade measures). {"k<node>": (1, H, A)} and stats."""
        import torch

        from bombpot.valuenet.gpu_subgame import GpuSubgame

        key = (d, i, r)
        with self.qlock:
            if key in self.quick:
                return self.quick[key]
            up, river = self.upper, self.river
            st = river.street(np.array([d]), np.array([r]))
            st.kappa = np.full(1, river.scale)
            valid = np.asarray(up.streets[1].valid[d], bool)
            rc = up.classes[2][c]
            tp, contrib = line_template(rc.template, l), float(rc.contrib[l])
            roots = [np.where(valid, x, 0.0) for x in reach]
            out = {"stats": np.zeros((2, 3))}
            if roots[0].sum() > 0 and roots[1].sum() > 0:
                g = GpuSubgame(st, tp, contrib, roots[0] / roots[0].sum(), roots[1] / roots[1].sum(),
                               dtype=torch.float32, device="cpu")
                ev = g.solve(self.iters, self.tol * 2.0 * contrib, 10)
                out["stats"][:] = (ev["iterations"], ev["exploitability"], g.seconds / 2)
                for k, x in enumerate(g.average_strategy()):
                    out[f"k{k}"] = x.cpu().numpy()
            self.quick[key] = out
            while len(self.quick) > 200:
                self.quick.pop(next(iter(self.quick)))
            return out

    # One public state ----------------------------------------------------------------------

    def state(self, q: dict) -> dict:
        """The node reached by q: flop actions `fa`, turn cards `turn`, turn actions `ta`, river
        cards `river`, river actions `ra` (actions as indices, cards as 'AsKd' = board A then B)."""
        up = self.upper
        acts = lambda s: [int(x) for x in s.split(",") if x != ""]
        fa, ta, ra = acts(q.get("fa", "")), acts(q.get("ta", "")), acts(q.get("ra", ""))
        reach = [np.ones(self.H), np.ones(self.H)]
        public = set(self.flops[0] + self.flops[1])
        boards = [list(self.flops[0]), list(self.flops[1])]
        path = []
        out = {"boards": None, "path": path, "need": None, "terminal": None}

        def done(**kw):
            ok = self.possible(public)
            out.update(boards=[[card_str(x) for x in b] for b in boards],
                       reach=[np.round(np.where(ok, r, 0.0), 5).tolist() for r in reach],
                       possible=ok.astype(int).tolist(), **kw)
            return out

        def walk(t, tp, contrib, stack, actions, strat):
            """Follow `actions` from node 0. Returns ("node", k) | ("fold", player) | ("close", j)."""
            k = 0
            for a in actions:
                if not 0 <= a < len(tp.actions[k]):
                    raise ValueError(f"no action {a} at {tp.hist[k]!r}")
                p = tp.players[k]
                reach[p] = reach[p] * strat(k)[:, a]
                path.append({"street": t, "player": p, "label": label(tp.actions[k][a], tp.hist[k], contrib, stack)})
                ch = tp.children[k][a]
                if ch[0] != "node":
                    return ch[0], (ch[1] if ch[0] == "fold" else ch[1]), tp.hist[k] + tp.actions[k][a]
                k = ch[1]
            return "node", k, tp.hist[k]

        def node(t, tp, k, contrib, stack, strat, extra=None):
            inn = chips(tp.hist[k], contrib, stack)
            return done(street=t, player=tp.players[k], contrib=contrib, stack=stack, added=inn,
                        pot=2 * contrib + sum(inn),
                        actions=[label(a, tp.hist[k], contrib, stack) for a in tp.actions[k]],
                        strategy=np.round(strat(k), 4).tolist(), **(extra or {}))

        # Flop
        lc = up.classes[0][0]
        tp, contrib, stack = lc.template, float(lc.contrib[0]), float(lc.stack[0])
        kind, x, hist = walk(0, tp, contrib, stack, fa, lambda k: self.row(0, k, 0, 0))
        if kind == "node":
            return node(0, tp, x, contrib, stack, lambda k: self.row(0, k, 0, 0))
        if kind == "fold":
            return done(terminal=f"P{x} folds", pot=2 * contrib + sum(chips(hist, contrib, stack)))
        # Turn
        l1 = self.turn_line[x]
        lc = up.classes[1][0]
        tp, contrib, stack = lc.template, float(lc.contrib[l1]), float(lc.stack[l1])
        if not q.get("turn"):
            return done(need="turn", pot=2 * contrib, street=1)
        ca, cb = parse_card(q["turn"][:2]), parse_card(q["turn"][2:4])
        if (ca, cb) not in self.turn_of:
            raise ValueError("those turn cards are not available")
        d = self.turn_of[(ca, cb)]
        boards[0].append(ca), boards[1].append(cb)
        public |= {ca, cb}
        strat = lambda k: self.row(1, k, d, l1)
        kind, x, hist = walk(1, tp, contrib, stack, ta, strat)
        if kind == "node":
            return node(1, tp, x, contrib, stack, strat)
        if kind == "fold":
            return done(terminal=f"P{x} folds", pot=2 * contrib + sum(chips(hist, contrib, stack)))
        # River
        if (l1, x) not in self.river_of:
            return done(terminal="All in: the river is dealt and the hands are shown", pot=2 * contrib + sum(chips(hist, contrib, stack)))
        c, l2, i = self.river_of[(l1, x)]
        rc = up.classes[2][c]
        tp, contrib, stack = rc.template, float(rc.contrib[l2]), float(rc.stack[l2])
        if not q.get("river"):
            return done(need="river", pot=2 * contrib, street=2,
                        solved=(d, i) in self.cache or (self.trunk / "river_cache" / f"d{d:04d}_l{i:03d}.npz").exists())
        ra_, rb_ = parse_card(q["river"][:2]), parse_card(q["river"][2:4])
        if ra_ == rb_ or {ra_, rb_} & public:
            raise ValueError("those river cards are not available")
        t0 = time.perf_counter()
        r = self.river_child(d, ra_, rb_)
        quick = q.get("mode") == "quick"
        if quick:
            sol = self.solve_quick(d, c, l2, i, r, [x.copy() for x in reach])
            r = 0
        else:
            sol = self.solve_river(d, c, l2, i, [x.copy() for x in reach])
        boards[0].append(ra_), boards[1].append(rb_)
        public |= {ra_, rb_}

        def strat(k):
            if f"k{k}" not in sol:  # the acting player never reaches this river line
                return np.full((self.H, len(tp.actions[k])), 1.0 / len(tp.actions[k]))
            x = sol[f"k{k}"][r].astype(np.float64)
            return x / np.maximum(x.sum(-1, keepdims=True), 1e-30)

        extra = {"resolve": {"iterations": sol["stats"][:, 0].tolist(), "gap": sol["stats"][:, 1].tolist(),
                             "seconds": sol["stats"][:, 2].tolist(), "waited": time.perf_counter() - t0,
                             "mode": "quick" if quick else "exact"}}
        kind, x, hist = walk(2, tp, contrib, stack, ra, strat)
        if kind == "node":
            return node(2, tp, x, contrib, stack, strat, extra)
        pot = 2 * contrib + sum(chips(hist, contrib, stack))
        return done(terminal=f"P{x} folds" if kind == "fold" else "Showdown", pot=pot, **extra)

    def meta(self) -> dict:
        checks = self.info.get("checks", {})
        last = checks[max(checks, key=int)] if checks else None
        return {"flops": [[card_str(c) for c in f] for f in self.flops],
                "hands": [[card_str(a), card_str(b)] for a, b in self.hands],
                "trunk": self.trunk.name, "iterations": self.info["args"]["iters"],
                "flop_turn_exploitability": last["flop_turn"]["exploitability"] if last else None,
                "grade": self.grade, "device": self.device, "tol": self.tol}


def serve(viewer: Viewer, port: int) -> None:
    page = (Path(__file__).parent / "viewer_page.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self.send(200, page, "text/html; charset=utf-8")
                if u.path == "/api/meta":
                    return self.send(200, json.dumps(viewer.meta()).encode(), "application/json")
                if u.path == "/api/progress":
                    return self.send(200, json.dumps(viewer.progress).encode(), "application/json")
                if u.path == "/api/state":
                    return self.send(200, json.dumps(viewer.state(q)).encode(), "application/json")
                self.send(404, b"not found", "text/plain")
            except Exception as e:  # shown in the page
                self.send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

        def log_message(self, *a):
            pass

    print(f"viewer at http://localhost:{port}  (Ctrl-C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    ap.add_argument("--tol", type=float, default=1e-2, help="re-solve tolerance, pots (the grade's)")
    ap.add_argument("--sub-iters", type=int, default=5000)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    if a.device is None:
        import torch
        a.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.perf_counter()
    viewer = Viewer(Path(a.trunk), a.device, a.tol, a.sub_iters)
    print(f"loaded {a.trunk} in {time.perf_counter() - t0:.0f}s; river re-solves on {a.device}", flush=True)
    serve(viewer, a.port)


if __name__ == "__main__":
    main()
