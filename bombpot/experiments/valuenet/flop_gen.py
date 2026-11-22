"""Flop-end labels for the noisy-vs-exact flop-net test (2026-11-22,
experiments/valuenet/PLAN_flopnet_test.md): one HDF5 shard per job of flop-end states (random
flop pair, flop-end line and ranges, as flop_datagen.sample_state) with their per-hand
counterfactual values in pots (the label).

A label solves the turn below the state over K turn deals (simultaneous DCFR, the turn-end
network at every turn close: flop_datagen.solve_state) and estimates the sum over all D turn
deals (flop_datagen's module docstring):
  --deals all   every deal: exact up to the iteration error
  --deals K     K uniform random deals with the check-down control variate
                (sum_D f_d + D / K sum_S (v_d - f_d), f = flop_datagen.proxy_values)
The control variate needs the turn-end equities at every deal (0.2-0.4 CPU-s each, ~2070 per
flop pair), so a pool of --workers processes computes the next flop pair's while the GPU
solves the current one, and --per-pair states share a flop pair.

Flop pairs come from one side of the turn-end data's split (data.heldout_mask, salt
real-multi-0, fraction 0.05: `--side train` or `heldout`). The G.4 test pairs and the three
validation pairs (real/val_pairs3.txt) are never drawn, on either side.

Shard (N states, H hands; data.py reads it with FeatureConfig board_cards 3, hand_extra 2):
  flops (N, 2, 3), hand_set (N,), hands (sets, H, 2)     card ids
  pot, stack, line (N,)                                  chips, chips behind each, flop-end line
  range0, range1 (N, H)                                  normalized ranges
  equity, equity_sd (N, H, 2)                            flop-end equity (mean turn-end equity
                                                         over the turn deals the hand can see)
                                                         and its spread, board A and B
  possible (N, H)                                        all True at the flop's end
  cfv0, cfv1 (N, H)                                      the label, in pots
  deals, iters, exploitability, seconds (N,)             K, iterations, the turn subgame's
                                                         exploitability vs the net (pots), GPU
                                                         seconds of the solve (+ proxy)
  with --deals all also: it<c>cfv0/1 (the label at every earlier check) and, for each K in
  --synth, k<K>cfv0/1: the control-variate label from one random K of the solved deals (the
  cheap label type on the same state, for free)
  attrs: the arguments, git, GPU, host, Slurm ids

Stops starting states when the next would end past --minutes (the per-state time measured so
far, or --first-seconds for the first), or on SIGUSR1 (sbatch --signal), keeping every finished
state.

    python -m bombpot.experiments.valuenet.flop_gen --model .../best.pt --side train --seed 0 \
        --deals 50 --iters 100 --per-pair 8 --minutes 52 --out /tmp/x.h5
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SALT, FRACTION = "real-multi-0", 0.05
NEVER = ("Td5c2s,As9h6c", "AsKh4d,KdQdTd", "KhJd9c,7h4c2c",  # G.4 test pairs
         "Kc7h7d,JsTc9c", "Qc8c3s,Tc6c4c", "AcJh6h,AdKhTh")  # validation pairs (real/val_pairs3.txt)


def card_str(flops) -> str:
    from bombpot.experiments.valuenet.flop_datagen import RANKS, SUITS

    return ",".join("".join(RANKS[c // 4] + SUITS[c % 4] for c in sorted(f, reverse=True)) for f in flops)


def never_keys() -> set:
    from bombpot.experiments.valuenet.flop_datagen import parse_flops
    from bombpot.valuenet.data import flop_pair_key

    return {flop_pair_key(*[list(f) for f in parse_flops(p, 13, 4)]) for p in NEVER}


def draw_pairs(rng, n: int, side: str) -> list:
    """n random flop pairs (tuples of sorted card ids, as parse_flops) from `side` of the split."""
    from bombpot.valuenet.data import flop_pair_key, heldout_mask

    bad, out = never_keys(), []
    while len(out) < n:
        cards = rng.permutation(52)
        flops = (tuple(sorted(int(c) for c in cards[:3])), tuple(sorted(int(c) for c in cards[3:6])))
        held = bool(heldout_mask(np.array([flops]), FRACTION, SALT)[0])
        if held != (side == "heldout") or flop_pair_key(*[list(f) for f in flops]) in bad:
            continue
        out.append(flops)
    return out


# Equity workers (spawned, one thread each) ------------------------------------------------

_FE: dict = {}


def _equity(flops, deals) -> np.ndarray:
    from bombpot.experiments.valuenet.flop_datagen import FlopEnd

    if flops not in _FE:
        _FE.clear()
        _FE[flops] = FlopEnd(flops)
    return _FE[flops].equity(np.asarray(deals))


class EquityPool:
    """All D turn-end equities of a flop pair, computed by `workers` processes in chunks."""

    def __init__(self, workers: int):
        import multiprocessing as mp

        saved = {k: os.environ.get(k) for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        os.environ.update(dict.fromkeys(saved, "1"))  # the children read these at import
        self.pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k)
            else:
                os.environ[k] = v
        self.workers = workers

    def submit(self, flops, D: int) -> list:
        chunks = np.array_split(np.arange(D), 4 * self.workers)
        return [self.pool.submit(_equity, flops, c) for c in chunks]

    @staticmethod
    def gather(futures) -> np.ndarray:
        return np.concatenate([f.result() for f in futures])

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        for p in list(getattr(self.pool, "_processes", {}).values()):
            p.terminate()


# Shard ------------------------------------------------------------------------------------


class Shard:
    """Resizable datasets made on the first row (per-hand arrays gzip'd in blocks of 64 rows)."""

    def __init__(self, path, attrs: dict):
        import h5py

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, "w")
        self.f.attrs.update(attrs)
        self.n, self.sets, self._set_of = 0, 0, {}

    def hand_set(self, flops, hands: np.ndarray) -> int:
        if flops not in self._set_of:
            if "hands" not in self.f:
                self.f.create_dataset("hands", (0,) + hands.shape, "i1", maxshape=(None,) + hands.shape,
                                      chunks=(16,) + hands.shape)
            ds = self.f["hands"]
            ds.resize(self.sets + 1, axis=0)
            ds[self.sets] = hands
            self._set_of[flops] = self.sets
            self.sets += 1
        return self._set_of[flops]

    def add(self, row: dict) -> None:
        for k, v in row.items():
            v = np.asarray(v)
            if k not in self.f:
                big = v.ndim >= 1 and v.size > 64
                self.f.create_dataset(k, (0,) + v.shape, v.dtype, maxshape=(None,) + v.shape,
                                      chunks=((64 if big else 256),) + v.shape,
                                      **({"compression": "gzip", "compression_opts": 4} if big else {}))
            ds = self.f[k]
            ds.resize(self.n + 1, axis=0)
            ds[self.n] = v
        self.n += 1
        self.f.flush()

    def close(self) -> None:
        self.f.attrs["n"] = self.n
        self.f.close()


# Labels -----------------------------------------------------------------------------------


def cv_label(f: np.ndarray, v: np.ndarray, S: np.ndarray, per_pair: int) -> np.ndarray:
    """The control-variate label (2, H) from the proxy f (2, D, H) and the per-deal values v
    (2, K, H) at the deals S (both in pots x per_pair, flop_datagen's per-deal scale)."""
    D, K = f.shape[1], len(S)
    return (f.sum(1) + (v - f[:, S]).sum(1) * (D / K)) / per_pair


def flop_equity(fe) -> tuple:
    """(H, 2) flop-end equity and its spread over the turn deals the hand can be held at."""
    eq = fe.equity_all.astype(np.float64)
    w = fe.turn.valid[..., None].astype(np.float64)
    n = np.maximum(w.sum(0), 1)
    m = (w * eq).sum(0) / n
    sd = np.sqrt(np.maximum((w * eq ** 2).sum(0) / n - m ** 2, 0))
    return m.astype(np.float32), sd.astype(np.float32)


class Stop(Exception):
    pass


_SIG = {"writing": False, "stop": False}


def _on_signal(signum, frame):
    _SIG["stop"] = True
    if not _SIG["writing"]:
        raise Stop


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--side", choices=("train", "heldout"), required=True)
    ap.add_argument("--seed", type=int, required=True, help="flop pairs, states and deal subsets")
    ap.add_argument("--deals", default="50", help="'all' or K")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--checks", default="", help="--deals all: earlier checks kept as it<c>cfv (e.g. 100)")
    ap.add_argument("--synth", default="", help="--deals all: K values for free k<K>cfv labels (e.g. 50,400)")
    ap.add_argument("--alternating", action="store_true", help="alternating updates (default simultaneous)")
    ap.add_argument("--per-pair", type=int, default=1, help="states per flop pair")
    ap.add_argument("--max-states", type=int, default=100000)
    ap.add_argument("--minutes", type=float, default=50.0, help="stop starting states that would end after this")
    ap.add_argument("--first-seconds", type=float, default=60.0, help="a priori seconds of one state (A5000)")
    ap.add_argument("--workers", type=int, default=3, help="equity processes")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    start = time.perf_counter()
    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    import torch

    from bombpot.experiments.valuenet import datagen as dg
    from bombpot.experiments.valuenet import flop_datagen as fd
    from bombpot.valuenet.model import load_model

    rng = np.random.default_rng([a.seed, 0 if a.side == "train" else 1])
    pool = EquityPool(a.workers)
    pairs = [draw_pairs(rng, 1, a.side)[0]]
    pending = pool.submit(pairs[0], 2070)  # D = 46 x 45 turn deals on 52 cards
    model = load_model(a.model, "cuda")
    gpu = torch.cuda.get_device_name(0)
    slow = 1.65 if ("2080" in gpu or "RTX 6000" in gpu or "TITAN" in gpu) else 1.0  # Turing vs the A5000 prior
    checks = sorted({int(c) for c in a.checks.split(",") if c} | {a.iters})
    synth = [int(k) for k in a.synth.split(",") if k]
    attrs = {"args": json.dumps(vars(a)), "git": dg.git_hash(), "gpu": gpu, "host": socket.gethostname(),
             "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "slurm_array": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
             "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID", ""), "salt": SALT, "fraction": FRACTION,
             "side": a.side, "units_cfv": "pots (root value / pot)"}
    shard = Shard(a.out, attrs)
    print(f"seed {a.seed} side {a.side} deals {a.deals} iters {a.iters} per-pair {a.per_pair} on {gpu} "
          f"({socket.gethostname()}), {a.workers} equity workers", flush=True)
    budget, est, done, waited = 60 * a.minutes, a.first_seconds * slow, 0, 0.0
    try:
        for i in range(100000):
            flops = pairs[i]
            t0 = time.perf_counter()
            eq = pool.gather(pending)
            waited += time.perf_counter() - t0
            pairs.append(draw_pairs(rng, 1, a.side)[0])
            pending = pool.submit(pairs[i + 1], 2070)
            fe = fd.FlopEnd(flops, 13, 4, eq)
            strength, (feq, fsd) = fe.strength(), flop_equity(fe)
            hs = shard.hand_set(flops, fe.hands.astype(np.int8))
            for j in range(a.per_pair):
                now = time.perf_counter() - start
                if done >= a.max_states or now + 1.15 * est > budget:
                    raise Stop
                t1 = time.perf_counter()
                state = fd.sample_state(fe, rng, strength)
                K = fe.D if a.deals == "all" else int(a.deals)
                deals = np.arange(fe.D) if K == fe.D else np.sort(rng.choice(fe.D, K, replace=False))
                rec = fd.solve_state(fe, model, state, deals, a.iters, checks, amp=not a.no_amp,
                                     simultaneous=not a.alternating, per_deal=K < fe.D or bool(synth),
                                     log=lambda m: None)
                row = {"flops": np.array(flops, np.int8), "hand_set": hs, "pot": state["pot"],
                       "stack": state["stack"], "line": state["line"], "range0": state["range0"].astype(np.float32),
                       "range1": state["range1"].astype(np.float32), "equity": feq, "equity_sd": fsd,
                       "possible": np.ones(fe.H, bool), "deals": K, "iters": a.iters,
                       "exploitability": rec["exploitability"][-1]}
                if K < fe.D or synth:
                    f, _ = fd.proxy_values(fe, model, state, np.arange(fe.D), amp=not a.no_amp)
                if K < fe.D:  # deal_cfv is over `deals`, in order
                    label = cv_label(f, rec["deal_cfv"][-1].astype(np.float64), deals, fe.per_pair)
                else:
                    label = rec["label_cfv"][-1]
                    for c, lab in zip(rec["checks"][:-1], rec["label_cfv"][:-1]):
                        row[f"it{c}cfv0"], row[f"it{c}cfv1"] = lab[0], lab[1]
                    for k in synth:
                        S = np.sort(rng.choice(fe.D, k, replace=False))
                        lab = cv_label(f, rec["deal_cfv"][-1][:, S].astype(np.float64), S, fe.per_pair)
                        row[f"k{k}cfv0"], row[f"k{k}cfv1"] = lab[0].astype(np.float32), lab[1].astype(np.float32)
                row["cfv0"], row["cfv1"] = (np.asarray(x, np.float32) for x in label)
                sec = time.perf_counter() - t1
                row["seconds"] = sec
                _SIG["writing"] = True
                shard.add(row)
                _SIG["writing"] = False
                done += 1
                est = sec if done == 1 else max(est, sec)  # the slowest state so far
                print(f"state {done}: {card_str(flops)} line {state['line']} pot {state['pot']:g} K {K} "
                      f"{sec:.1f}s exploitability {rec['exploitability'][-1]:.2e} | {time.perf_counter() - start:.0f}s "
                      f"elapsed, {waited:.0f}s waiting for equities", flush=True)
                if _SIG["stop"]:
                    raise Stop
    except Stop:
        print(f"stopping after {done} states ({time.perf_counter() - start:.0f}s)", flush=True)
    finally:
        _SIG["writing"] = True
        shard.f.attrs["equity_wait_seconds"] = waited
        shard.f.attrs["seconds"] = time.perf_counter() - start
        shard.close()
        pool.close()
    print(f"DONE {done} states", flush=True)


if __name__ == "__main__":
    main()
