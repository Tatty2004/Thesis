"""The viewer's live mode (experiments/valuenet/live.py, viewer_server --live) on the small deck
(5 ranks x 3 suits, the real no-limit betting), from a two-iteration real_trunk with a tiny
network on the CPU:

- the live turn -> river chain equals turn_resolve's re-solve followed by a safe river
  re-solve (gpu_subgame.resolve_gpu) from the re-solving player's reach under its re-solved
  turn, with the opponent's values from that turn re-solve;
- the 'backup' T values equal the turn-root values of the average turn strategy with the
  trunk's averaged leaf values as the leaf (real_grade.StoredLeaf);
- LeafGadget keep_best ends on the best check exactly (= a fresh run of that many iterations);
- turn_resolve.OnDevice (the mps path) gives NetLeaf's values;
- the server's endpoints: page, meta, progress, stored and live states, errors.
"""
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bombpot.core.solvers.subgame import line_template  # noqa: E402
from bombpot.experiments.valuenet import real_grade as rg  # noqa: E402
from bombpot.experiments.valuenet import turn_resolve as tr  # noqa: E402
from bombpot.experiments.valuenet import viewer_server as vs  # noqa: E402

LIVE = {"cfv": "avg", "iters": 20, "check_every": 5, "keep_best": True}
TOL, SUB_ITERS = 1e-2, 60


@pytest.fixture(scope="module")
def trunk(tmp_path_factory):
    from bombpot.experiments.valuenet import real_trunk
    from bombpot.valuenet.features import FeatureConfig
    from bombpot.valuenet.model import HandSetNet, ModelConfig, save_model

    root = tmp_path_factory.mktemp("live")
    torch.manual_seed(0)
    model = HandSetNet(FeatureConfig("hand_set", 36, num_ranks=5, num_suits=3),
                       ModelConfig(kind="set", depth=1, width=8, heads=0))
    save_model(root / "net.pt", model)
    out = root / "trunk" / "run"
    real_trunk.main(["--model", str(root / "net.pt"), "--flops", "2c3c4d,5c6d4c", "--deck", "5,3", "--iters", "2",
                     "--checks", "2", "--samples", "0", "--defer", "--device", "cpu", "--turn-cfv", "--out", str(out)])
    return out


@pytest.fixture(scope="module")
def viewer(trunk, tmp_path_factory):
    return vs.Viewer(trunk, "cpu", TOL, SUB_ITERS, live=LIVE, cache=tmp_path_factory.mktemp("cache"))


def checked_line(v):
    """(fa, turn line l1, d, turn cards): flop check-check, then a turn deal."""
    tc = v.upper.classes[1][0]
    flop = v.upper.classes[0][0].template
    k1 = flop.children[0][0][1]
    close = flop.children[k1][0][1]
    d = 7
    a, b = (int(x) for x in v.river.turn_cards[d])
    return "0,0", v.turn_line[close], d, vs.card_str(a, 3) + vs.card_str(b, 3), tc


def test_live_chain_equals_turn_resolve_then_river_resolve(viewer, trunk):
    from bombpot.valuenet.gpu_subgame import resolve_gpu

    v = viewer
    fa, l1, d, turn, tc = checked_line(v)
    st = v.state({"fa": fa, "turn": turn, "live": "1"})
    assert st["source"]["kind"] == "live_turn" and st["player"] == 0
    # the turn: turn_resolve's re-solve of the same spot, each player
    spot = tr.Spot(trunk, "cpu")
    c, l = 0, l1
    s = spot.setup(d, c, l, "avg")
    tp = s["cls"].template
    lt = v.live_turn(d, l1)
    gad = {}
    for p in (0, 1):
        g, ev = spot.resolve(s, p, LIVE["iters"], None, LIVE["check_every"], keep_best=True)
        gad[p] = g
        avg = g.average_strategy()[0]
        for k, q in enumerate(tp.players):
            if q == p:
                np.testing.assert_array_equal(lt[f"p{p}_k{k}"], avg[k][0, 0].astype(np.float32))
        np.testing.assert_allclose(lt["stats"][p, 2], ev["exploitability"], rtol=1e-12)
        np.testing.assert_array_equal(lt[f"p{p}_leaf"], g.leaf_average(1 - p)[0][0, 0])
    np.testing.assert_allclose(st["strategy"], np.round(lt["p0_k0"].astype(np.float64), 4), atol=1e-12)
    # check-check on the turn, then a river: each player's safe re-solve from its own turn
    k1 = tp.children[0][0][1]
    close = tp.children[k1][0][1]
    ci, l2, i = v.river_of[(l1, close)]
    rem = list(v.river.remaining[d])
    ra, rb = int(rem[0]), int(rem[1])
    st = v.state({"fa": fa, "turn": turn, "ta": "0,0", "river": vs.card_str(ra, 3) + vs.card_str(rb, 3), "mode": "exact",
                  "live": "1"})
    assert st["source"]["kind"] == "live_river" and st["source"]["turn"]["kind"] == "live_turn"
    sol = v.cache[(d, i, True)]
    rch = [s["reach"][q][0, 0].copy() for q in (0, 1)]
    as32 = lambda x: x.astype(np.float32).astype(np.float64)  # the live turn's strategies are kept in float32
    rch[0] = rch[0] * as32(gad[0].average_strategy()[0][0][0, 0])[:, 0]
    rch[1] = rch[1] * as32(gad[1].average_strategy()[0][k1][0, 0])[:, 0]
    rc = v.upper.classes[2][ci]
    rtp, contrib = line_template(rc.template, l2), float(rc.contrib[l2])
    street = v.river.street(*v.river.all_children([d]))
    street.kappa = np.full(street.num_deals, v.river.scale)
    turn_st = v.upper.streets[1]
    kt, valid = float(turn_st.kappa[d]), np.asarray(turn_st.valid[d], bool)
    r = v.river_child(d, ra, rb)
    for p in (0, 1):
        o = 1 - p
        rp = np.where(valid, rch[p], 0.0)
        m = rp.sum()
        cfv = gad[p].leaf_average(o)[0][0, 0][close]
        ev = resolve_gpu(street, rtp, contrib, rp / m, np.where(valid, cfv, 0.0) / (kt * m), p, iters=SUB_ITERS,
                         tol=2.0 * TOL * 2.0 * contrib, check_every=10, device="cpu", keep_strategy=True)
        assert sol["stats"][p, 0] == ev["iterations"]
        np.testing.assert_allclose(sol["stats"][p, 1], ev["exploitability"], rtol=1e-9)
        for k, q in enumerate(rtp.players):
            if q == p:
                np.testing.assert_allclose(sol[f"k{k}"].astype(np.float64), ev["strategy"][k], atol=1e-3)
    st_r0 = np.asarray(st["strategy"])
    x = sol["k0"][r].astype(np.float64)
    np.testing.assert_allclose(st_r0, np.round(x / x.sum(-1, keepdims=True), 4), atol=1e-12)
    # the stored-trunk river of the same spot is a different solve, with its own cache entry
    st2 = v.state({"fa": fa, "turn": turn, "ta": "0,0", "river": vs.card_str(ra, 3) + vs.card_str(rb, 3), "mode": "exact"})
    assert st2["source"]["kind"] == "river" and st2["source"]["turn"]["kind"] == "trunk"
    assert (d, i, False) in v.cache


@pytest.mark.parametrize("simultaneous", [False, True])
def test_paired_equals_one_after_the_other(trunk, simultaneous):
    from bombpot.experiments.valuenet.live import LiveSolver

    args = dict(LIVE, simultaneous=simultaneous)
    a = LiveSolver(trunk, "cpu", paired=True, **args).turn(9, 0, 3)
    b = LiveSolver(trunk, "cpu", paired=False, **args).turn(9, 0, 3)
    assert a.keys() == b.keys()
    for k in a:
        if k == "stats":  # seconds differ
            np.testing.assert_allclose(a[k][:, :4], b[k][:, :4], rtol=1e-9)
        else:
            np.testing.assert_allclose(a[k], b[k], rtol=1e-6, atol=1e-9)


def test_backup_values_equal_stored_leaf_backup(trunk):
    spot = tr.Spot(trunk, "cpu")
    up = spot.upper
    avg = rg.load_strategy(trunk / "strategy.npz", up)
    from bombpot.core.solvers.range_cfr import forward

    reach = forward(up, avg)
    roots = [r[0] for r in reach[1]]
    leaf = rg.StoredLeaf(up, np.load(trunk / "leafcfv.npy"))
    bp = tr.blueprint_values(up.streets[1], up.classes[1], avg[1], roots, leaf)
    for d, l in ((3, 0), (11, 5)):
        s = spot.setup(d, 0, l, "backup")
        for q in (0, 1):
            np.testing.assert_allclose(s["cfv"][q, 0, 0], bp["values"][q][0][d, l], rtol=1e-10, atol=1e-15)
    assert spot.cfv_source() == "avg"
    # without turncfv.npy auto falls back to backup
    spot.turncfv = None
    assert spot.cfv_source() == "backup"


def test_keep_best_is_the_best_check(trunk):
    spot = tr.Spot(trunk, "cpu")
    s = spot.setup(5, 0, 2, "onpolicy")
    g, ev = spot.resolve(s, 1, 30, None, 3, keep_best=True)
    b = ev["best_iteration"]
    assert ev["exploitability"] == min(x for _, x in ev["history"])
    g2, ev2 = spot.resolve(s, 1, b)
    for x, y in zip(g.average_strategy()[0], g2.average_strategy()[0]):
        np.testing.assert_array_equal(x, y)
    for q in (0, 1):
        np.testing.assert_array_equal(g.leaf_average(q)[0], g2.leaf_average(q)[0])
    np.testing.assert_allclose(ev["exploitability"], ev2["exploitability"], rtol=1e-12)


def test_on_device_wrapper_gives_net_leaf_values(trunk):
    from bombpot.core.solvers.range_cfr import class_reach, stack_closing

    spot = tr.Spot(trunk, "cpu")
    s = spot.setup(4, 0, 1, "avg")
    tp = s["cls"].template
    sig = [np.full(s["reach"].shape[1:] + (len(a),), 1.0 / len(a)) for a in tp.actions]
    _, closing = class_reach(tp, sig, (s["reach"][0], s["reach"][1]))
    a = s["leaf"].values_both(1, stack_closing([closing]))
    spot.model = tr.OnDevice(spot.model, "cpu")
    b = spot.leaf(4, 0, [1]).values_both(1, stack_closing([closing]))
    for q in (0, 1):
        np.testing.assert_allclose(a[q][0], b[q][0], rtol=1e-6, atol=1e-12)


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=600) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_endpoints(viewer):
    server = vs.make_server(viewer, 0)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        code, body = get(base, "/")
        assert code == 200 and b"Live re-solve" in body
        code, body = get(base, "/api/meta")
        meta = json.loads(body)
        assert code == 200 and meta["suits"] == "cdh" and len(meta["hands"]) == viewer.H and "live" in meta
        code, body = get(base, "/api/progress")
        assert code == 200 and json.loads(body)["active"] is False
        code, body = get(base, "/api/state?fa=")
        st = json.loads(body)
        assert code == 200 and st["source"]["kind"] == "trunk" and st["street"] == 0
        fa, l1, d, turn, _ = checked_line(viewer)
        code, body = get(base, f"/api/state?fa={fa}&turn={turn}")
        st = json.loads(body)
        assert code == 200 and st["source"]["kind"] == "trunk" and st["street"] == 1
        np.testing.assert_allclose(st["strategy"], np.round(viewer.row(1, 0, d, l1), 4), atol=1e-12)
        code, body = get(base, f"/api/state?fa={fa}&turn={turn}&live=1")
        st = json.loads(body)
        assert code == 200 and st["source"]["kind"] == "live_turn"
        assert len(st["source"]["gap_pots"]) == 2 and all(g >= 0 for g in st["source"]["gap_pots"])
        assert viewer.live_turn_path(d, l1).exists()  # cached on disk
        code, body = get(base, f"/api/state?fa={fa}&turn={turn}&ta=0,0&live=1")
        st = json.loads(body)
        assert code == 200 and st["need"] == "river" and st["source"]["kind"] == "live_turn"
        rem = list(viewer.river.remaining[d])
        river = vs.card_str(int(rem[2]), 3) + vs.card_str(int(rem[3]), 3)
        code, body = get(base, f"/api/state?fa={fa}&turn={turn}&ta=0,0&river={river}&mode=quick&live=1")
        st = json.loads(body)
        assert code == 200 and st["source"]["kind"] == "live_river_quick" and st["resolve"]["live"] is True
        code, body = get(base, f"/api/state?fa={fa}&turn=AhAh")
        assert code == 400 and "error" in json.loads(body)
        assert json.loads(get(base, "/api/meta")[1])["live"]["tag"] == viewer.live.tag
    finally:
        server.shutdown()
        server.server_close()
