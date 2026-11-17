"""Synthetic checks of player_net.py (no real data needed: fake matches in a temporary folder).

    python pipeline/test_player_net.py        (or: python -m pytest pipeline/test_player_net.py)
"""

import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import player_net as pn  # noqa: E402
import decoder as dec  # noqa: E402

N_FRAMES, NH, NA, EXTRA = 3000, 5, 5, 2


def fake_match(seed, n=N_FRAMES):
    """A match in convert_dfl's format whose possession can be read from the geometry: the team in
    possession pushes forward (towards the goal it attacks) and one of its players has the ball.
    Home attacks +x in the first half, -x in the second. Two never-used columns (substitutes)."""
    rng = np.random.default_rng(seed)
    P = NH + NA + EXTRA
    half = np.r_[np.ones(n // 2, np.int8), np.full(n - n // 2, 2, np.int8)]
    alive = np.ones(n, np.int8)
    alive[1000:1100] = 0
    alive[2200:2260] = 0
    poss = np.empty(n, "<U1")
    owner_team, t_next = "H", 0
    for i in range(n):
        if i == n // 2:
            owner_team, t_next = "A", i + int(rng.integers(150, 260))
        if i >= t_next and i != n // 2:
            owner_team = "A" if owner_team == "H" else "H"
            t_next = i + int(rng.integers(150, 260))
        poss[i] = owner_team
    poss[alive == 0] = ""
    team = np.array(["H"] * NH + ["A"] * NA + ["H", "A"])
    players = np.full((n, P, 2), np.nan, np.float32)
    phase = rng.uniform(0, 2 * np.pi, (P, 2))
    ys = np.linspace(0.15, 0.85, NH)
    ball = np.zeros((n, 3), np.float32)
    push = 0.0
    for i in range(n):
        hdir = 1.0 if half[i] == 1 else -1.0                     # home attacks +x in half 1
        target = (0.12 if poss[i] == "H" else -0.12 if poss[i] == "A" else 0.0)
        push += (target - push) * 0.05                            # the block moves smoothly
        tt = i / 25.0
        for k in range(NH + NA):
            side = 1.0 if k < NH else -1.0
            base_x = 0.5 - side * 0.15 * hdir + push * hdir
            wob = 0.02 * np.sin(0.3 * tt + phase[k])
            players[i, k] = [base_x + wob[0], ys[k % NH] + wob[1]]
        if poss[i] in ("H", "A"):
            ks = list(range(NH)) if poss[i] == "H" else list(range(NH, NH + NA))
            k = ks[(i // 50) % len(ks)]
            ball[i, :2] = players[i, k] + 0.004
        else:
            ball[i, :2] = [0.5, 0.5]
    return dict(frame=np.arange(n), half=half, players=players, team=team,
                keeper=np.zeros(P, bool), referees=np.full((n, 1, 2), np.nan, np.float32),
                ball=ball, possession=poss, alive=alive)


class Env:
    """Temporary folders for converted matches, outputs and models (never data/)."""
    KEYS = ("DFL_CONVERTED", "DFL_OUT", "PLAYER_NET_MODELS")

    def __init__(self, n_matches=3):
        self.tmp = tempfile.mkdtemp()
        self.dirs = {k: os.path.join(self.tmp, k.lower()) for k in self.KEYS[:3]}
        for p in self.dirs.values():
            os.makedirs(p)
        for m in range(n_matches):
            np.savez(os.path.join(self.dirs["DFL_CONVERTED"], "DFL-MAT-FAKE{}.npz".format(m)), **fake_match(m))

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.KEYS}
        os.environ.update(self.dirs)
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_inputs_match_decoder_simulate():
    """Same simulated match as decoder.simulate: same ball gaps and same nearest player."""
    d = fake_match(0)
    inp = pn.degraded_inputs(d, 0)
    ev = dec.simulate(d, 0)
    assert (np.isnan(inp["ours"][:, 0]) == ev["dead"]).all()
    assert (pn.chain_nearest(inp) == ev["near"]).all()
    assert np.isnan(inp["ours"][:, 0]).mean() > 0.02, "the ball has gaps"
    assert np.isnan(inp["metrica"]).all() and not inp["dead"].any()


def test_attack_sign():
    d = fake_match(1)
    s = pn.attack_sign(d["players"], d["team"], d["half"])
    assert (s[d["half"] == 1] == 1).all() and (s[d["half"] == 2] == -1).all()
    sw = np.where(d["team"] == "H", "A", "H")      # labels swapped: directions flip
    s2 = pn.attack_sign(d["players"], sw, d["half"])
    assert (s2 == -s).all()
    # 2nd half graded from before its kick-off, players still on the wrong side: read at the kick-off
    p = d["players"].copy()
    h2 = np.flatnonzero(d["half"] == 2)
    early = h2[:100]
    p[early, :, 0] = 1 - p[early, :, 0]
    assert (pn.attack_sign(p, d["team"], d["half"])[h2] == 1).all()          # fooled without it
    s3 = pn.attack_sign(p, d["team"], d["half"], starts=[0, h2[100]])
    assert (s3[d["half"] == 1] == 1).all() and (s3[h2] == -1).all()


def test_select_columns():
    pres = np.zeros((10, 4), bool)
    pres[:, 2] = True
    pres[3:6, 0] = True
    cols = pn.select_columns(pres, np.array([4, 8]), np.array([0, 7]), np.array([10, 10]), p_max=3)
    assert cols[0].tolist() == [2, 0, -1] and cols[1].tolist() == [2, -1, -1]


def test_param_counts():
    counts = {ch: pn.n_params(dict(pn.DEFAULTS, channels=ch)) for ch in pn.GRID_CH}
    print("  parametri:", counts, "| GRU 64:", pn.n_params(dict(pn.DEFAULTS, encoder="gru")))
    assert counts[32] < counts[64] < counts[128]
    assert 80_000 < counts[64] < 500_000 and counts[128] < 600_000


def test_permutation_invariance_and_cpu():
    """The order of the player slots does not change the output; NaN slots and a team per frame
    (real tracking) are accepted."""
    pn.setup(0)
    model = pn.PlayerNet(dict(pn.DEFAULTS, channels=16)).eval()
    d = fake_match(2, n=600)
    d["alive"][:] = 1
    perm = np.random.default_rng(0).permutation(d["players"].shape[1])
    kw = dict(half=d["half"], stride=3, device="cpu")
    p1 = pn.predict_p_home(model, d["players"], d["team"], d["ball"][:, :2], None, **kw)
    p2 = pn.predict_p_home(model, d["players"][:, perm], d["team"][perm], d["ball"][:, :2], None, **kw)
    assert p1.shape == (600,) and p1.dtype == np.float32 and np.isfinite(p1).all()
    assert np.allclose(p1, p2, atol=1e-5), np.abs(p1 - p2).max()
    team_f = np.broadcast_to(d["team"], d["players"].shape[:2]).copy()
    team_f[100:200, 0] = ""                         # a fragment with unknown team
    pl = d["players"].copy()
    pl[300:400, 1] = np.nan
    dead = np.zeros(600, bool)
    dead[500:] = True
    p3 = pn.predict_p_home(model, pl, team_f, None, None, half=d["half"], dead=dead, stride=1, device="cpu")
    assert np.isfinite(p3).all() and (p3[500:] == 0.5).all()


def test_frame_logp_net_channel():
    near = np.array(["H", "", ""])
    dmin = np.array([0.5, np.inf, np.inf])
    base = dec.frame_logp(near, dmin)
    same = dec.frame_logp(near, dmin, p_home_net=None)
    assert np.allclose(base, same), "no file, no change"
    lp = dec.frame_logp(near, dmin, p_home_net=np.array([0.5, 0.9, 0.1]))
    p = np.exp(lp)
    assert np.allclose(p[0], np.exp(base)[0])
    assert p[1, 0] > p[1, 1] and p[2, 1] > p[2, 0], "the network speaks where the ball is missing"
    lp0 = dec.frame_logp(near, dmin, p_home_net=np.array([0.5, 0.9, 0.1]), net_weight=0.0)
    assert np.allclose(lp0, base)


def test_end_to_end():
    """Dataset -> LOMO grid (2 sizes) -> choice -> _net.npz -> decoder downstream -> final model
    saved and reloaded -> predict on CPU -> decoder.evaluate_dfl reads the files."""
    t0 = time.time()
    with Env(3) as env:
        res = pn.run(grid_w=(100,), grid_ch=(16, 32), epochs=6, stride=4, device="cpu",
                     base_cfg=dict(batch=64, patience=2), verbose=False)
        g = res["grid"]
        print("  griglia:\n" + g[["params", "acc_all", "acc_seen", "acc_gap", "near_gap"]].to_string())
        print("  senza palla:", res["ablation"][["acc_senza_palla"]].mean().round(3).to_dict())
        print("  decoder:\n" + res["downstream"].to_string())
        assert set(g.index) == {"W100_ch16", "W100_ch32"} and res["chosen"] in g.index
        assert g.loc[res["chosen"], "acc_all"] > 0.75, g
        assert np.isfinite(g.loc[res["chosen"], "acc_gap"])
        assert set(res["downstream"].index) == {"A1 senza rete", "A1 + rete"}
        for m in range(3):
            z = np.load(os.path.join(env.dirs["DFL_OUT"], "DFL-MAT-FAKE{}_net.npz".format(m)))
            assert z["p_home"].shape == (N_FRAMES,) and z["p_home"].dtype == np.float32
            assert str(z["variant"]).startswith("C1")
            assert ((z["p_home"] > 0) & (z["p_home"] < 1)).all()
        assert os.path.isfile(os.path.join(env.dirs["PLAYER_NET_MODELS"], "player_net.pt"))
        assert os.path.isfile(os.path.join(env.dirs["PLAYER_NET_MODELS"], "player_net.json"))
        model = pn.load_model(device="cpu")
        assert pn.cfg_name(model.cfg) == res["chosen"]
        d = fake_match(0)
        p = pn.predict_p_home(model, d["players"], d["team"], d["ball"][:, :2], None, d["half"],
                              d["alive"] == 0, stride=2, device="cpu")
        ok = np.isin(d["possession"], ["H", "A"])
        acc = np.mean((p[ok] > 0.5) == (d["possession"][ok] == "H"))
        assert acc > 0.75, acc
        ext = dec._external(dec._bench()[0], "DFL-MAT-FAKE0", N_FRAMES)
        assert "p_home_net" in ext and len(ext["p_home_net"]) == N_FRAMES
        r = dec.evaluate_dfl(seeds=(0,), save=False, verbose=False)
        assert "C1 net: yes" in r["sources"], r["sources"]
    print("  end-to-end: {:.0f} s".format(time.time() - t0))


def test_symmetries():
    """Con le simmetrie in previsione, scambiare casa e ospite da' esattamente 1 - p."""
    pn.setup(0)
    model = pn.PlayerNet(dict(pn.DEFAULTS, channels=16)).eval()
    d = fake_match(3, n=600)
    kw = dict(half=d["half"], stride=5, device="cpu", tta=True)
    p = pn.predict_p_home(model, d["players"], d["team"], d["ball"][:, :2], None, **kw)
    sw = np.where(d["team"] == "H", "A", "H")
    q = pn.predict_p_home(model, d["players"], sw, d["ball"][:, :2], None, **kw)
    live = np.abs(p - 0.5) > 1e-6
    assert live.any() and np.allclose(p[live], 1 - q[live], atol=2e-3), np.abs(p + q - 1).max()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print("{} test superati".format(len(tests)))
