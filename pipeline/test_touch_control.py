"""Synthetic checks of the touch + control rule (no real data needed).

    python pipeline/test_touch_control.py        (or: python -m pytest pipeline/test_touch_control.py)
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import touch_control as tc  # noqa: E402
from dfl_bench import SCALE  # noqa: E402

F = 400
FAR = 20.0


def feats(n=F, dead=None, src=None):
    """Blank scene: H has the ball at its feet (0.3 m), A far, slow ball."""
    near = np.full(n, "H", dtype="<U1")
    dh = np.full(n, 0.3)
    da = np.full(n, FAR)
    speed = np.full(n, 10.0)
    dead = np.zeros(n, bool) if dead is None else dead
    return near, dh, da, speed, dead, src


def build(near, dh, da, speed, dead, src):
    dmin = np.minimum(dh, da)
    near = np.where(dh <= da, "H", "A")
    return tc.Features(near, dmin, dh, da, speed, dead, src)


def a_has_ball(dh, da, a, z, d_a=0.4, d_h=6.0):
    da[a:z], dh[a:z] = d_a, d_h


def ball_free(dh, da, a, z):
    dh[a:z], da[a:z] = 8.0, 9.0


def test_interception_controlled():
    near, dh, da, speed, dead, src = feats()
    ball_free(dh, da, 30, 100)                   # H passes at 30
    a_has_ball(dh, da, 100, F)                   # A intercepts at 100 and keeps it
    touch = np.zeros(F, bool)
    touch[[10, 30, 100]] = True
    f = build(near, dh, da, speed, dead, src)
    lab, p, info = tc.run_touch_control(f, touch, T1=1.0, R=1.0, N=4)
    assert (lab[:100] == "H").all() and (lab[100:] == "A").all(), "change dated at the first touch"
    assert info["changes"][-1][:2] == (100, "A")
    assert (p[100:] < 0.5).all() and (p[:100] > 0.5).all()


def test_deflection_not_a_change():
    near, dh, da, speed, dead, src = feats()
    ball_free(dh, da, 30, 140)
    da[100], dh[100] = 0.8, 7.0                  # A brushes the ball at 100, then it runs free
    dh[140:] = 0.4                               # H team-mate receives at 140
    touch = np.zeros(F, bool)
    touch[[30, 100, 140]] = True
    f = build(near, dh, da, speed, dead, src)
    lab, _, info = tc.run_touch_control(f, touch, T1=1.0, R=1.0, N=4)
    assert (lab == "H").all(), "a touch without control keeps the possession"
    assert (100, "A") in info["unconfirmed"]
    m4 = tc.method4_labels(f, touch)
    assert (m4[100:140] == "A").all(), "method 4 does change at every touch"


def test_second_touch_confirms():
    near, dh, da, speed, dead, src = feats()
    ball_free(dh, da, 30, F)
    da[100] = 0.8                                # A one-touch pass ...
    da[115] = 0.8                                # ... to an A team-mate
    touch = np.zeros(F, bool)
    touch[[30, 100, 115]] = True
    f = build(near, dh, da, speed, dead, src)
    lab, _, info = tc.run_touch_control(f, touch, T1=1.0, R=1.0, N=4)
    assert (lab[100:] == "A").all() and (lab[:100] == "H").all()
    assert info["changes"][-1][2] == "a"
    lab_short, _, _ = tc.run_touch_control(f, touch, T1=0.4, R=1.0, N=4)  # 10-frame window
    assert (lab_short == "H").all(), "outside T1 the second touch does not confirm"


def test_window_closed_by_opponent_touch():
    near, dh, da, speed, dead, src = feats()
    ball_free(dh, da, 30, F)
    da[100] = 0.8
    dh[105] = 0.8                                # H touches back before A controls
    a_has_ball(dh, da, 112, F)                   # later A gets close, but without a touch
    touch = np.zeros(F, bool)
    touch[[30, 100, 105]] = True
    f = build(near, dh, da, speed, dead, src)
    lab, _, _ = tc.run_touch_control(f, touch, T1=1.6, R=1.0, N=4)
    assert (lab[:300] == "H").all(), "the A touch is cut by the H touch; no new A touch"
    lab_fb, _, info = tc.run_touch_control(f, touch, T1=1.6, R=1.0, N=4, fallback_n=25)
    assert (lab_fb[112:] == "A").all() and info["changes"][-1][2] == "fallback"


def test_ball_gap_fallback_control():
    src = np.zeros(F, int)
    src[150:250] = 1                             # ball interpolated: no touch seen
    near, dh, da, speed, dead, _ = feats(src=src)
    a_has_ball(dh, da, 160, F)                   # A wins it inside the gap
    touch = np.zeros(F, bool)
    touch[[10, 50]] = True
    f = build(near, dh, da, speed, dead, src)
    lab, p, _ = tc.run_touch_control(f, touch, fallback_n=None)
    assert (lab == "H").all(), "without fallback a change without touch is not seen"
    lab, p, _ = tc.run_touch_control(f, touch, R=1.0, fallback_n=25)
    assert (lab[160:] == "A").all() and (lab[:160] == "H").all(), "dated at the start of control"
    # confidence decays faster while the ball is not ours
    assert abs(p[155] - 0.5) < abs(p[140] - 0.5)


def test_dead_ball_and_restart():
    dead = np.zeros(F, bool)
    dead[200:260] = True
    near, dh, da, speed, _, src = feats(dead=dead)
    ball_free(dh, da, 260, 270)                  # restart: ball reappears loose ...
    a_has_ball(dh, da, 270, F)                   # ... and A is the first to control it
    touch = np.zeros(F, bool)
    touch[[10, 270]] = True
    f = build(near, dh, da, speed, dead, src)
    lab, p, _ = tc.run_touch_control(f, touch)
    assert (lab[200:260] == "D").all() and (p[200:260] == 0.5).all()
    assert (lab[:200] == "H").all()
    assert (lab[260:] == "A").all(), "first confirmed control after D decides, back to the restart"


def test_backward_control_dates_earlier():
    near, dh, da, speed, dead, src = feats()
    ball_free(dh, da, 30, 90)
    a_has_ball(dh, da, 90, F)                    # A received at 90 (touch not seen), dribbles
    touch = np.zeros(F, bool)
    touch[[30, 120]] = True                      # first A touch seen only at 120
    f = build(near, dh, da, speed, dead, src)
    lab, _, info = tc.run_touch_control(f, touch, R=1.0, N=4)
    assert (lab[90:] == "A").all() and (lab[:90] == "H").all() and info["changes"][-1][2] == "d"


def test_end_to_end_from_tracking():
    """Positions in metres -> touches -> rule: an H pass intercepted and controlled by A."""
    n = 200
    ball = np.zeros((n, 2))
    ball[:30] = [10, 34]
    for i in range(30, 80):                      # pass at 15 m/s along x
        ball[i] = [10 + 15 * (i - 30) / 25, 34]
    for i in range(80, n):                       # A stops it and dribbles at 2 m/s along y
        ball[i] = [40, 34 + 2 * (i - 80) / 25]
    players = np.full((n, 4, 2), np.nan)
    players[:, 0] = [10, 34.3]                   # H passer
    players[:, 1] = [60, 20]                     # H far
    players[:80, 2] = [40, 34.4]                 # A interceptor waits ...
    players[80:, 2] = ball[80:] + [0, 0.4]       # ... and runs with the ball
    players[:, 3] = [70, 50]
    team = np.array(["H", "H", "A", "A"])
    norm = lambda x: x / SCALE  # noqa: E731
    dead = np.zeros(n, bool)
    f = tc.Features.from_tracking(norm(players), team, norm(ball), dead)
    touch = tc.detect_touches(norm(ball), f.dmin, np.zeros(n, int), 2.0)
    assert touch[25:36].any() and touch[75:86].any(), "the kick and the stop are touches"
    lab, p, _ = tc.run_touch_control(f, touch)
    first_a = int(np.flatnonzero(lab == "A")[0])
    assert 75 <= first_a <= 82 and (lab[:first_a] == "H").all() and (lab[first_a:] == "A").all()
    assert p.dtype == np.float32 and ((p > 0.5) == (lab == "H")).all()


def test_pipeline_on_fake_matches():
    """evaluate_all + lomo + save on two tiny fake converted matches (format of convert_dfl.py)."""
    tmp = tempfile.mkdtemp()
    conv, out = os.path.join(tmp, "conv"), os.path.join(tmp, "out")
    os.makedirs(conv)
    rng = np.random.default_rng(0)
    n, P = 3000, 6
    for m in range(2):
        ball = np.zeros((n, 3), np.float32)
        players = rng.uniform(0.2, 0.8, (n, P, 2)).astype(np.float32)
        poss = np.empty(n, "<U1")
        owner = 0
        for i in range(n):
            if i % 150 == 0:
                owner = int(rng.integers(0, P))
            players[i, owner] = players[i - 1, owner] + 0.002 if i % 150 else players[i, owner]
            ball[i, :2] = players[i, owner] + 0.004
            poss[i] = "H" if owner < P // 2 else "A"
        alive = np.ones(n, np.int8)
        alive[1000:1100] = 0
        half = np.r_[np.ones(n // 2, np.int8), np.full(n - n // 2, 2, np.int8)]
        np.savez(os.path.join(conv, "DFL-MAT-FAKE{}.npz".format(m)), players=players, ball=ball,
                 team=np.array(["H"] * (P // 2) + ["A"] * (P - P // 2)), alive=alive,
                 possession=poss, half=half)
    keys = ("DFL_CONVERTED", "DFL_OUT", "DFL_BALL_RUNS")
    old = {k: os.environ.get(k) for k in keys}
    os.environ["DFL_CONVERTED"], os.environ["DFL_OUT"] = conv, out
    os.environ["DFL_BALL_RUNS"] = "synthetic"   # never touch data/ in a test
    try:
        df, held, chosen = tc.main(["--quick", "--seeds", "0"])
        assert set(held.method) == {"baseline +/-50", "metodo 4 (solo tocco)", "tocco + controllo"}
        z = np.load(os.path.join(out, "DFL-MAT-FAKE0_rule.npz"))
        assert z["labels"].shape == (n,) and z["p_home"].dtype == np.float32
        assert set(np.unique(z["labels"])) <= {"H", "A", "D"} and str(z["variant"]).startswith("B2")
    finally:
        for k, v in old.items():  # restore the caller's settings (the notebook's, on Kaggle)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print("{} test superati".format(len(tests)))
