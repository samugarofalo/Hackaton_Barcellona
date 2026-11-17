"""Ball in the air from one camera: flight from the gravity signature, height from the time of flight.

Test only, not in the submission chain (CONTESTO.md sections 10 and 13 list what was tried before).

The homography puts a ball at height z where its line of sight meets the grass:
G = C + (X - C) * H / (H - z), C = camera foot, H = camera height, X = point under the ball.
  1. flight   in free flight z'' = -g, so G accelerates TOWARDS the camera by about g * d / H
              (d = ground distance camera-ball): a known value, no free parameter. A rolling ball
              only brakes along its own motion, by a few m/s^2. A sliding quadratic over +/- HALF
              frames gives the acceleration; frames where its part towards the camera is
              >= RATIO * g d / H (and >= A_MIN) are flight. ball3d.py instead compared two free
              parabolas (ground vs flight), which fit a projected flight equally well.
  2. height   each run of flight frames is grown both ways while the next point stays within GATE
              of the arc's extrapolation; where it stops is the kick or the bounce (t0, t1). An arc
              that leaves and returns to the grass has z(t) = g (t - t0)(t1 - t) / 2: the time of
              flight gives the height, not the curvature. Then X = C + (G - C) * (H - z) / H.
Not done here (point 3): arcs that start or end in the air (header, chest, volley).

    .venv/bin/python pipeline/ball_flight.py demo      # synthetic lob, asserts
    .venv/bin/python pipeline/ball_flight.py dfl       # DFL: flights found, position error, touch rule
    .venv/bin/python pipeline/ball_flight.py decoder   # DFL, A1 decoder of ../hb-notebooks, 4 balls
    .venv/bin/python pipeline/ball_flight.py chain     # DFL, B2 + A1: Metrica / ours / ours + a second camera; team errors
Prints only aggregates.
"""

import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

G, FPS = 9.81, 25.0
HALF = 4         # sliding window +/- frames (9 frames: acceleration known to ~2 m/s^2 at 0.03 m noise)
MIN_PTS = 7      # observed frames needed in a window
RATIO = 0.5      # of g d / H
A_MIN = 4.0      # m/s^2, above any rolling deceleration
MIN_RUN = 4      # window centres in a row
MERGE = 25       # frames without data bridged inside one arc
GATE = 0.15      # m from the arc's extrapolation = kink (about 4 sigma of the prediction at 0.03 m)
FIT_N = 12       # observed points used to extrapolate
MAX_MISS = 5     # unobserved frames an arc may be grown across
GEFF_T, GEFF_G = (0.5, 1.25, 2.0, 3.25), (10.5, 9.9, 8.4, 6.5)  # drag and lift: effective gravity (m/s^2) by flight
                                                                # time (s), medians of the DFL grass-to-grass flights
Z_BLANK = 0.5    # m: above this estimated height the ball is 'in the air' for the rules
CAM2_OFFSET = np.array([0.0, 6.0, -7.0])  # broadcast camera relative to the tactical one (CONTESTO.md section 11)
METRICA_AIR = 0.25                        # Metrica's ball in the air: 3-4 m off above 2 m where our projection is 8-27 m (section 13)
CAM2_WHITE, CAM2_DRIFT = 0.10, 0.15       # m per axis: with our 0.03 m the two ground points differ by 0.22 m (median),
                                          # measured 0.23 m; the slow part gives the measured height drift of +/- 0.4 m


def _fit(xy, half=HALF):
    """Sliding quadratic on the observed points of each window: centre (F,2), accel (F,2), ok (F,)."""
    F = len(xy)
    t = np.arange(-half, half + 1) / FPS
    X = np.c_[np.ones_like(t), t, t * t]
    pad = np.full((half, 2), np.nan)
    Y = np.concatenate([pad, xy, pad])[np.arange(F)[:, None] + np.arange(2 * half + 1)[None]]
    w = np.isfinite(Y[..., 0]).astype(float)
    ok = (w.sum(1) >= min(MIN_PTS, 2 * half + 1)) & (w[:, :half].sum(1) >= 2) & (w[:, half + 1:].sum(1) >= 2)
    A = np.einsum("fw,wi,wj->fij", w, X, X)
    A[~ok] = np.eye(3)
    coef = np.linalg.solve(A, np.einsum("fw,wi,fwc->fic", w, X, np.nan_to_num(Y)))
    return coef[:, 0], 2 * coef[:, 2], ok


def _grow(xy, a, b, step):
    """Move one end of the arc [a, b] outwards while the next observed point fits the arc."""
    edge, nxt, miss = (b if step > 0 else a), (b if step > 0 else a) + step, 0
    while 0 <= nxt < len(xy):
        if not np.isfinite(xy[nxt, 0]):
            miss += 1
            if miss > MAX_MISS:
                break
            nxt += step
            continue
        idx = np.arange(min(a, edge), max(b, edge) + 1)
        idx = idx[np.isfinite(xy[idx, 0])]
        idx = idx[-FIT_N:] if step > 0 else idx[:FIT_N]
        if len(idx) < 5:
            break
        t = (idx - nxt) / FPS
        pred = np.linalg.lstsq(np.c_[np.ones_like(t), t, t * t], xy[idx], rcond=None)[0][0]
        if np.linalg.norm(xy[nxt] - pred) > GATE:
            break
        edge, miss = nxt, 0
        nxt += step
    return edge


def flight_arcs(xy, cam):
    """[(t0, t1)] frames of take-off and landing of each flight. xy (F,2) metres, NaN = no ball."""
    c, acc, ok = _fit(xy)
    D = c - cam[:2]
    d = np.maximum(np.linalg.norm(D, axis=1), 1e-9)
    a_cam = -np.einsum("fc,fc->f", acc, D) / d
    state = np.where(ok & (a_cam >= np.maximum(A_MIN, RATIO * G * d / cam[2])), 1, np.where(ok, 0, 2))
    arcs, i, F, end = [], 0, len(xy), 0
    while i < F:
        if state[i] != 1:
            i += 1
            continue
        j = last = i
        hole = 0
        while j + 1 < F and (state[j + 1] == 1 or hole < (MERGE if state[j + 1] == 2 else 2)):
            j += 1
            last, hole = (j, 0) if state[j] == 1 else (last, hole + 1)
        if (state[i:last + 1] == 1).sum() >= MIN_RUN:
            a, b = max(i - HALF + 1, end), min(last + HALF - 1, F - 1)
            if b > a:
                a, b = max(_grow(xy, a, b, -1), end), _grow(xy, a, b, +1)
                arcs.append((a, b))
                end = b
        i = max(last, end) + 1
    return arcs


def correct(xy, cam, arcs=None, ends=None):
    """Ground point under the ball, height and flight mask. arcs: [(t0, t1)] (default: found here).
    ends: optional {(t0, t1): (z0, z1)} heights at the two ends (0, 0 if absent)."""
    arcs = flight_arcs(xy, cam) if arcs is None else arcs
    z = np.zeros(len(xy))
    for a, b in arcs:
        t = np.arange(a, b + 1)
        z0, z1 = (ends or {}).get((a, b), (0.0, 0.0))
        s = (t - a) / max(b - a, 1)
        z[a:b + 1] = z0 * (1 - s) + z1 * s + 0.5 * np.interp((b - a) / FPS, GEFF_T, GEFF_G) * (t - a) * (b - t) / FPS ** 2
    z = np.minimum(z, 0.9 * cam[2])
    out = cam[:2] + (xy - cam[:2]) * ((cam[2] - z) / cam[2])[:, None]
    return out, z, z > 0


def project(true_xy, z, cam):
    return cam[:2] + (true_xy - cam[:2]) * (cam[2] / np.maximum(cam[2] - z, 1.0))[:, None]


def triangulate(g1, c1, g2, c2):
    """3D point nearest to the two lines of sight c1 -> g1 and c2 -> g2 (g = where each camera's
    homography puts the ball on the grass, metres)."""
    u, v, w = np.c_[g1, np.zeros(len(g1))] - c1, np.c_[g2, np.zeros(len(g2))] - c2, c1 - c2
    a, b, c, d, e = (u * u).sum(1), (u * v).sum(1), (v * v).sum(1), u @ w, v @ w
    den = a * c - b * b
    return (c1 + ((b * e - c * d) / den)[:, None] * u + c2 + ((a * e - b * d) / den)[:, None] * v) / 2


def demo():
    """A ball rolls 1 s, is lobbed for 1.5 s (apex about 2.6 m), lands and rolls on."""
    rng = np.random.default_rng(0)
    cam = np.array([52.5, -48.0, 29.0])
    t = np.arange(100) / FPS
    x = 30 + 8 * t
    y = np.full_like(t, 30.0)
    tf = np.clip(t - 1.0, 0, 1.5)
    z = 0.5 * np.interp(1.5, GEFF_T, GEFF_G) * tf * (1.5 - tf)  # a real ball: gravity as seen with drag
    true = np.c_[x, y]
    xy, zh, fl = correct(project(true, z, cam) + rng.normal(0, 0.03, true.shape), cam)
    err = np.linalg.norm(xy - true, axis=1)
    raw = np.linalg.norm(project(true, z, cam) - true, axis=1)
    print("demo: apex {:.2f} m (true {:.2f}); error p90 {:.2f} m (projected {:.2f} m); flight frames {} of {}".format(
        zh.max(), z.max(), np.percentile(err, 90), np.percentile(raw, 90), fl.sum(), (z > 0).sum()))
    assert abs(zh.max() - z.max()) < 0.3 and np.percentile(err, 90) < 0.5 and not fl[:20].any()
    c2 = cam + CAM2_OFFSET
    assert np.abs(triangulate(project(true, z, cam), cam, project(true, z, c2), c2) - np.c_[true, z]).max() < 1e-6


def _truth_arcs(ztrue):
    """Oracle: free-flight stretches of the true ball (z'' near -g, z above the grass)."""
    _, acc, ok = _fit(np.c_[ztrue, np.zeros_like(ztrue)], 2)
    free = ok & (acc[:, 0] < -5) & (acc[:, 0] > -15) & (ztrue > 0.15)
    e = np.diff(np.r_[0, free.astype(np.int8), 0])
    return [(max(a - 2, 0), min(b + 1, len(ztrue) - 1)) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)) if b - a >= 4]


def dfl():
    sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), HERE]
    import pandas as pd
    import dfl_degrade as dd
    dd.ALIVE_GAP_SHARE = 0.0
    from dfl_bench import CONVERTED, SCALE, load, score
    from dfl_control import touches, nearest, run_rule
    rows, acc = [], {}
    quick = bool(os.environ.get("QUICK"))
    for ang in ((25.0,) if quick else (19.5, 25.0, 35.0)):
        H = 30.0
        cam = np.array([52.5, 34 - H / np.tan(np.radians(ang)), H])
        for path in sorted(glob.glob(os.path.join(CONVERTED, "*.npz")))[:3 if quick else None]:
            d, gt = load(path)
            P, alive = d["players"], d["alive"]
            team = np.broadcast_to(d["team"], P.shape[:2])
            rng = np.random.default_rng(0)
            dead = ~np.isfinite(dd.metrica_like_ball(rng, d["ball"][:, :2], alive)[:, 0])
            ztrue = np.clip(np.nan_to_num(d["ball"][:, 2]), 0, None)
            true = d["ball"][:, :2] * SCALE
            proj = project(true, ztrue, cam) + rng.normal(0, 0.03, true.shape)
            proj[dead] = np.nan
            Pq = P + rng.normal(0, 0.02, P.shape) / SCALE
            live = (alive == 1) & np.isfinite(proj[:, 0])
            ta = _truth_arcs(np.where(dead, np.nan, ztrue))
            ests = {"projected": (proj, np.zeros(len(proj)), np.zeros(len(proj), bool)),
                    "signature + time of flight": correct(proj, cam),
                    "true arcs, ends on grass": correct(proj, cam, ta),
                    "true arcs, true end heights": correct(proj, cam, ta, {(a, b): (ztrue[a], ztrue[b]) for a, b in ta})}
            for name, (est, zh, fl) in ests.items():
                for key, m in (("all", live), ("air > 1 m", live & (ztrue > 1)), ("grass < 0.3 m", live & (ztrue < 0.3))):
                    acc.setdefault((ang, name, "error " + key), []).append(np.linalg.norm(est[m] - true[m], axis=1))
                acc.setdefault((ang, name, "height error, air > 1 m"), []).append(np.abs(zh - ztrue)[live & (ztrue > 1)])
                for key, num, den in (("flight called, of ball > 1 m", fl & (ztrue > 1), ztrue > 1),
                                      ("flight called, of ball > 0.5 m", fl & (ztrue > 0.5), ztrue > 0.5),
                                      ("ball > 0.3 m, of flight called", fl & (ztrue > 0.3), fl),
                                      ("flight called, of live frames", fl, np.ones(len(fl), bool)),
                                      ("est. height > 0.5 m, of ball > 1 m", (zh > Z_BLANK) & (ztrue > 1), ztrue > 1),
                                      ("ball > 0.3 m, of est. height > 0.5 m", (zh > Z_BLANK) & (ztrue > 0.3), zh > Z_BLANK)):
                    acc.setdefault((ang, name, key), []).append((int((num & live).sum()), int((den & live).sum())))
                for key, thr in (("true ball > 0.3 m, of live frames", 0.3), ("true ball > 1 m, of live frames", 1.0)):
                    acc.setdefault((ang, name, key), []).append((int((live & (ztrue > thr)).sum()), int(live.sum())))
                b = est / SCALE
                near, dist = nearest(Pq, team, b)
                tc = touches(b, dist, 2.0)
                rows.append((ang, name, "touch", score(d, gt, run_rule(near, dist, dead, None, 0, tc))[1]))
                rows.append((ang, name, "touch, none while est. height > 0.5 m", score(d, gt, run_rule(near, dist, dead, None, 0, tc & (zh <= Z_BLANK)))[1]))
            print(os.path.basename(path)[-10:-4], ang, flush=True)
    print("\nshare of live frames with the true ball > 0.5 m / > 1 m: see 'flight called' denominators")
    for (ang, name, key), v in acc.items():
        if isinstance(v[0], tuple):
            n, den = np.sum(v, axis=0)
            print("  {:>4.1f} deg  {:<28} {:<32} {:.1%}".format(ang, name, key, n / max(den, 1)))
        else:
            v = np.concatenate(v)
            print("  {:>4.1f} deg  {:<28} {:<32} p50 {:.2f} m  p90 {:.2f} m".format(ang, name, key, *np.percentile(v, [50, 90])))
    df = pd.DataFrame(rows, columns=["angle", "ball", "rule", "spell_f1"])
    print("\nspellF1 (reference: true ground ball, touch 0.705)")
    print(df.groupby(["angle", "ball", "rule"], sort=False).spell_f1.mean().round(3).to_string())


def _hb():
    """The colleague's modules (../hb-notebooks, files untouched) in corrected conditions: players
    0.02 m, no Metrica gaps in live play."""
    hb = os.environ.get("HB", os.path.join(os.path.dirname(ROOT), "hb-notebooks"))
    sys.path.insert(0, os.path.join(hb, "pipeline"))
    os.environ.setdefault("DFL_CONVERTED", os.path.join(ROOT, "data", "external", "bassek2025", "converted"))
    os.environ.setdefault("DFL_BALL_RUNS", os.path.join(ROOT, "data", "cleanup", "ball_runs.npz"))
    import decoder as dec
    import dfl_degrade
    import dfl_method4 as m4
    m4.PLAYER_NOISE_M, dfl_degrade.ALIVE_GAP_SHARE = 0.02, 0.0
    return dec, dfl_degrade, m4


def decoder():
    """A1 decoder of ../hb-notebooks (its files untouched) on four balls, corrected conditions
    (players 0.02 m, no Metrica gaps in live play), without B1 / B2 files, one seed."""
    dec, dfl_degrade, m4 = _hb()
    ang = float(os.environ.get("ANGLE", 30.0))
    cam = np.array([52.5, 34 - 30.0 / np.tan(np.radians(ang)), 30.0])
    fill0 = m4.fill

    def fill_corrected(ours, metrica, max_gap):
        return fill0(correct(ours * m4.SCALE, cam)[0] / m4.SCALE, metrica, max_gap)

    def fill_blank(ours, metrica, max_gap):
        b, src = fill0(ours, metrica, max_gap)
        b[correct(ours * m4.SCALE, cam)[1] > Z_BLANK] = np.nan
        return b, src

    xy0, true = m4.our_ball_xy, {}

    def our_ball_xy(d):  # keeps the true height of the match being simulated, for the two oracle rows
        true["z"] = np.clip(np.nan_to_num(d["ball"][:, 2]), 0, None)
        return xy0(d)
    m4.our_ball_xy = our_ball_xy

    def fill_true_ends(ours, metrica, max_gap, true_arcs=False):
        xy, z = ours * m4.SCALE, true["z"]
        arcs = _truth_arcs(np.where(np.isfinite(metrica[:, 0]), z, np.nan)) if true_arcs else flight_arcs(xy, cam)
        return fill0(correct(xy, cam, arcs, {(a, b): (z[a], z[b]) for a, b in arcs})[0] / m4.SCALE, metrica, max_gap)

    only = os.environ.get("ONLY")
    out = []
    for name, env, fill in (("Metrica only", {"DFL_OURS_NONE": "1"}, fill0),
                            ("ours, no height error (old sim)", {}, fill0),
                            ("ours, projected", {"DFL_CAM_ANGLE": str(ang)}, fill0),
                            ("ours, signature + time of flight", {"DFL_CAM_ANGLE": str(ang)}, fill_corrected),
                            ("ours, blank while in the air", {"DFL_CAM_ANGLE": str(ang)}, fill_blank),
                            # oracles: what point 3 (height at the ends of each arc) could give at best
                            ("ours, our arcs + true end heights", {"DFL_CAM_ANGLE": str(ang)}, fill_true_ends),
                            ("ours, true arcs + true end heights", {"DFL_CAM_ANGLE": str(ang)},
                             lambda o, m, g: fill_true_ends(o, m, g, True))):
        if only and only not in name:
            continue
        for k in ("DFL_OURS_NONE", "DFL_CAM_ANGLE"):
            os.environ.pop(k, None)
        os.environ.update(env)
        m4.fill = fill
        s = dec.evaluate_dfl(seeds=(0,), use_files=False, save=False, verbose=False)["summary"]["spell_f1"]["mean"]
        out.append((name, s))
        print(name, "done", flush=True)
    print("\nspellF1, camera {:.0f} deg, mean of 7 matches".format(ang))
    for name, s in out:
        print("  {:<34} {}".format(name, "  ".join("{} {:.3f}".format(k.split(" (")[0], v) for k, v in s.items())))


def chain():
    """Whole chain of ../hb-notebooks -- B2 written to a scratch $DFL_OUT (no B1 there), then the A1
    decoder reading it -- in corrected conditions, one seed, B2 and decoder parameters chosen on
    the other matches. Balls: Metrica only / ours / ours with the height from a second camera where
    it sees the ball in the air; Metrica's ball exact in the air (as simulated so far) or off by
    METRICA_AIR of our projection error. Players: the carrier's team wrong in 0, 1, 5% of live play.
    $ONLY picks rows, $ANGLE the tactical camera (default 19.5, the real one at the centre spot),
    $ABSORB the challenge-like truth of dfl_definition.py (parameters are then chosen against it),
    $WIDE=1 a wider decoder grid."""
    dec, dfl_degrade, m4 = _hb()
    import pandas as pd
    import dfl_bench
    import dfl_paths
    import touch_control as tc
    from dfl_definition import absorb_short  # after the colleague's modules: it puts this repository first on the path
    tc.PLAYER_NOISE_M = m4.PLAYER_NOISE_M
    absorb = float(os.environ.get("ABSORB", 0))  # s: DFL possessions shorter than this go to the previous team
    if os.environ.get("WIDE"):  # the decoder's grid, extended past its edge
        dec.SWITCH_COSTS += (256.0,)
        dec.MIN_DURS += ((75, 25), (100, 25))
    ang = float(os.environ.get("ANGLE", 19.5))
    cam = np.array([52.5, 34 - 30.0 / np.tan(np.radians(ang)), 30.0])
    fill0, xy0, deg0, true = m4.fill, m4.our_ball_xy, dfl_degrade.degrade, {}
    load0, met0 = dfl_bench.load, (dfl_degrade.metrica_like_ball, tc.metrica_dead_ball)

    def load(path):  # the height of the match being simulated, for Metrica's error in the air
        npz, gt = load0(path)
        true["z"] = np.clip(np.nan_to_num(npz["ball"][:, 2]), 0, None)
        return npz, absorb_short(gt, npz["half"], absorb) if absorb else gt
    dfl_bench.load = tc.load = load

    def air_error(f, k):
        def g(rng, ball, alive, *a):
            xy = ball * m4.SCALE
            return f(rng, (xy + k * (project(xy, true["z"], cam) - xy)) / m4.SCALE, alive, *a)
        return g

    def our_ball_xy(d):
        true["xyz"] = np.c_[d["ball"][:, :2] * m4.SCALE, np.clip(np.nan_to_num(d["ball"][:, 2]), 0, None)]
        return xy0(d)

    def fill_true_ends(ours, metrica, max_gap):  # oracle: the most the flights found can give
        xy, z = ours * m4.SCALE, true["xyz"][:, 2]
        arcs = flight_arcs(xy, cam)
        return fill0(correct(xy, cam, arcs, {(a, b): (z[a], z[b]) for a, b in arcs})[0] / m4.SCALE, metrica, max_gap)

    def fill_cam2(share, by_arc=False, seen_n=150):
        def fill(ours, metrica, max_gap):
            n, rng, c2 = len(ours), np.random.default_rng(7), cam + CAM2_OFFSET
            drift = np.stack([np.convolve(rng.normal(size=n + 249), np.ones(250) / 250, "valid") for _ in range(2)], 1)
            g2 = project(true["xyz"][:, :2], true["xyz"][:, 2], c2) + rng.normal(0, CAM2_WHITE, (n, 2)) + drift * CAM2_DRIFT * 250 ** 0.5
            seen = np.ones(n, bool) if share >= 1 else m4.runs_mask(rng, n, rng.geometric(1 / seen_n, 500),
                                                                   rng.geometric(share / (seen_n * (1 - share)), 500))
            xy = ours * m4.SCALE
            z = np.where(seen, triangulate(xy, cam, g2, c2)[:, 2], np.nan)
            true["high"] = true.get("high", 0) + int((np.isfinite(xy[:, 0]) & (true["xyz"][:, 2] > Z_BLANK)).sum())
            if by_arc:  # only on the flights found by the gravity signature: camera 2 gives the two end heights
                ends = {}
                for a, b in flight_arcs(xy, cam):
                    t, ok = np.arange(a, b + 1), np.isfinite(z[a:b + 1])
                    if ok.sum() >= 5 and np.ptp(t[ok]) >= (b - a) / 2:
                        sh = (t - a) / max(b - a, 1)
                        par = 0.5 * np.interp((b - a) / FPS, GEFF_T, GEFF_G) * (t - a) * (b - t) / FPS ** 2
                        ends[(a, b)] = tuple(np.clip(np.linalg.lstsq(np.c_[1 - sh, sh][ok], (z[a:b + 1] - par)[ok], rcond=None)[0], 0, None))
                xy, zh, _ = correct(xy, cam, list(ends), ends)
                true["air"] = true.get("air", 0) + int((zh > Z_BLANK).sum())
                return fill0(xy / m4.SCALE, metrica, max_gap)
            z = pd.Series(z).rolling(7, center=True, min_periods=3).mean().to_numpy()
            air = z > Z_BLANK  # False where either camera misses the ball
            true["air"] = true.get("air", 0) + int(air.sum())
            xy[air] = cam[:2] + (xy[air] - cam[:2]) * ((cam[2] - z[air]) / cam[2])[:, None]
            return fill0(xy / m4.SCALE, metrica, max_gap)
        return fill

    cam_env, none = {"DFL_CAM_ANGLE": str(ang)}, {"DFL_OURS_NONE": "1"}
    only, out = os.environ.get("ONLY"), []
    rows = (("Metrica only", none, fill0, 0.01, 0),
            ("Metrica only, carrier's team never wrong", none, fill0, 0.0, 0),
            ("Metrica only, carrier's team wrong 5%", none, fill0, 0.05, 0),
            ("ours, projected", cam_env, fill0, 0.01, 0),
            ("ours + camera 2 in the air, seen 48%", cam_env, fill_cam2(0.48), 0.01, 0),
            ("ours + camera 2 in the air, always seen", cam_env, fill_cam2(1.0), 0.01, 0),
            ("ours + camera 2 on the flights found, seen 48%", cam_env, fill_cam2(0.48, True), 0.01, 0),
            ("ours + camera 2 on the flights found, always seen", cam_env, fill_cam2(1.0, True), 0.01, 0),
            ("ours, flights found + true end heights", cam_env, fill_true_ends, 0.01, 0),
            ("ours, no height error", {}, fill0, 0.01, 0),
            ("Metrica off in the air: Metrica only", none, fill0, 0.01, METRICA_AIR),
            ("Metrica off in the air: ours, projected", cam_env, fill0, 0.01, METRICA_AIR),
            ("Metrica off in the air: ours + camera 2, seen 48%", cam_env, fill_cam2(0.48), 0.01, METRICA_AIR))
    for name, env, fill, swap, off in rows:
        if only and (only != name if only in [r[0] for r in rows] else only not in name):  # a full name picks that row alone
            continue
        for k in ("DFL_OURS_NONE", "DFL_CAM_ANGLE"):
            os.environ.pop(k, None)
        os.environ.update(env)
        os.environ["DFL_OUT"] = os.path.join(os.environ.get("OUT", "/tmp/ball_flight_chain"), re.sub(r"\W+", "_", name))
        m4.fill = tc.fill = fill
        m4.our_ball_xy = tc.our_ball_xy = our_ball_xy
        dfl_degrade.degrade = tc.degrade = lambda kind, level, *a, swap=swap: deg0(kind, swap, *a)
        dfl_degrade.metrica_like_ball, tc.metrica_dead_ball = air_error(met0[0], off), air_error(met0[1], off)
        true.pop("air", None), true.pop("high", None)
        paths = dfl_paths.match_paths()
        df, kept, _ = tc.evaluate_all(paths, (0,), tc.GRID, "metrica", False, {dfl_paths.match_name(p) for p in paths}, False)
        held, chosen = tc.lomo(df)
        tc.save_outputs(kept, chosen, 0)
        res = dec.evaluate_dfl(seeds=(0,), use_files=True, save=True, verbose=False)
        assert "p_home B2: yes" in res["sources"], res["sources"]
        s = dict(res["summary"]["spell_f1"]["mean"])
        s["B2"] = held[held.method == "tocco + controllo"].spell_f1.mean()
        out.append((name, s, true.get("air"), true.get("high")))
        print(name, "done; chosen B2", pd.Series({m: c for (meth, m), c in chosen.items() if meth == "tocco + controllo"}).value_counts().to_dict(),
              "| chosen A1", pd.Series(res["params"]).value_counts().to_dict(), flush=True)
        for t in (0, 2, 3):  # the labels just written (parameters chosen against $ABSORB), scored against each truth
            sc = []
            for path in paths:
                npz, gt = load0(path)
                sc.append([dfl_bench.score(npz, absorb_short(gt, npz["half"], t) if t else gt,
                                           np.load(dfl_paths.out_path(dfl_paths.match_name(path), k))["labels"])[1] for k in ("rule", "decoded")])
            print("  scored against DFL with possessions under {} s given to the previous team: B2 {:.3f}  decoder {:.3f}".format(t, *np.mean(sc, axis=0)), flush=True)
    print("\nspellF1, tactical camera {:.1f} deg, mean of 7 matches, DFL possessions under {:g} s given to the previous team".format(ang, absorb))
    for name, s, air, high in out:
        print("  {:<50} {}".format(name, "  ".join("{} {:.3f}".format(k.split(" (")[0], v) for k, v in s.items())))
        if air is not None:  # counted over B2's and the decoder's simulations, the same frames twice
            print("  {:<50} height from camera 2 used in {:.0%} as many frames as our ball has above {} m".format("", air / max(high, 1), Z_BLANK))


if __name__ == "__main__":
    {"demo": demo, "dfl": dfl, "decoder": decoder, "chain": chain}[sys.argv[1]]()
