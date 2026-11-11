"""Straight flights: seen from above, a ball between two touches goes straight at nearly constant
speed; one camera bends that line (height moves the projected ball along the line of sight). So:
cut the projected track at sharp velocity changes (kick, bounce, header) and gaps, and where a
segment strays from the chord between its two ends by more than DEV metres, replace it with the
chord walked at constant speed. The ends of a segment are kicks and bounces, i.e. ball on the grass,
where the projection is right. Curled ground passes stay below DEV and are left alone.

Test only, not in the submission chain.
  dfl     DFL matches (true height known): one camera simulated as in ball3d.py, position error and
          spellF1 of the touch rules on the projected vs straightened ball
  match   our match: our final ball vs the 3D ball of the two cameras (triangulate.py) as referee

    .venv/bin/python pipeline/straight_ball.py dfl|match
Prints only aggregates.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), HERE]

SPLIT_DV = 5.0     # m/s over 4 frames: gravity alone bends the projection by < 3 m/s there (ball3d.py)
MIN_SEG = 8        # frames
DEV_GRID = (1.0, 2.0, 3.0, 5.0)
ANGLES = (25.0, 35.0)


def straighten(xy, dev, split_dv=SPLIT_DV, min_seg=MIN_SEG, cut_from=None):
    """xy (F,2) metres, NaN = no ball. Returns straightened copy and the bool mask of changed frames.
    cut_from: track to find the cuts on (default xy itself; the true ground ball = oracle cuts)."""
    F = len(xy)
    sm = pd.DataFrame(xy if cut_from is None else cut_from).rolling(5, center=True, min_periods=1).mean().to_numpy()
    v = np.gradient(sm, axis=0) * 25
    dv = np.zeros(F)
    dv[2:-2] = np.linalg.norm(v[4:] - v[:-4], axis=1)
    dv = np.nan_to_num(dv, nan=99.0)
    peak = (dv >= np.roll(dv, 1)) & (dv >= np.roll(dv, -1)) & (dv > split_dv)
    cut = peak | ~np.isfinite(xy[:, 0])
    out, changed = xy.copy(), np.zeros(F, bool)
    edges = np.r_[0, np.flatnonzero(cut), F]
    for a, b in zip(edges[:-1], edges[1:]):
        if a > 0 and not np.isfinite(xy[a, 0]):
            a += 1
        if b < F and np.isfinite(xy[b, 0]):
            b += 1  # the cut frame (kick / bounce) is the shared end of both segments
        if b - a < min_seg or not np.all(np.isfinite(xy[a:b, 0])):
            continue
        p, q = xy[a], xy[b - 1]
        u = q - p
        L = np.linalg.norm(u)
        if L < 1.0:
            continue
        off = np.abs(np.cross(u / L, xy[a:b] - p))
        if off.max() > dev:
            s = np.linspace(0, 1, b - a)[:, None]
            out[a:b] = p + s * u
            changed[a:b] = True
    return out, changed


def dfl():
    import dfl_degrade as dd
    dd.ALIVE_GAP_SHARE = 0.0
    from dfl_bench import CONVERTED, SCALE, load, score
    from dfl_control import touches, nearest, run_rule, sight_touches
    rows, err = [], {}
    for ang in ANGLES:
        H = 30.0
        cam = np.array([52.5, 34 - H / np.tan(np.radians(ang)), H])
        cot = 1 / np.tan(np.radians(ang))
        for path in sorted(glob.glob(os.path.join(CONVERTED, "*.npz"))):
            d, gt = load(path)
            P, alive = d["players"], d["alive"]
            team = np.broadcast_to(d["team"], P.shape[:2])
            rng = np.random.default_rng(0)
            dead = ~np.isfinite(dd.metrica_like_ball(rng, d["ball"][:, :2], alive)[:, 0])
            ztrue = np.clip(d["ball"][:, 2], 0, None)
            true = d["ball"][:, :2] * SCALE
            k = cam[2] / np.maximum(cam[2] - ztrue, 1.0)
            proj = cam[:2] + (true - cam[:2]) * k[:, None] + rng.normal(0, 0.03, true.shape)
            proj[dead] = np.nan
            Pq = P + rng.normal(0, 0.02, P.shape) / SCALE
            live = (alive == 1) & np.isfinite(proj[:, 0])
            ests = [("projected", proj, None)] + [("straight dev {:g}".format(v),) + straighten(proj, v) for v in DEV_GRID]
            ests += [("oracle cuts dev {:g}".format(v),) + straighten(proj, v, 2.0, cut_from=true) for v in (1.0, 3.0)]
            for name, est, ch in ests:
                for key, m in (("all", live), ("air>1m", live & (ztrue > 1)), ("ground<0.3m", live & (ztrue < 0.3))):
                    err.setdefault((ang, name, key), []).append(np.linalg.norm(est[m] - true[m], axis=1))
                if ch is not None:
                    err.setdefault((ang, name, "changed share"), []).append(np.array([ch[live].mean()]))
                    err.setdefault((ang, name, "changed, ball >1m"), []).append(np.array([(ztrue[live & ch] > 1).mean()]))
                b = est / SCALE
                near, dist = nearest(Pq, team, b)
                rows.append((ang, name, "touch", score(d, gt, run_rule(near, dist, dead, None, 0, touches(b, dist, 2.0)))[1]))
                tc, dm, kk = sight_touches(b, Pq, 2.0, cam[:2], 2.0 * cot)
                nt = np.where(np.isfinite(dm), team[np.arange(len(kk)), kk], "")
                rows.append((ang, name, "touch across sight", score(d, gt, run_rule(nt, dm, dead, None, 0, tc))[1]))
            print(os.path.basename(path), ang, flush=True)
    print("\nposition error, metres (p50 / p90); changed share of live frames")
    for (ang, name, key), v in err.items():
        v = np.concatenate(v)
        if key.startswith("changed"):
            print("  {:.0f} deg {:<18} {:<18} {:.1%}".format(ang, name, key, v.mean()))
        else:
            print("  {:.0f} deg {:<18} {:<18} {:.2f} / {:.2f}".format(ang, name, key, *np.percentile(v, [50, 90])))
    df = pd.DataFrame(rows, columns=["angle", "ball", "rule", "spell_f1"])
    print(df.groupby(["angle", "rule", "ball"], sort=False).spell_f1.mean().round(3).to_string())
    print("reference: true ground ball, touch 0.705")


def match():
    from games import half_video_blocks, GAME
    from fuse import load, SCALE
    z = np.load(os.path.join(ROOT, "data", "cleanup", "ball_3d.npz"))
    good = z["gap"] < 1.0
    ref = {f: i for i, f in enumerate(z["frame"].tolist()) if good[i]}
    acc = {}
    for half, a, b in half_video_blocks(GAME):
        n = b - a + 1
        v = load(a, n)
        interp = np.isfinite(v["ball_interp"][:, 0])
        final = np.where(interp[:, None], v["ball_interp"], v["ball_mfill"]) * SCALE
        met = v["metrica"][2] * SCALE
        k = np.array([i for i in range(n) if a + i in ref])
        idx = np.array([ref[a + i] for i in k])
        h, true = z["xyz"][idx, 2], z["xyz"][idx, :2]
        ests = [("ours final", final, np.zeros(n, bool)), ("Metrica", met, np.zeros(n, bool))]
        ests += [("straight dev {:g}".format(dv),) + straighten(final, dv) for dv in DEV_GRID]
        for name, est, ch in ests:
            e = np.linalg.norm(est[k] - true, axis=1)
            acc.setdefault(name, []).append((h, e, ch[k], np.full(len(k), ch.mean())))
    print("referee: 3D ball of the two cameras (rays within 1 m); errors in metres, p50 / p90")
    for name, parts in acc.items():
        h, e, ch, share = (np.concatenate(x) for x in zip(*parts))
        ok = np.isfinite(e)
        line = "{:<18}".format(name)
        for lo, hi in ((-1, 0.3), (0.3, 1), (1, 2), (2, 4), (4, 30)):
            m = ok & (h >= lo) & (h < hi)
            line += " | h {:.1f}-{:.0f}: {:5.2f} /{:5.1f}".format(lo, hi, *np.percentile(e[m], [50, 90]))
        if ch.any() or name.startswith("straight"):
            line += " | changed {:.1%} of frames, of them h>1 m {:.0%}".format(share.mean(), (h[ch] > 1).mean() if ch.any() else 0)
        print(line)


if __name__ == "__main__":
    {"dfl": dfl, "match": match}[sys.argv[1]]()
