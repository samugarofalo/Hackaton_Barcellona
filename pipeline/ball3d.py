"""Ball height from one camera: ballistic fit on the rays, checked on DFL (ball height known there).

The homography puts a ball in the air where its line of sight meets the grass, behind the real
ball. Each projected point G_t defines a ray from the camera centre C. Over a short window
(+/- HALF frames) two models are fitted:
  ground  the ball rolls: G_t follows a quadratic in time (x and y), residual = distance of G_t
          from the fitted curve;
  flight  the ball flies: p(t) = p0 + v0 t + g t^2 / 2 with g = (0, 0, -9.81); p(t) must lie on
          each ray, (p(t) - C) x d_t = 0 -- linear in p0, v0, so one least-squares solve per
          window; residual = distance of p(t) from the rays.
Flight wins where its residual is lower, the fitted height at the centre is >= MIN_Z and stays
>= -0.3 m across the window. The ball's ground position is then the fitted point's x, y (under
the ball) and its height z; otherwise G_t and z = 0.

    .venv/bin/python pipeline/ball3d.py
Simulates our camera on the DFL matches (angle from data/cleanup measurements, see CONTESTO.md),
reconstructs, and scores position error and the touch rules.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), HERE]

G = 9.81
HALF = 6
MIN_Z = 0.5


def _windows(a, half):
    """(F, 2*half+1, ...) centred windows, edges padded with NaN."""
    pad = np.full((half,) + a.shape[1:], np.nan)
    a = np.concatenate([pad, a, pad])
    idx = np.arange(len(a) - 2 * half)[:, None] + np.arange(2 * half + 1)[None]
    return a[idx]


def reconstruct(ground_xy, cam, half=HALF):
    """ground_xy (F,2) metres, projected ball (NaN = none); cam (3,) metres.
    Returns xy (F,2) estimated ground position under the ball, z (F,), flight (F,) bool."""
    F = len(ground_xy)
    t = (np.arange(2 * half + 1) - half) / 25.0
    Gw = _windows(ground_xy, half)                       # (F,W,2)
    ok = np.all(np.isfinite(Gw[..., 0]), axis=1)

    # ground model: quadratic in t, via the hat matrix of a fixed design
    X = np.c_[np.ones_like(t), t, t * t]
    hat = X @ np.linalg.pinv(X)
    fit = np.einsum("ij,fjc->fic", hat, np.nan_to_num(Gw))
    r_ground = np.sqrt(np.mean(np.sum((np.nan_to_num(Gw) - fit) ** 2, axis=2), axis=1))

    # flight model
    P = np.concatenate([np.nan_to_num(Gw), np.zeros(Gw.shape[:2] + (1,))], axis=2)   # (F,W,3)
    d = P - cam
    d /= np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-9)
    def cross_mat(v):
        z = np.zeros(v.shape[:-1])
        return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1),
                         np.stack([v[..., 2], z, -v[..., 0]], -1),
                         np.stack([-v[..., 1], v[..., 0], z], -1)], -2)       # a x v = -[v]x a ... see below
    Dx = cross_mat(d)                                   # [d]x, so [d]x a = d x a
    # (p - C) x d = 0  <=>  d x (p - C) = 0  <=>  [d]x p = [d]x C ; p = p0 + t v0 + 0.5 g t^2
    gvec = np.array([0, 0, -G])
    M = np.concatenate([Dx, Dx * t[None, :, None, None]], axis=3)            # (F,W,3,6)
    rhs = np.einsum("fwij,j->fwi", Dx, cam) - np.einsum("fwij,j->fwi", Dx, gvec) * (0.5 * t * t)[None, :, None]
    A = np.einsum("fwki,fwkj->fij", M, M) + 1e-9 * np.eye(6)
    b = np.einsum("fwki,fwk->fi", M, rhs)
    sol = np.linalg.solve(A, b[..., None])[..., 0]                           # (F,6)
    p0, v0 = sol[:, :3], sol[:, 3:]
    traj = p0[:, None] + v0[:, None] * t[None, :, None] + 0.5 * gvec * (t * t)[None, :, None]
    r_flight = np.sqrt(np.mean(np.sum(np.cross(traj - cam, d) ** 2, axis=2), axis=1))
    zmin = traj[..., 2].min(axis=1)

    flight = ok & (r_flight < r_ground) & (p0[:, 2] >= MIN_Z) & (zmin >= -0.3)
    xy = ground_xy.copy()
    xy[flight] = p0[flight, :2]
    z = np.where(flight, p0[:, 2], 0.0)
    return xy, z, flight


def main():
    import dfl_degrade as dd
    dd.ALIVE_GAP_SHARE = 0.0
    from dfl_bench import CONVERTED, SCALE, load, score
    from dfl_control import touches, nearest, run_rule, sight_touches
    from baseline import smooth

    rows = []
    for ang in (25.0, 35.0):
        H = 30.0
        cam = np.array([52.5, 34 - H / np.tan(np.radians(ang)), H])
        cot = 1 / np.tan(np.radians(ang))
        err = {"proj": [], "rec": [], "rec_air": [], "proj_air": []}
        cls = []
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
            xy, zhat, fl = reconstruct_multiscale(proj, cam)
            live = (alive == 1) & np.isfinite(proj[:, 0])
            air = live & (ztrue > 1)
            for key, est, m in (("proj", proj, live), ("rec", xy, live), ("proj_air", proj, air), ("rec_air", xy, air)):
                err[key].append(np.linalg.norm(est[m] - true[m], axis=1))
            cls.append((ztrue[live] > MIN_Z, fl[live]))
            Pq = P + rng.normal(0, 0.02, P.shape) / SCALE
            for name, est in (("projected", proj), ("reconstructed", xy)):
                b = est / SCALE
                near, dist = nearest(Pq, team, b)
                rows.append((ang, name, "touch", score(d, gt, run_rule(near, dist, dead, None, 0, touches(b, dist, 2.0)))[1]))
                tc, dm, kk = sight_touches(b, Pq, 2.0, cam[:2], 2.0 * cot)
                nt = np.where(np.isfinite(dm), team[np.arange(len(kk)), kk], "")
                rows.append((ang, name, "touch across sight", score(d, gt, run_rule(nt, dm, dead, None, 0, tc))[1]))
                if name == "reconstructed":
                    reach = zhat < 2.5  # a ball higher than a header is touched by nobody
                    tc2 = touches(b, dist, 2.0) & reach
                    rows.append((ang, name, "touch, ball reachable", score(d, gt, run_rule(near, dist, dead, None, 0, tc2))[1]))
            print(os.path.basename(path), ang, flush=True)
        a = np.concatenate([c[0] for c in cls]); f = np.concatenate([c[1] for c in cls])
        print("angle {:.0f}: flight called {:.1%} of live frames; of balls above {} m called flight {:.1%}; "
              "of flight calls, ball really above {} m {:.1%}".format(ang, f.mean(), MIN_Z, f[a].mean(), MIN_Z, a[f].mean()))
        for key, v in err.items():
            v = np.concatenate(v)
            print("  position error {:<9} p50 {:.2f} m  p90 {:.2f} m".format(key, *np.percentile(v, [50, 90])))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["angle", "ball", "rule", "spell_f1"])
    print(df.groupby(["angle", "ball", "rule"], sort=False).spell_f1.mean().round(3).to_string())
    print("reference: true ground ball, touch 0.705")



SPLIT_DV = 5.0      # m/s: sharp change of projected velocity = kick, header, landing; gravity alone
                    # bends the projection by ~17 m/s^2, i.e. < 3 m/s over the 4-frame difference
MIN_SEG = 8


def _fit_flight(G, t, cam):
    """Least squares flight on the rays through G (n,2) at times t; returns (p0, v0, traj (n,3))."""
    P = np.c_[G, np.zeros(len(G))]
    d = P - cam
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    z = np.zeros(len(d))
    Dx = np.stack([np.stack([z, -d[:, 2], d[:, 1]], -1), np.stack([d[:, 2], z, -d[:, 0]], -1),
                   np.stack([-d[:, 1], d[:, 0], z], -1)], -2)
    gvec = np.array([0, 0, -G_ACC])
    M = np.concatenate([Dx, Dx * t[:, None, None]], axis=2).reshape(-1, 6)
    rhs = (Dx @ cam - (Dx @ gvec) * (0.5 * t * t)[:, None]).reshape(-1)
    sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
    traj = sol[:3] + sol[3:] * t[:, None] + 0.5 * gvec * (t * t)[:, None]
    return traj


G_ACC = G


def project_to_ground(p, cam):
    k = cam[2] / np.maximum(cam[2] - p[:, 2], 1e-3)
    return cam[:2] + (p[:, :2] - cam[:2]) * k[:, None]


def reconstruct_segments(ground_xy, cam, split_dv=SPLIT_DV, min_seg=MIN_SEG, ratio=0.7):
    """Segment-wise version of reconstruct(): cut the projected track at sharp velocity changes
    and gaps, then per segment choose ground (quadratic in x, y) or flight (parabola on the rays),
    comparing both residuals on the ground plane."""
    F = len(ground_xy)
    xy, z, flight = ground_xy.copy(), np.zeros(F), np.zeros(F, bool)
    import pandas as pd
    sm = pd.DataFrame(ground_xy).rolling(5, center=True, min_periods=1).mean().to_numpy()
    v = np.gradient(sm, axis=0) * 25
    dv = np.zeros(F)
    dv[2:-2] = np.linalg.norm(v[4:] - v[:-4], axis=1)
    dv = np.nan_to_num(dv, nan=99.0)
    peak = (dv >= np.roll(dv, 1)) & (dv >= np.roll(dv, -1)) & (dv > split_dv)
    cut = peak | ~np.isfinite(ground_xy[:, 0])
    edges = np.r_[0, np.flatnonzero(cut), F]
    for a, b in zip(edges[:-1], edges[1:]):
        a = a + 1 if a > 0 and cut[a] else a
        if b - a < min_seg or not np.all(np.isfinite(ground_xy[a:b, 0])):
            continue
        Gs = ground_xy[a:b]
        t = (np.arange(b - a) - (b - a) / 2) / 25.0
        X = np.c_[np.ones_like(t), t, t * t]
        res_g = np.sqrt(np.mean(np.sum((Gs - X @ np.linalg.lstsq(X, Gs, rcond=None)[0]) ** 2, axis=1)))
        traj = _fit_flight(Gs, t, cam)
        res_f = np.sqrt(np.mean(np.sum((project_to_ground(traj, cam) - Gs) ** 2, axis=1)))
        if res_f < ratio * res_g and traj[:, 2].max() >= MIN_Z and traj[:, 2].min() >= -0.3:
            xy[a:b], z[a:b], flight[a:b] = traj[:, :2], np.clip(traj[:, 2], 0, None), True
    return xy, z, flight


def reconstruct_multiscale(ground_xy, cam, lengths=(9, 15, 25, 41), max_res=0.06, ground_factor=2.0):
    """Sliding windows of several lengths (stride L/3). A window is a flight when the parabola on
    the rays leaves a ground-plane residual <= max_res (about twice our 0.03 m noise), the ground
    quadratic leaves >= ground_factor times more, and the fitted height is plausible. Each frame
    takes the accepted window with the smallest residual."""
    F = len(ground_xy)
    xy, z, flight = ground_xy.copy(), np.zeros(F), np.zeros(F, bool)
    best = np.full(F, np.inf)
    finite = np.isfinite(ground_xy[:, 0])
    run = np.cumsum(~finite)
    for L in lengths:
        t = (np.arange(L) - L / 2) / 25.0
        X = np.c_[np.ones_like(t), t, t * t]
        hat = X @ np.linalg.pinv(X)
        for a in range(0, F - L, max(L // 3, 1)):
            b = a + L
            if run[b - 1] - (run[a - 1] if a else 0):
                continue
            Gs = ground_xy[a:b]
            rg = np.sqrt(np.mean(np.sum((Gs - hat @ Gs) ** 2, axis=1)))
            if rg < ground_factor * 0.03:
                continue  # a clean ground track: nothing to explain
            tr = _fit_flight(Gs, t, cam)
            rf = np.sqrt(np.mean(np.sum((project_to_ground(tr, cam) - Gs) ** 2, axis=1)))
            if rf <= max_res and rg >= ground_factor * rf and tr[:, 2].max() >= MIN_Z and tr[:, 2].min() >= -0.3:
                sel = rf < best[a:b]
                idx = np.arange(a, b)[sel]
                xy[idx], z[idx], flight[idx], best[idx] = tr[sel, :2], np.clip(tr[sel, 2], 0, None), True, rf
    return xy, z, flight


if __name__ == "__main__":
    main()
