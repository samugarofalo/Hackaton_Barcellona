"""Possession rules without a long vote window, scored on the DFL matches under Metrica-like errors.

Rules (each returns H/A/D per frame; D where the dead-ball source says so):
  nearest_w50   starter baseline, for reference.
  control       possession changes only when a player of the other team stays within R metres of
                the ball for N consecutive frames; during a pass it stays with the passer's team.
  touch         a touch is a discontinuity in the ball's motion (sudden stop or turn): the change
                of ball velocity over +/- 2 frames exceeds T m/s at a local peak, with a player
                within TOUCH_M. The touching player's team takes possession from that frame.
  touch+control either signal can hand possession over.

Conditions (errors added as in dfl_degrade.py, seeds averaged):
  clean, ball noise 0.5 / 1 m, carrier_swap 1% / 5%, carrier_drop 5%; each with the dead ball
  from the ground truth ("oracleD", a ceiling) and with the Metrica-like ball, where D is simply
  "the ball is missing" ("ballgap", what we can actually do).

    .venv/bin/python pipeline/dfl_control.py
Writes data/external/bassek2025/control.csv.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), HERE]
from baseline import smooth  # noqa: E402
from dfl_bench import CONVERTED, SCALE, load, score  # noqa: E402
from dfl_degrade import degrade, metrica_like_ball  # noqa: E402

SEEDS = (0, 1)
TOUCH_M = 2.0
SMOOTH_BALL = 5  # frames of moving average before differentiating (noise)


def nearest(players, team, ball):
    """(team code 'H'/'A' or '' when nothing, distance in m) of the player nearest the ball."""
    d = np.linalg.norm((players - ball[:, None]) * SCALE, axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    k = d.argmin(1)
    dmin = d.min(1)
    t = team[np.arange(len(k)), k]
    return np.where(np.isfinite(dmin), t, ""), dmin


def touches(ball, dmin, T):
    """Boolean per frame: sudden change of ball velocity (stop or turn) with a player close."""
    b = pd.DataFrame(ball * SCALE).rolling(SMOOTH_BALL, center=True, min_periods=1).mean().values
    v = np.gradient(b, axis=0) * 25  # m/s
    dv = np.full(len(b), 0.0)
    dv[2:-2] = np.linalg.norm(v[4:] - v[:-4], axis=1)
    dv = np.nan_to_num(dv)
    peak = (dv >= np.roll(dv, 1)) & (dv >= np.roll(dv, -1))
    return (dv > T) & peak & (dmin < TOUCH_M)


def run_rule(near, dmin, dead, R=1.0, N=3, touch=None):
    """Sequential state: who has the ball. dead -> D. touch (bool array) or None."""
    out = np.empty(len(near), "<U1")
    cur, run = "", 0
    for i in range(len(near)):
        if dead[i]:
            cur, run = "", 0
            out[i] = "D"
            continue
        t = near[i]
        if touch is not None and touch[i] and t:
            cur, run = t, 0
        elif R is not None and t and dmin[i] < R and t != cur:
            run += 1
            if run >= N or not cur:
                cur, run = t, 0
        else:
            run = 0
        out[i] = cur or t or "D"
    return out


CONDITIONS = [("clean", None, 0), ("ball_noise", "ball_noise", 0.5), ("ball_noise", "ball_noise", 1.0),
              ("carrier_swap", "carrier_swap", 0.01), ("carrier_swap", "carrier_swap", 0.05),
              ("carrier_drop", "carrier_drop", 0.05)]


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(CONVERTED, "*.npz"))):
        d, gt = load(path)
        players, alive = d["players"], d["alive"]
        team = np.broadcast_to(d["team"], players.shape[:2])
        match = os.path.basename(path)[:-4]
        for seed in SEEDS:
            gap_ball = metrica_like_ball(np.random.default_rng(seed), d["ball"][:, :2], alive)
            for name, kind, level in CONDITIONS:
                for dead_src in ("oracleD", "ballgap"):
                    ball = d["ball"][:, :2] if dead_src == "oracleD" else gap_ball
                    p, t, b = (players, team, ball) if kind is None else degrade(
                        kind, level, np.random.default_rng(seed + 100), players, team, ball, alive)
                    near, dmin = nearest(p, t, b)
                    dead = (gt == "D") if dead_src == "oracleD" else ~np.isfinite(b[:, 0])
                    codes = np.where(near == "H", 0, np.where(near == "A", 1, -1)).astype(np.int8)
                    w50 = smooth(np.where(dead, -1, codes), 50)
                    w50[dead] = "D"  # as the other rules: no possession while dead
                    preds = {"nearest_w50": w50,
                             "control_R1_N3": run_rule(near, dmin, dead, 1.0, 3)}
                    for T in (3.0, 5.0, 8.0):
                        tc = touches(b, dmin, T)
                        preds["touch_T{:g}".format(T)] = run_rule(near, dmin, dead, None, 0, tc)
                        preds["touch_T{:g}+control".format(T)] = run_rule(near, dmin, dead, 1.0, 3, tc)
                    for rule, pred in preds.items():
                        rows.append((match, seed, "{}_{:g}".format(name, level), dead_src, rule,
                                     *score(d, gt, pred)))
        print(match, "done", flush=True)
    df = pd.DataFrame(rows, columns=["match", "seed", "condition", "dead", "rule", "macro_f1", "spell_f1"])
    df.to_csv(os.path.join(os.path.dirname(CONVERTED), "control.csv"), index=False)
    for dead_src in ("oracleD", "ballgap"):
        t = df[df.dead == dead_src].pivot_table(index="rule", columns="condition", values="spell_f1")
        print("\nspellF1, dead ball from", dead_src)
        print(t.round(3).to_string())


if __name__ == "__main__":
    main()


def touches_across_gaps(ball, dmin, T, src, edge=3):
    """touches(), but where the ball was interpolated (src == 1) the straight line hides the touch:
    drop the touches the line's corners would fire, and instead mark one touch inside the gap --
    at its frame nearest a player -- when the velocity out of the gap differs from the velocity
    into it by more than T and a player came within TOUCH_M during the gap."""
    tc = touches(ball, dmin, T)
    b = ball * SCALE
    gap = src == 1
    e = np.diff(np.r_[0, gap.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):  # gap = [a, z)
        lo, hi = max(a - edge - 2, 0), min(z + edge + 2, len(b))
        tc[lo:hi] = False
        if a - edge - 1 < 0 or z + edge >= len(b):
            continue
        v_in = (b[a - 1] - b[a - 1 - edge]) * 25 / edge
        v_out = (b[z + edge] - b[z]) * 25 / edge
        if np.all(np.isfinite(v_in)) and np.all(np.isfinite(v_out)) and np.linalg.norm(v_out - v_in) > T:
            k = a + int(np.nanargmin(np.where(np.isfinite(dmin[a:z]), dmin[a:z], np.inf)))
            if dmin[k] < TOUCH_M:
                tc[k] = True
    return tc


def sight_touches(ball, players, T, cam_xy, reach_m, smooth_n=SMOOTH_BALL):
    """Touch rule immune to the ball's height. Seen from one camera, height only moves the projected
    ball along the line of sight (towards / away from the camera's ground point cam_xy, metres).
    So: (1) only the part of the velocity change ACROSS the line of sight counts; (2) the toucher is
    the player nearest the segment of that line where the real ball can be -- from the projected
    point back towards the camera by reach_m (the displacement of a ball at reachable height).
    Returns (touch bool per frame, distance per frame used for attribution, nearest index)."""
    b = pd.DataFrame(ball * SCALE).rolling(smooth_n, center=True, min_periods=1).mean().values
    u = b - cam_xy
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = np.gradient(b, axis=0) * 25
    dvec = np.full_like(b, np.nan)
    dvec[2:-2] = v[4:] - v[:-4]
    across = dvec - np.sum(dvec * u, axis=1, keepdims=True) * u
    dv = np.nan_to_num(np.linalg.norm(across, axis=1))
    peak = (dv >= np.roll(dv, 1)) & (dv >= np.roll(dv, -1))
    p = players * SCALE
    a = b[:, None, :]                       # segment from the projected point ...
    c = (b - u * reach_m)[:, None, :]       # ... back towards the camera
    ab = c - a
    t = np.clip(np.sum((p - a) * ab, axis=2) / np.maximum(np.sum(ab * ab, axis=2), 1e-9), 0, 1)
    d = np.linalg.norm(p - (a + t[..., None] * ab), axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    k = d.argmin(1)
    dmin = d.min(1)
    return (dv > T) & peak & (dmin < TOUCH_M), dmin, k
