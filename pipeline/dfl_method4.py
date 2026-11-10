"""Method 4 + final ball, simulated on the DFL matches, scored with the touch rule alone.

Inputs shaped like ours (other games, so tuning here is allowed):
  dead       ball missing where the Metrica-like ball is missing (dfl_degrade.metrica_like_ball).
  Metrica    that ball smoothed over METRICA_SMOOTH frames (the blur measured on the real ATD).
  ours       the true DFL ball + JITTER_M noise, present/absent in runs drawn from the run lengths of
             our detector on the provided match (data/cleanup/ball_runs.npz, aggregate only).
  players    DFL + PLAYER_NOISE_M, plus carrier_swap 1% (measured level, carrier_fix.py).
Ball variants: ours only; gaps <= G interpolated and longer ones filled with Metrica, for several G;
everything filled with Metrica. Touches are not searched where the ball is Metrica's (its blur
and the switch between sources would invent them).
Rule: touch only (dfl_control.run_rule without the control part): at a sudden change of ball
velocity with a player within TOUCH_M, possession goes to that player's team.

    .venv/bin/python pipeline/dfl_method4.py
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
from dfl_degrade import metrica_like_ball, degrade  # noqa: E402
from dfl_control import touches, nearest, run_rule  # noqa: E402

METRICA_SMOOTH, JITTER_M, PLAYER_NOISE_M = 12, 0.03, 0.2  # jitter: residual from a 5-frame mean, p50 0.03 m on our track
GAPS = (0, 2, 5, 13, 25, 50)
SEEDS = (0, 1)


def runs_mask(rng, n, seen, gaps):
    """True where 'our' ball is seen, alternating runs drawn from the measured lengths."""
    m = np.zeros(n, bool)
    i, on = 0, True
    while i < n:
        L = int(rng.choice(seen if on else gaps))
        m[i:i + L] = on
        i, on = i + L, not on
    return m


def fill(ours, metrica, max_gap):
    df = pd.DataFrame(ours)
    run = df[0].isna().astype(int).groupby(df[0].notna().cumsum()).transform("sum").to_numpy()
    interp = df.interpolate(limit_area="inside").to_numpy(copy=True)
    long_gap = np.isnan(ours[:, 0]) & (run > max_gap)
    interp[long_gap] = metrica[long_gap]
    src = np.where(np.isfinite(ours[:, 0]), 0, np.where(long_gap, 2, 1))  # 0 ours, 1 interp, 2 metrica
    return interp, src


def main():
    r = np.load(os.path.join(ROOT, "data", "cleanup", "ball_runs.npz"))
    rows = []
    for path in sorted(glob.glob(os.path.join(CONVERTED, "*.npz"))):
        d, gt = load(path)
        players, alive = d["players"], d["alive"]
        team = np.broadcast_to(d["team"], players.shape[:2])
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            gap_ball = metrica_like_ball(rng, d["ball"][:, :2], alive)
            dead = ~np.isfinite(gap_ball[:, 0])
            metrica = pd.DataFrame(gap_ball).rolling(METRICA_SMOOTH, center=True, min_periods=1).mean().to_numpy(copy=True)
            metrica[dead] = np.nan
            ours = d["ball"][:, :2] + rng.normal(0, JITTER_M, (len(alive), 2)) / SCALE
            ours[~runs_mask(rng, len(alive), r["seen"], r["gaps"])] = np.nan
            ours[dead] = np.nan
            p = players + rng.normal(0, PLAYER_NOISE_M, players.shape) / SCALE
            p, t, _ = degrade("carrier_swap", 0.01, np.random.default_rng(seed + 100), p, team, metrica, alive)

            near_m, dm = nearest(p, t, metrica)
            codes = np.where(near_m == "H", 0, np.where(near_m == "A", 1, -1))
            w50 = smooth(np.where(dead, -1, codes), 50).copy()
            w50[dead] = "D"
            rows.append((seed, "baseline ±50 (Metrica ball)", *score(d, gt, w50)))

            variants = [("our ball only", ours, np.zeros(len(alive), int))]
            variants += [("gaps <= {} interpolated, rest Metrica".format(g), *fill(ours, metrica, g)) for g in GAPS[1:]]
            variants += [("all gaps Metrica", *fill(ours, metrica, 0))]
            for name, b, src in variants:
                near, dist = nearest(p, t, b)
                for T in (2.0, 3.0, 5.0):
                    tc = touches(b, dist, T)
                    near_src = pd.Series(src == 2).rolling(5, center=True, min_periods=1).max().to_numpy().astype(bool)
                    tc &= ~near_src
                    rows.append((seed, "{}, touch T{:g}".format(name, T),
                                 *score(d, gt, run_rule(near, dist, dead, None, 0, tc))))
        print(os.path.basename(path), flush=True)
    df = pd.DataFrame(rows, columns=["seed", "variant", "macro_f1", "spell_f1"])
    out = df.groupby("variant", sort=False)[["spell_f1", "macro_f1"]].agg(["mean", "min", "max"]).round(3)
    print(out.to_string())

    # How wrong is a straight line across a gap of a given length? (DFL ball, alive play)
    errs = {}
    for path in sorted(glob.glob(os.path.join(CONVERTED, "*.npz")))[:3]:
        d, _ = load(path)
        b = d["ball"][:, :2] * SCALE
        live = np.flatnonzero(d["alive"] == 1)
        rng = np.random.default_rng(0)
        for L in (2, 5, 13, 25, 50):
            starts = rng.choice(live[:-L - 1], 3000)
            ok = d["alive"][starts + L] == 1
            s = starts[ok]
            k = rng.integers(1, L + 1, len(s))
            lin = b[s] + (b[s + L + 1] - b[s]) * (k / (L + 1))[:, None]
            errs.setdefault(L, []).append(np.linalg.norm(lin - b[s + k], axis=1))
    for L, e in errs.items():
        e = np.concatenate(e)
        print("gap of {:>2} frames: straight-line error p50 {:.2f} m, p90 {:.2f} m".format(L, *np.percentile(e, [50, 90])))


if __name__ == "__main__":
    main()
