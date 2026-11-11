"""Fair ball comparison on DFL: our ball as one camera really sees it vs Metrica's.

dfl_method4.py gave 'ours' the true ball + jitter, i.e. no height error, while Metrica's ball is
good in the air (CONTESTO.md section 13). Here 'ours' is the true ball projected from a camera like
ours (height moves it along the line of sight, as in ball3d.py), + JITTER_M, with the gap runs
measured on our match, gaps <= 50 frames interpolated and longer ones filled with Metrica.
Metrica = the true ball smoothed over 12 frames (its measured blur), no projection error.
Players: DFL + 0.02 m (our measured jitter) + 1% carrier swap. Dead ball: where the Metrica-like
ball is missing, no extra gaps in live play (corrected simulation, CONTESTO.md section 6).

Each ball is scored with: baseline +-50, control (R 1 m, 3 frames), touch T 2 m/s (no touches
near Metrica fill), touch across the line of sight + control.

    .venv/bin/python pipeline/dfl_ball_fair.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), HERE]
import dfl_degrade as dd  # noqa: E402
dd.ALIVE_GAP_SHARE = 0.0
from baseline import smooth  # noqa: E402
from dfl_bench import CONVERTED, SCALE, load, score  # noqa: E402
from dfl_control import touches, nearest, run_rule, sight_touches  # noqa: E402
from dfl_method4 import runs_mask, fill, METRICA_SMOOTH, JITTER_M  # noqa: E402

PLAYER_NOISE_M = 0.02
ANGLES = (25.0, 35.0)
SEEDS = (0, 1)
CAM_H = 30.0


def rules(b, p, t, dead, cam, cot, src):
    near, dist = nearest(p, t, b)
    codes = np.where(near == "H", 0, np.where(near == "A", 1, -1))
    w50 = smooth(np.where(dead, -1, codes), 50).copy()
    w50[dead] = "D"
    out = {"baseline ±50": w50, "control": run_rule(near, dist, dead, 1.0, 3)}
    tc = touches(b, dist, 2.0)
    if src is not None:
        tc &= ~pd.Series(src == 2).rolling(5, center=True, min_periods=1).max().to_numpy().astype(bool)
    out["touch T2"] = run_rule(near, dist, dead, None, 0, tc)
    sc, sd, sk = sight_touches(b, p, 2.0, cam[:2], 2.0 * cot)
    st = np.where(np.isfinite(sd), t[np.arange(len(sk)), sk], "")
    out["touch across sight"] = run_rule(st, sd, dead, None, 0, sc)
    out["touch across sight + control"] = run_rule(st, sd, dead, 1.0, 3, sc)
    return out


def main():
    r = np.load(os.path.join(ROOT, "data", "cleanup", "ball_runs.npz"))
    rows = []
    for path in sorted(glob.glob(os.path.join(CONVERTED, "*.npz"))):
        d, gt = load(path)
        players, alive = d["players"], d["alive"]
        team = np.broadcast_to(d["team"], players.shape[:2])
        true = d["ball"][:, :2] * SCALE
        z = np.clip(d["ball"][:, 2], 0, None)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            gap_ball = dd.metrica_like_ball(rng, d["ball"][:, :2], alive)
            dead = ~np.isfinite(gap_ball[:, 0])
            metrica = pd.DataFrame(gap_ball).rolling(METRICA_SMOOTH, center=True, min_periods=1).mean().to_numpy(copy=True)
            metrica[dead] = np.nan
            seen = runs_mask(rng, len(alive), r["seen"], r["gaps"])
            jit = rng.normal(0, JITTER_M, true.shape)
            p = players + rng.normal(0, PLAYER_NOISE_M, players.shape) / SCALE
            p, t, _ = dd.degrade("carrier_swap", 0.01, np.random.default_rng(seed + 100), p, team, metrica, alive)
            for ang in ANGLES:
                cam = np.array([52.5, 34 - CAM_H / np.tan(np.radians(ang)), CAM_H])
                cot = 1 / np.tan(np.radians(ang))
                k = cam[2] / np.maximum(cam[2] - z, 1.0)
                balls = {}
                for name, xy in (("ours, true height-free ball (old sim)", true), ("ours, projected (real)", cam[:2] + (true - cam[:2]) * k[:, None])):
                    o = (xy + jit) / SCALE
                    o[~seen | dead] = np.nan
                    balls[name] = fill(o, metrica, 50)
                balls["Metrica only"] = (metrica, np.zeros(len(alive), int))
                for name, (b, src) in balls.items():
                    for rule, pred in rules(b, p, t, dead, cam, cot, src if name != "Metrica only" else None).items():
                        rows.append((ang, seed, name, rule, score(d, gt, pred)[1]))
        print(os.path.basename(path), flush=True)
    df = pd.DataFrame(rows, columns=["angle", "seed", "ball", "rule", "spell_f1"])
    print(df.pivot_table(index="rule", columns="ball", values="spell_f1", aggfunc="mean").round(3).to_string())
    print()
    print(df.pivot_table(index=["angle", "rule"], columns="ball", values="spell_f1", aggfunc="mean").round(3).to_string())
    m = df.groupby(["angle", "seed", "ball", "rule"]).spell_f1.mean().unstack("ball")
    print("\nprojected ours - Metrica, per (angle, seed), min / max:")
    print((m["ours, projected (real)"] - m["Metrica only"]).groupby("rule").agg(["min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
