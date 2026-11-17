"""How much does each kind of tracking error cost possession scores? Measured on the DFL matches.

The DFL tracking is clean. We break it on purpose, one kind of error at a time and at several
strengths, and score the starter baseline (nearest player's team, D where the ball is missing,
majority vote +/- 50 frames) with the competition's metric. The drop from the "Metrica-like"
reference tells how much fixing that error in the Metrica ATD could be worth. Other games, so
all of this is allowed.

Reference ("metrica_like"): the ball is missing over most dead-ball stretches, as in the Metrica
ATD (41% of graded frames missing, almost all with the game stopped): each dead stretch longer
than 1 s loses the ball with probability BALL_GONE_DEAD, plus short gaps in play.

Errors, each added on top of the reference, in episodes (consecutive frames), seeds averaged:
  ball_noise_<s>m       ball position + Gaussian noise, sigma s metres
  carrier_drop_<p>      the player nearest the ball disappears for 1-4 s, p of alive time
  carrier_swap_<p>      ... carries the other team's label instead (a duel mis-tracked)
  carrier_frozen_<p>    ... stays frozen where he was (the tracker lost him)
  player_drop_<p>       random players disappear for 1-6 s, p of all player-frames
  frozen_<p>            random players freeze in place for 2-20 s, p of player-frames
  team_swap_<p>         random players carry the other team's label for 1-6 s, p of player-frames
  player_noise_<s>m     every player + Gaussian noise, sigma s metres

    python pipeline/dfl_degrade.py
Writes data/external/bassek2025/degrade.csv.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), HERE]
from baseline import smooth, HOME, AWAY, NOBODY  # noqa: E402
from dfl_bench import CONVERTED, SCALE, load, score  # noqa: E402

BALL_GONE_DEAD = 0.9
ALIVE_GAP_SHARE = 0.02
SEEDS = (0, 1)
FPS = 25


def runs(mask):
    edge = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def episodes(rng, n_frames, share, lo_s, hi_s):
    """Random [start, stop) episodes covering about `share` of n_frames."""
    out, covered = [], 0
    while covered < share * n_frames:
        length = int(rng.uniform(lo_s, hi_s) * FPS)
        start = int(rng.integers(0, max(1, n_frames - length)))
        out.append((start, start + length))
        covered += length
    return out


def metrica_like_ball(rng, ball, alive):
    ball = ball.copy()
    for a, z in runs(alive == 0):
        if z - a > FPS and rng.random() < BALL_GONE_DEAD:
            ball[a:z] = np.nan
    for a, z in episodes(rng, len(ball), ALIVE_GAP_SHARE, 0.2, 2.0):
        ball[a:z][alive[a:z] == 1] = np.nan
    return ball


def baseline(players, team, ball):
    dist = np.linalg.norm(players * SCALE - (ball * SCALE)[:, None], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    k = dist.argmin(axis=1)
    codes = np.where(team[np.arange(len(k)), k] == "H", HOME, AWAY).astype(np.int8)
    codes[~np.isfinite(dist.min(axis=1))] = NOBODY
    return smooth(codes, 50)


def degrade(kind, level, rng, players, team, ball, alive):
    p, t, b = players.copy(), team.copy(), ball.copy()
    F, P = p.shape[:2]
    on = np.isfinite(p[:, :, 0])
    if kind == "ball_noise":
        b = b + rng.normal(0, level, b.shape) / SCALE
    elif kind == "player_noise":
        p = p + rng.normal(0, level, p.shape) / SCALE
    elif kind in ("carrier_drop", "carrier_swap", "carrier_frozen"):
        # Errors where tracking really fails: on the player at the ball (overlaps, duels).
        d = np.linalg.norm((p - b[:, None]) * SCALE, axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        for a, z in episodes(rng, int(alive.sum()), level, 1, 4):
            frames = np.flatnonzero(alive == 1)[a:z]
            if len(frames) and np.isfinite(d[frames[0]].min()):
                k = d[frames[0]].argmin()
                frames = frames[on[frames, k]]
                if not len(frames):
                    continue
                if kind == "carrier_drop":
                    p[frames, k] = np.nan
                elif kind == "carrier_swap":
                    t[frames, k] = "A" if team[0, k] == "H" else "H"
                else:
                    p[frames, k] = p[frames[0], k]
    else:
        lo, hi = {"player_drop": (1, 6), "frozen": (2, 20), "team_swap": (1, 6)}[kind]
        for a, z in episodes(rng, F * P, level, lo, hi):
            k = rng.integers(0, P)
            a, z = a % F, min(a % F + (z - a), F)
            sel = np.arange(a, z)[on[a:z, k]]
            if not len(sel):
                continue
            if kind == "player_drop":
                p[sel, k] = np.nan
            elif kind == "frozen":
                p[sel, k] = p[sel[0], k]
            else:
                t[sel, k] = "A" if team[0, k] == "H" else "H"
    return p, t, b


GRID = [("ball_noise", s) for s in (1.0, 2.0, 4.0)] + \
       [("carrier_drop", q) for q in (0.02, 0.05, 0.10)] + \
       [("carrier_swap", q) for q in (0.02, 0.05, 0.10)] + \
       [("carrier_frozen", q) for q in (0.02, 0.05, 0.10)] + \
       [("player_drop", q) for q in (0.05, 0.10)] + \
       [("frozen", q) for q in (0.02, 0.05)] + \
       [("team_swap", q) for q in (0.01, 0.03)] + \
       [("player_noise", s) for s in (0.5, 1.0)]


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(CONVERTED, "*.npz"))):
        d, gt = load(path)
        players, alive = d["players"], d["alive"]
        team = np.broadcast_to(d["team"], players.shape[:2])
        match = os.path.basename(path)[:-4]
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            ball = metrica_like_ball(rng, d["ball"][:, :2], alive)
            ref = score(d, gt, baseline(players, team, ball))
            rows.append((match, seed, "metrica_like", 0, *ref))
            for kind, level in GRID:
                p, t, b = degrade(kind, level, np.random.default_rng(seed + 100), players, team, ball, alive)
                rows.append((match, seed, kind, level, *score(d, gt, baseline(p, t, b))))
        print(match, "done", flush=True)
    df = pd.DataFrame(rows, columns=["match", "seed", "error", "level", "macro_f1", "spell_f1"])
    df.to_csv(os.path.join(os.path.dirname(CONVERTED), "degrade.csv"), index=False)
    ref = df[df.error == "metrica_like"].groupby(["match", "seed"])[["macro_f1", "spell_f1"]].first()
    df = df.join(ref, on=["match", "seed"], suffixes=("", "_ref"))
    df["d_macro"] = df.macro_f1 - df.macro_f1_ref
    df["d_spell"] = df.spell_f1 - df.spell_f1_ref
    out = df.groupby(["error", "level"]).agg(macro=("macro_f1", "mean"), spell=("spell_f1", "mean"),
                                              d_macro=("d_macro", "mean"), d_spell=("d_spell", "mean"),
                                              d_spell_min=("d_spell", "min"), d_spell_max=("d_spell", "max"))
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
