"""One simulation of "the provided tracking (Metrica ATD) on a DFL match", shared by B1, B2, C1, A1.

Why one. Each module used to degrade the DFL matches in its own way, and all of them assumed that
the provided ball goes missing only while the game is stopped. DATA.md says otherwise (the ball is
also lost in play: occlusions, crowded boxes, balls in the air) and a count on the public window
agrees, so "ball missing" is no longer taken as "ball dead" anywhere: it is one piece of evidence.

Inputs are the provided tracking only: players without identity, their team, one ball. No ball of
our own detector, no ball_runs.npz.

Levels. How bad the provided tracking really is cannot be measured on the provided match without
labels, and the public window must not be used to tune. So nothing is fitted: three levels of
round numbers bracket it, and a choice is kept only if it holds on all three.

                         easy    medium   hard
    gone_dead            0.90    0.75     0.60    share of stoppages > 1 s that lose the ball
    edge_s               0       1        2       s: the gap starts up to this late into the
                                                  stoppage and ends up to half of it early
    alive_gap            0       0.04     0.08    share of live play with the ball missing ...
    gap_s                -       0.2-5    0.2-10  ... in gaps of this length, s
    noise_m              0.02    0.10     0.20    Gaussian noise on every player, m
    drop                 0       0.05     0.10    share of player-frames missing (1-6 s episodes)
    track_s              -       30       10      mean life of a track before it moves to another
                                                  slot, s (slots are not persons, DATA.md)
"easy" is close to what the modules assumed before. $DFL_SIM picks the level (default medium).

    from dfl_sim import inputs
    s = inputs(d, seed)      # players (F,S,2), team (F,S), ball (F,2), missing (F,)
"""

import os

import numpy as np
import pandas as pd

from dfl_bench import SCALE
from dfl_degrade import FPS, degrade, episodes, runs

METRICA_SMOOTH = (
    12  # frames: the provided ball is as blurred as a 12-frame mean (CONTESTO.md)
)
CARRIER_SWAP = 0.01  # share of live play with the wrong team on the player at the ball
N_SLOTS = 46  # track slots of the provided tactical ATD (DATA.md)
MIN_TRACK = FPS  # frames: no piece of a track shorter than 1 s
MASK_GAPS = np.array(
    [5, 12, 25, 50, 125]
)  # frames: ball-hiding blocks for C1's training
LEVELS = {
    "easy": dict(
        gone_dead=0.9,
        edge_s=0.0,
        alive_gap=0.0,
        gap_s=(0.2, 2.0),
        noise_m=0.02,
        drop=0.0,
        track_s=None,
    ),
    "medium": dict(
        gone_dead=0.75,
        edge_s=1.0,
        alive_gap=0.04,
        gap_s=(0.2, 5.0),
        noise_m=0.1,
        drop=0.05,
        track_s=30.0,
    ),
    "hard": dict(
        gone_dead=0.6,
        edge_s=2.0,
        alive_gap=0.08,
        gap_s=(0.2, 10.0),
        noise_m=0.2,
        drop=0.10,
        track_s=10.0,
    ),
}


def level():
    name = os.environ.get("DFL_SIM") or "medium"
    if name not in LEVELS:
        raise ValueError(
            "DFL_SIM must be one of {}, not {!r}".format(sorted(LEVELS), name)
        )
    return name


def metrica_ball(rng, ball, alive, lv):
    """The true ball with the gaps of the provided one: most stoppages (edges late), some live play."""
    ball = ball.copy()
    edge = int(lv["edge_s"] * FPS)
    for a, z in runs(alive == 0):
        if z - a > FPS and rng.random() < lv["gone_dead"]:
            a2 = a + int(rng.integers(0, edge + 1))
            z2 = z - int(rng.integers(0, edge // 2 + 1))
            if z2 > a2:
                ball[a2:z2] = np.nan
    if lv["alive_gap"]:
        for a, z in episodes(rng, len(ball), lv["alive_gap"], *lv["gap_s"]):
            ball[a:z][alive[a:z] == 1] = np.nan
    return ball


def reslot(rng, players, team, track_s, n_slots=N_SLOTS):
    """Tracks without identity: every player's track is cut at random times (mean life track_s) and
    each piece goes to a slot that is free then. Returns players (F,S,2), team (F,S), '' = empty."""
    F, P = players.shape[:2]
    on = np.isfinite(players[:, :, 0])
    pieces = []
    for k in range(P):
        for a, z in runs(on[:, k]):
            while a < z:
                cut = min(z, a + max(MIN_TRACK, int(rng.exponential(track_s * FPS))))
                pieces.append((a, cut, k))
                a = cut
    out = np.full((F, n_slots, 2), np.nan, players.dtype)
    tm = np.full((F, n_slots), "", "<U1")
    free = np.zeros(n_slots, np.int64)  # first frame at which each slot is free again
    for a, z, k in sorted(pieces):
        cand = np.flatnonzero(free <= a)
        if not len(cand):
            continue  # more people than slots: this piece is lost
        s = int(rng.choice(cand))
        out[a:z, s] = players[a:z, k]
        tm[a:z, s] = team[a:z, k]
        free[s] = z
    return out, tm


def inputs(d, seed, name=None):
    """One DFL match as the provided tracking would give it. d: players, team, ball, alive.
    Returns players (F,S,2), team (F,S), ball (F,2) with NaN where missing, missing (F,), level."""
    name = name or level()
    lv = LEVELS[name]
    players, alive = d["players"], d["alive"]
    team = np.broadcast_to(d["team"], players.shape[:2])
    rng = np.random.default_rng(seed)
    gap_ball = metrica_ball(rng, d["ball"][:, :2], alive, lv)
    missing = ~np.isfinite(gap_ball[:, 0])
    ball = (
        pd.DataFrame(gap_ball)
        .rolling(METRICA_SMOOTH, center=True, min_periods=1)
        .mean()
        .to_numpy(copy=True)
    )
    ball[missing] = np.nan
    p = players + (rng.normal(0, lv["noise_m"], players.shape) / SCALE).astype(
        players.dtype
    )
    p, t, _ = degrade(
        "carrier_swap",
        CARRIER_SWAP,
        np.random.default_rng(seed + 100),
        p,
        team,
        ball,
        alive,
    )
    if lv["drop"]:
        p, t, _ = degrade(
            "player_drop",
            lv["drop"],
            np.random.default_rng(seed + 200),
            p,
            t,
            ball,
            alive,
        )
    if lv["track_s"]:
        p, t = reslot(np.random.default_rng(seed + 300), p, t, lv["track_s"])
    return dict(players=p, team=t, ball=ball, missing=missing, level=name)
