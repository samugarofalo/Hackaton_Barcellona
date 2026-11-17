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
    track_s              -       120      30      mean life of a track before it moves to another
                                                  slot, s (slots are not persons, DATA.md)
"easy" is close to what the modules assumed before. $DFL_SIM picks the level (default medium).
Checked against the provided match on unlabeled counts only (12/11): a track lives 36 s median
and 107 s mean, a 10 s window holds 22 tracks (p90 23); the ball has ~113 gaps > 1 s per 90 min
(median 12 s), close to medium and hard. track_s was 30 / 10, which split tracks far more than
the provided tracking does.

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
        track_s=120.0,
    ),
    "hard": dict(
        gone_dead=0.6,
        edge_s=2.0,
        alive_gap=0.08,
        gap_s=(0.2, 10.0),
        noise_m=0.2,
        drop=0.10,
        track_s=30.0,
    ),
    # "tarato" (14/11): medium for the players; the ball made as bad as the provided match on unlabeled
    # counts (experiments/2026-11-14/sim_vs_partita/confronto.py): longer gaps in live play and the
    # ball in the air projected onto the pitch from a tactical camera (air_h m high, air_back m behind
    # the near touchline), where the true ball has a height (DFL, SkillCorner, PFF). Override any value
    # with $DFL_TARATO (JSON) while calibrating.
    "tarato": dict(
        gone_dead=0.75,
        edge_s=1.0,
        alive_gap=0.09,
        gap_s=(1.0, 15.0),
        noise_m=0.1,
        drop=0.05,
        track_s=120.0,
        air_h=31.0,
        air_back=15.0,
    ),
    # scomposizione degli errori (14/11, experiments/2026-11-14/scomposizione): medium con un solo
    # ingresso reso perfetto. Palla vera ovunque e nitida, giocatori come medium:
    "palla_perfetta": dict(
        gone_dead=0.0,
        edge_s=0.0,
        alive_gap=0.0,
        gap_s=(0.2, 2.0),
        noise_m=0.1,
        drop=0.05,
        track_s=120.0,
        smooth=1,
    ),
    # ... giocatori veri con identita' e senza errori, palla come medium:
    "giocatori_perfetti": dict(
        gone_dead=0.75,
        edge_s=1.0,
        alive_gap=0.04,
        gap_s=(0.2, 5.0),
        noise_m=0.0,
        drop=0.0,
        track_s=None,
        swap=0.0,
    ),
    # DFL as it is (experiment): true ball everywhere, sharp, true players with identity, no errors
    "perfect": dict(
        gone_dead=0.0,
        edge_s=0.0,
        alive_gap=0.0,
        gap_s=(0.2, 2.0),
        noise_m=0.0,
        drop=0.0,
        track_s=None,
        smooth=1,
        swap=0.0,
    ),
}


def level():
    name = os.environ.get("DFL_SIM") or "medium"
    if name not in LEVELS:
        raise ValueError(
            "DFL_SIM must be one of {}, not {!r}".format(sorted(LEVELS), name)
        )
    return name


def air_projection(ball, lv):
    """(F,3) normalised x, y + height in m -> (F,2): where a camera air_h m high, air_back m behind the
    near touchline on the halfway line, sees the ball on the pitch plane (higher = further away)."""
    xy = ball[:, :2] * SCALE
    z = np.nan_to_num(ball[:, 2], nan=0.0).clip(0, lv["air_h"] * 0.8) if ball.shape[1] > 2 else 0.0
    cam = np.array([SCALE[0] / 2, -lv["air_back"]], np.float32)
    k = (lv["air_h"] / (lv["air_h"] - z))[:, None] if np.ndim(z) else 1.0
    return ((cam + (xy - cam) * k) / SCALE).astype(ball.dtype)


def metrica_ball(rng, ball, alive, lv):
    """The true ball with the gaps of the provided one: most stoppages (edges late), some live play."""
    ball = air_projection(ball, lv) if lv.get("air_h") else ball.copy()
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


# -- our own tracking instead of the provided one (experiments, CONTESTO.md sections 6, 10, 11, 14) --
# $DFL_BALL: metrica (default) | ours_ideal | ours_proj | ours_proj_nofill | ours_proj_allmetrica
#   ours_*: our detector's ball, sharp (no 12-frame blur) + JITTER_M, seen/missing in runs drawn
#   from the run lengths measured on the provided match (data/cleanup/ball_runs.npz, $DFL_BALL_RUNS),
#   independent of the provided ball's gaps in live play; missing too in the stoppages where the
#   provided ball is missing. ideal = true position (no height error, the old optimistic sim);
#   proj = projected from a camera like ours (CAM_ANGLE, CAM_H): a ball in the air moves along the
#   line of sight. Gaps <= FILL_GAP frames interpolated, longer ones filled with the provided ball
#   (nofill: left missing; allmetrica: every gap filled with the provided ball).
# $DFL_PLAYERS: metrica (default) | ours  -- method 4: OURS_PLAYERS replaces the level's player errors.
JITTER_M, CAM_ANGLE, CAM_H, FILL_GAP = 0.03, 30.0, 30.0, 50
AIR_Z = 0.5  # m: niente tocchi dove l'altezza stimata supera questo valore (prova 6)
OURS_PLAYERS = dict(noise_m=0.11, drop=0.045, track_s=30.0)  # feet 0.10-0.12 m, found 95.5-96%,
                                                              # IDF1 0.36-0.38 (SoccerTrack v2)
BALLS = ("metrica", "ours_ideal", "ours_proj", "ours_proj_nofill", "ours_proj_allmetrica")


def our_ball(d, seed, metrica, missing, kind):
    """Our ball fused with the provided one as the chain would use it. Returns ball (F,2), src (F,)
    0 ours, 1 interpolated, 2 provided ball (touches are not searched near 2)."""
    from dfl_method4 import fill, runs_mask
    path = os.environ.get("DFL_BALL_RUNS") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cleanup", "ball_runs.npz")
    r = np.load(path)
    rng = np.random.default_rng(seed + 400)
    F = len(metrica)
    true = d["ball"][:, :2] * SCALE
    if kind == "ours_ideal":
        xy = true
    else:
        z = np.clip(d["ball"][:, 2], 0, None)
        cam = np.array([52.5, 34 - CAM_H / np.tan(np.radians(CAM_ANGLE)), CAM_H])
        k = cam[2] / np.maximum(cam[2] - z, 1.0)
        xy = cam[:2] + (true - cam[:2]) * k[:, None]
    ours = (xy + rng.normal(0, JITTER_M, xy.shape)) / SCALE
    ours[~runs_mask(rng, F, r["seen"], r["gaps"]) | ((d["alive"] == 0) & missing)] = np.nan
    if kind == "ours_proj_nofill":
        b, src = fill(ours, np.full_like(metrica, np.nan), FILL_GAP)
    else:
        b, src = fill(ours, metrica, 0 if kind == "ours_proj_allmetrica" else FILL_GAP)
    if os.environ.get("DFL_AIR_TOUCH") == "filter" and kind != "ours_ideal":
        # prova 6: niente tocchi dentro un volo riconosciuto dalla firma della gravita' (ball_flight.py),
        # tenuti quelli entro AIR_EDGE fotogrammi dal decollo e dall'atterraggio (calcio, ricezione,
        # colpo di testa che spezza la parabola)
        # solo dove l'altezza stimata dal tempo di volo supera AIR_Z (gli archi interi coprivano il 29%
        # del gioco vivo e toglievano il 36% dei tocchi)
        from ball_flight import correct
        _, zest, _ = correct(ours * SCALE, cam)
        src = np.where(zest > AIR_Z, np.maximum(src, 3), src)
    return b, src


PLACE_S, CARRY_NOISE_M, PLACED_NOISE_M = 3.0, 0.3, 0.02


def frozen_dead_ball(ball, alive):
    """True se la palla e' ferma (spostamento esattamente 0) in oltre meta' dei fotogrammi fermi:
    l'artefatto DFL. Gli altri dataset hanno la palla vera a gioco fermo e non vanno toccati."""
    dead = np.flatnonzero(alive[1:] == 0) + 1
    if not len(dead):
        return False
    step = np.abs(ball[dead, :2] - ball[dead - 1, :2]).sum(1)
    return np.mean(step[np.isfinite(step)] == 0) > 0.5 if np.isfinite(step).any() else False


def moving_dead_ball(ball, alive, rng):
    """DFL freezes the ball while the game is stopped (speed exactly 0 on every dead frame, against
    0.4-0.9% of live frames): an artefact the provided tracking does not have, and B1 learned it.
    With $DFL_DEAD_BALL=moving, every dead stretch [a, z) is replaced by a carried ball (straight from
    where it went dead to where play restarts, + smoothed noise CARRY_NOISE_M) and, in the last
    PLACE_S seconds, a placed ball (restart point + PLACED_NOISE_M). Height 0. No label is added:
    alive and possession stay the DFL ones."""
    out = ball.copy()
    F = len(ball)
    for a, z in runs(alive == 0):
        start = ball[a, :2]
        end = ball[min(z, F - 1), :2] if z < F else start
        n = z - a
        place = min(int(PLACE_S * FPS), n // 2)
        carry = n - place
        t = (np.arange(carry) + 1) / max(carry, 1)
        noise = pd.DataFrame(rng.normal(0, 1, (carry, 2))).rolling(25, center=True, min_periods=1).mean().to_numpy(copy=True)
        noise *= CARRY_NOISE_M / max(noise.std(), 1e-9)
        out[a:a + carry, :2] = start + (end - start) * t[:, None] + noise / SCALE
        out[a + carry:z, :2] = end + rng.normal(0, PLACED_NOISE_M, (place, 2)) / SCALE
        out[a:z, 2] = 0.0
    return out


def inputs(d, seed, name=None):
    """One DFL match as the provided tracking would give it. d: players, team, ball, alive.
    Returns players (F,S,2), team (F,S), ball (F,2) with NaN where missing, missing (F,), level,
    src (F,) 2 where the ball is the provided one or missing (no touch searched there)."""
    name = name or level()
    lv = dict(LEVELS[name])
    if name == "tarato" and os.environ.get("DFL_TARATO"):
        import json
        lv.update(json.loads(os.environ["DFL_TARATO"]))
    if os.environ.get("DFL_PLAYERS", "metrica") == "ours":
        lv.update(OURS_PLAYERS)
    kind = os.environ.get("DFL_BALL", "metrica")
    if kind not in BALLS:
        raise ValueError("DFL_BALL must be one of {}, not {!r}".format(BALLS, kind))
    if os.environ.get("DFL_DEAD_BALL", "frozen") == "moving" and frozen_dead_ball(d["ball"], d["alive"]):
        d = {"players": d["players"], "team": d["team"], "alive": d["alive"],
             "ball": moving_dead_ball(d["ball"], d["alive"], np.random.default_rng(seed + 500))}
    players, alive = d["players"], d["alive"]
    team = np.broadcast_to(d["team"], players.shape[:2])
    rng = np.random.default_rng(seed)
    gap_ball = metrica_ball(rng, d["ball"] if lv.get("air_h") else d["ball"][:, :2], alive, lv)
    missing = ~np.isfinite(gap_ball[:, 0])
    ball = (
        pd.DataFrame(gap_ball)
        .rolling(lv.get("smooth", METRICA_SMOOTH), center=True, min_periods=1)
        .mean()
        .to_numpy(copy=True)
    )
    ball[missing] = np.nan
    p = players + (rng.normal(0, lv["noise_m"], players.shape) / SCALE).astype(
        players.dtype
    )
    p, t, _ = degrade(
        "carrier_swap",
        lv.get("swap", CARRIER_SWAP),
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
    src = np.where(missing, 2, 0)
    if kind != "metrica":  # after the players, so that they stay the same for every ball
        ball, src = our_ball(d, seed, ball, missing, kind)
        missing = ~np.isfinite(ball[:, 0])
        src = np.where(missing, 2, np.where(src == 3, 2, src))
    return dict(players=p, team=t, ball=ball, missing=missing, level=name, src=src)
