"""B2: possession rule "touch + confirmation of control", scored on the 7 DFL matches.

Why. The challenge changes possession only when the other team gains CONTROL of the ball. A touch
without control, a deflection that does not intercept, an aerial duel won without control: none of
them changes possession, and while the ball is loose it stays with the last team. Method 4
(dfl_method4.py) hands the ball over at EVERY touch, so each deflection splits a true possession in
three. Here a touch by the other team only changes possession if it is confirmed.

Rule (run_touch_control), inside every alive stretch (dead ball -> D):
  candidates   touches from dfl_control.touches (sudden change of ball velocity, player within
               TOUCH_M), team = team of the player nearest the ball at the touch.
  confirmation a touch at frame i by team X != current team is confirmed if, in the window
               (i, i + T1 s], cut short at the first touch by the other team and at the end of the
               alive stretch, at least one of:
                 (a) team X touches the ball again (>= MIN_SEP frames later): second touch, or a
                     pass to a team-mate;
                 (b) a player of X stays within R m of the ball, and nearer than every opponent, for
                     N consecutive frames (conduction);
                 (c) the ball slows below V_SLOW m/s within TOUCH_M of a player of X, nearer than
                     every opponent, for C_SLOW consecutive frames (reception);
                 (d) the toucher's team already had the ball within R m, nearer than every
                     opponent, for N frames up to the touch (a dribble or a set piece whose first
                     contact was not seen): then the change is dated at the start of that run.
               If confirmed, the change is dated at the touch i, NOT at the confirmation (a late
               boundary costs spellF1: ~0.35 per second of delay). Unconfirmed touches leave the
               possession where it was.
  fallback     (optional, fallback_n) where no touch is seen (ball interpolated or Metrica's), a
               player of X within R m and nearer than every opponent for fallback_n frames hands
               the ball to X, dated at the start of that run.
  restart      after a dead stretch the team is unknown: the first confirmed event decides it and
               is written back to the first alive frame (the restart always has a clear owner). No
               confirmed event at all: first touch's team, else the majority of the nearest team.

p_home (soft output for the A1 decoder): 0.5 at dead frames; at alive frames 0.5 +/- c / 2, with c
the confidence of the last event of the current team (confirmed change, own touch), decaying with
time (TAU_S) and faster while the ball is not ours (interpolated / Metrica, TAU_GAP_S), reduced by
unconfirmed touches of the other team. Its side always agrees with `labels`.

Evaluation (evaluate_all / main): the realistic conditions of dfl_method4.py, with the corrected
chain of CONTESTO.md section 6 (no Metrica gaps during alive play): dead = Metrica-like ball
missing, our ball = truth + 0.03 m with gaps drawn from our detector's run lengths, gaps <= 2 s
interpolated and Metrica elsewhere, players + 0.2 m, 1% wrong team at the ball, fixed seeds.
Compared: baseline +/-50, method 4 (touch only), touch+control. Parameters chosen leave-one-match-
out (choose on 6, measure on the 7th). Only aggregates are printed.

    python pipeline/touch_control.py [--seeds 0 1] [--quick] [--dead auto|metrica|b1] [--no-save]
Writes <match>_rule.npz (labels, p_home, variant) to dfl_paths.out_dir(), with the parameters
chosen without that match.
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import possession_metrics as pm  # noqa: E402
import dfl_paths  # noqa: E402
from baseline import smooth  # noqa: E402
from dfl_bench import SCALE, load, with_halves  # noqa: E402
from dfl_control import TOUCH_M, touches, touches_across_gaps, run_rule  # noqa: E402
from dfl_degrade import FPS, degrade, episodes, runs  # noqa: E402
from dfl_method4 import JITTER_M, METRICA_SMOOTH, PLAYER_NOISE_M, fill, runs_mask  # noqa: E402

# -- parameters ----------------------------------------------------------------------------------
# Touch threshold T is a change of ball velocity (m/s) over +/- 2 frames (0.16 s), so
# T m/s ~ T / 0.16 = 6.25 T m/s^2. Link & Hoernig 2017 use an acceleration peak >= 4 m/s^2 on clean
# TRACAB data (T ~ 0.64); with our noise the useful range is higher.
DEFAULTS = dict(T=2.0, T1=1.0, R=1.0, N=4, fallback_n=None)
GRID = dict(T=(1.0, 2.0, 3.0),          # ~6, 12.5, 19 m/s^2
            T1=(0.6, 1.0, 1.6),          # s, window to confirm
            R=(1.0, 1.5),                # m, control zone (Vidal-Codina 2022: 1 m on less precise
                                         # tracking; Borghesi 2023: 1-1.5 m on noisy tracking)
            N=(4, 8),                    # frames of conduction
            fallback_n=(None, 25))       # control without a seen touch (1 s), off / on
QUICK_GRID = dict(T=(2.0,), T1=(0.6, 1.0), R=(1.0, 1.5), N=(4,), fallback_n=(None, 25))
M4_T = (1.0, 2.0, 3.0)                   # method 4, chosen leave-one-match-out as well
MIN_SEP = 3        # frames: a touch fired again within this is the same touch
V_SLOW = 5.0       # m/s: ball "slowed" near a player (Morra/Borghesi use 8.6 m/s as controllable)
C_SLOW = 3         # frames of slow ball near the same team
GAP_FILL = 50      # frames (2 s) interpolated, Metrica beyond (CONTESTO.md section 6)
BALL_GONE_DEAD = 0.9
SEEDS = (0,)
TOL = 25           # frames, tolerance to match a true change of possession in the error analysis
TAU_S, TAU_GAP_S = 8.0, 2.0
CONF = dict(touch=0.9, control=0.8, own=0.9, restart=0.6, guess=0.3)
DOUBT = 0.6        # factor on the confidence at an unconfirmed touch of the other team
C_MIN = 0.05


# -- features ------------------------------------------------------------------------------------

def team_distances(players, team, ball):
    """Distance (m) from the ball to the nearest player overall / of H / of A, and nearest team."""
    d = np.linalg.norm((players - ball[:, None]) * SCALE, axis=2).astype(np.float32)
    d = np.where(np.isnan(d), np.inf, d)
    team = np.broadcast_to(team, d.shape)
    d_home = np.where(team == "H", d, np.inf).min(1)
    d_away = np.where(team == "A", d, np.inf).min(1)
    k = d.argmin(1)
    dmin = d[np.arange(len(k)), k]
    near = np.where(np.isfinite(dmin), team[np.arange(len(k)), k], "")
    return near, dmin, d_home, d_away


def ball_speed(ball, n=5):
    b = pd.DataFrame(ball * SCALE).rolling(n, center=True, min_periods=1).mean().to_numpy()
    return np.linalg.norm(np.gradient(b, axis=0) * FPS, axis=1)


def detect_touches(ball, dmin, src, T, gap_touches=False):
    """Touches as in dfl_method4: none searched near the Metrica ball (src == 2)."""
    tc = touches_across_gaps(ball, dmin, T, src) if gap_touches else touches(ball, dmin, T)
    near_src = pd.Series(src == 2).rolling(5, center=True, min_periods=1).max().to_numpy().astype(bool)
    return tc & ~near_src


def run_length(mask):
    """Consecutive True frames ending at each frame."""
    mask = np.asarray(mask, bool)
    c = np.cumsum(mask)
    reset = np.maximum.accumulate(np.where(~mask, c, 0))
    return c - reset


class Features:
    """Everything the rule needs per frame; built once per (match, condition)."""

    def __init__(self, near, dmin, d_home, d_away, speed, dead, src=None):
        self.near, self.dmin, self.dead = near, dmin, np.asarray(dead, bool)
        self.d = {"H": d_home, "A": d_away}
        self.speed = speed
        self.src = np.zeros(len(near), int) if src is None else src
        self._runs = {}

    @classmethod
    def from_tracking(cls, players, team, ball, dead, src=None):
        near, dmin, dh, da = team_distances(players, team, ball)
        return cls(near, dmin, dh, da, ball_speed(ball), dead, src)

    def ahead(self, X):
        Y = "A" if X == "H" else "H"
        return (self.d[X] <= self.d[Y]) & ~self.dead

    def control_run(self, X, R):
        key = ("b", X, R)
        if key not in self._runs:
            self._runs[key] = run_length((self.d[X] < R) & self.ahead(X))
        return self._runs[key]

    def slow_run(self, X):
        key = ("c", X)
        if key not in self._runs:
            self._runs[key] = run_length((self.speed < V_SLOW) & (self.d[X] < TOUCH_M) & self.ahead(X))
        return self._runs[key]


# -- the rule ------------------------------------------------------------------------------------

def _alive_stretches(dead):
    return runs(~np.asarray(dead, bool))


def _confirm(f, i, X, z, t_idx, t_team, T1, R, N):
    """(kind, frame) of the confirmation of a touch by X at frame i, or (None, None).
    z = end (exclusive) of the alive stretch."""
    end = min(i + int(round(T1 * FPS)), z - 1)
    lo = np.searchsorted(t_idx, i + MIN_SEP)
    hi = np.searchsorted(t_idx, end, side="right")
    second = None
    for j in range(lo, hi):
        if t_team[j] == X:
            second = t_idx[j]
            break
        if t_team[j]:
            end = t_idx[j] - 1  # the other team touched first: the window closes there
            break
    best = (None, None)
    run = f.control_run(X, R)
    back = int(max(run[i], run[i - 1] if i > 0 else 0))
    if back >= N:  # (d) X already had the ball close up to this touch (dribble, set piece)
        return "d", i - back + (1 if run[i] >= back else 0)
    if second is not None:
        best = ("a", int(second))
    if end > i:
        w = np.arange(i + 1, end + 1)
        since = w - i + 1
        ok = np.flatnonzero(np.minimum(f.control_run(X, R)[w], since) >= N)
        if len(ok) and (best[1] is None or w[ok[0]] < best[1]):
            best = ("b", int(w[ok[0]]))
        ok = np.flatnonzero(np.minimum(f.slow_run(X)[w], since) >= C_SLOW)
        if len(ok) and (best[1] is None or w[ok[0]] < best[1]):
            best = ("c", int(w[ok[0]]))
    return best


def _fallback_events(f, R, n):
    out = []
    for X in ("H", "A"):
        for a, z in runs((f.d[X] < R) & f.ahead(X)):
            if z - a >= n:
                out.append((int(a), X, "control"))
    return out


def run_touch_control(f, touch, T1=DEFAULTS["T1"], R=DEFAULTS["R"], N=DEFAULTS["N"],
                      fallback_n=DEFAULTS["fallback_n"], soft=True):
    """Touch + confirmation of control.

    f      Features (nearest team, distances per team, ball speed, dead, ball source)
    touch  (F,) bool, detected touches (detect_touches / dfl_control.touches*)
    Returns labels (F,) '<U1' in {H, A, D}, p_home (F,) float32 (None if soft=False) and info:
    dict of lists 'changes' [(frame, team, kind, confirm_frame)], 'unconfirmed' [(frame, team)].
    """
    n = len(f.near)
    dead = f.dead
    touch = np.asarray(touch, bool) & ~dead
    t_idx = np.flatnonzero(touch)
    t_team = f.near[t_idx]
    cand = [(int(i), str(X), "touch") for i, X in zip(t_idx, t_team) if X]
    if fallback_n:
        cand += _fallback_events(f, R, fallback_n)
    cand.sort(key=lambda c: (c[0], c[2] != "touch"))
    cand_f = np.array([c[0] for c in cand], dtype=np.int64)

    labels = np.full(n, "D", dtype="<U1")
    changes, unconfirmed, support = [], [], []   # support: (frame, team, strength)
    for a, z in _alive_stretches(dead):
        cur, first_touch = None, None
        bounds = []  # (frame, team) where the label switches inside [a, z)
        for k in range(np.searchsorted(cand_f, a), np.searchsorted(cand_f, z)):
            i, X, kind = cand[k]
            if kind == "touch":
                if X == cur:
                    support.append((i, X, CONF["own"]))
                    continue
                if first_touch is None:
                    first_touch = X
                ck, cf = _confirm(f, i, X, z, t_idx, t_team, T1, R, N)
                if ck is None:
                    unconfirmed.append((i, X))
                    continue
            else:
                if X == cur:
                    continue
                ck, cf = "fallback", i + fallback_n
            if ck == "d":  # the control started before the touch: date the change there
                i = max(cf, a, bounds[-1][0] + 1 if bounds else a)
            if cur is None:
                bounds.append((a, X))
                support.append((a, X, CONF["restart"]))
            else:
                bounds.append((i, X))
            support.append((i, X, CONF["touch" if kind == "touch" else "control"]))
            changes.append((i, X, ck, cf))
            cur = X
        if cur is None:  # nothing confirmed in this alive stretch
            if first_touch is None:
                nn = f.near[a:z]
                h, aw = np.sum(nn == "H"), np.sum(nn == "A")
                first_touch = "H" if h >= aw else "A"
            bounds.append((a, first_touch))
            support.append((a, first_touch, CONF["guess"]))
        for (s, X), nxt in zip(bounds, bounds[1:] + [(z, None)]):
            labels[s:nxt[0]] = X
    info = {"changes": changes, "unconfirmed": unconfirmed}
    p_home = _soft(f, labels, support, unconfirmed) if soft else None
    return labels, p_home, info


def _soft(f, labels, support, unconfirmed):
    """p_home per frame: decaying confidence of the last event of the labelled team."""
    n = len(labels)
    base = np.zeros(n, np.float32)
    has = np.zeros(n, bool)
    for i, X, s in support:
        if labels[i] == X:
            base[i] = max(base[i], s)
            has[i] = True
    idx = np.maximum.accumulate(np.where(has, np.arange(n), 0))
    # the label switch also restarts the clock (restart / change at that frame carry support)
    t = np.arange(n) - idx
    not_ours = np.cumsum(f.src != 0)
    gap = not_ours - not_ours[idx]
    c = base[idx] * np.exp(-t / (TAU_S * FPS)) * np.exp(-gap / (TAU_GAP_S * FPS))
    if unconfirmed:
        doubt = np.zeros(n)
        for i, X in unconfirmed:
            if labels[i] != X:
                doubt[i] += 1
        d_cum = np.cumsum(doubt)
        c = c * DOUBT ** (d_cum - d_cum[idx])
    c = np.clip(c, C_MIN, 1.0)
    p = 0.5 + 0.5 * c * np.where(labels == "H", 1.0, -1.0)
    p[labels == "D"] = 0.5
    return p.astype(np.float32)


def method4_labels(f, touch):
    """Method 4: possession to the team of every touch (dfl_control.run_rule, no control part)."""
    return run_rule(f.near, f.dmin, f.dead, None, 0, touch)


# -- realistic conditions ------------------------------------------------------------------------

def ball_run_lengths():
    """Run lengths (seen / missing) of our ball detector, aggregate only, as dfl_method4 uses.
    $DFL_BALL_RUNS, <repo>/data/cleanup/ball_runs.npz, or any ball_runs.npz under /kaggle/input
    (same search as decoder.ball_runs); else (or DFL_BALL_RUNS=synthetic) a synthetic stand-in
    (warned)."""
    env = os.environ.get("DFL_BALL_RUNS")
    paths = [] if env == "synthetic" else [env, os.path.join(ROOT, "data", "cleanup", "ball_runs.npz")]
    if env != "synthetic":
        paths += sorted(glob.glob("/kaggle/input/**/ball_runs.npz", recursive=True))
    for path in paths:
        if path and os.path.isfile(path):
            r = np.load(path)
            return r["seen"], r["gaps"], "measured"
    rng = np.random.default_rng(12345)
    print("WARNING: ball_runs.npz not found, using synthetic run lengths (seen ~50, gaps ~12 frames)."
          " Set DFL_BALL_RUNS for the measured ones.")
    return rng.geometric(1 / 50, 5000), rng.geometric(1 / 12, 5000), "synthetic"


def metrica_dead_ball(rng, ball, alive, alive_gaps=False):
    """Metrica-like ball: missing over BALL_GONE_DEAD of the stoppages longer than 1 s. Without the
    2% alive gaps of dfl_degrade by default (corrected chain, CONTESTO.md section 6)."""
    ball = ball.copy()
    for a, z in runs(alive == 0):
        if z - a > FPS and rng.random() < BALL_GONE_DEAD:
            ball[a:z] = np.nan
    if alive_gaps:
        for a, z in episodes(rng, len(ball), 0.02, 0.2, 2.0):
            ball[a:z][alive[a:z] == 1] = np.nan
    return ball


def simulate(d, seed, run_lengths, alive_gaps=False):
    """Inputs shaped like ours on one DFL match. Returns dict of arrays."""
    seen, gaps = run_lengths
    players, alive = d["players"], d["alive"]
    team = np.broadcast_to(d["team"], players.shape[:2])
    rng = np.random.default_rng(seed)
    gap_ball = metrica_dead_ball(rng, d["ball"][:, :2], alive, alive_gaps)
    dead = ~np.isfinite(gap_ball[:, 0])
    metrica = pd.DataFrame(gap_ball).rolling(METRICA_SMOOTH, center=True, min_periods=1).mean().to_numpy(copy=True)
    metrica[dead] = np.nan
    ours = d["ball"][:, :2] + rng.normal(0, JITTER_M, (len(alive), 2)) / SCALE
    ours[~runs_mask(rng, len(alive), seen, gaps)] = np.nan
    ours[dead] = np.nan
    p = players + rng.normal(0, PLAYER_NOISE_M, players.shape) / SCALE
    p, t, _ = degrade("carrier_swap", 0.01, np.random.default_rng(seed + 100), p, team, metrica, alive)
    ball, src = fill(ours, metrica, GAP_FILL)
    return dict(players=p, team=t, ball=ball, src=src, metrica=metrica, dead=dead)


def dead_from_b1(match):
    path = dfl_paths.out_path(match, "deadball")
    if os.path.isfile(path):
        return np.load(path)["p_dead"] > 0.5
    return None


# -- scoring -------------------------------------------------------------------------------------

def _team_changes(codes, rr):
    return [t for t in pm.transitions(codes, rr) if t[1] in (pm.H, pm.A) and t[2] in (pm.H, pm.A)]


def evaluate(d, gt, pred):
    """spellF1, macroF1, possessions predicted / true, errors on the changes of possession."""
    g = pm.label_codes(with_halves(d, gt))
    p = pm.label_codes(with_halves(d, pred))
    res = pm.evaluate(g, p, taus=(TOL,))
    rr = pm.valid_runs(g)
    pairs, missed, spurious = pm.match_transitions(_team_changes(g, rr), _team_changes(p, rr), TOL)
    lat = [pp[0] - gg[0] for gg, pp in pairs]
    return dict(spell_f1=res["spell_f1_mean"], macro_f1=res["macro_f1"],
                n_pred=res["n_pred_spells"], n_gt=res["n_gt_spells"],
                n_changes_gt=len(pairs) + len(missed), missed=len(missed), spurious=len(spurious),
                lat_median=float(np.median(lat)) if lat else np.nan,
                abs_lat_median=float(np.median(np.abs(lat))) if lat else np.nan)


def grid_configs(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def cfg_name(c):
    return "T{T:g} T1={T1:g}s R{R:g} N{N} fb{fb}".format(fb=c["fallback_n"] or 0, **c)


def evaluate_all(paths=None, seeds=SEEDS, grid=GRID, dead_source="metrica", gap_touches=False,
                 keep=None, verbose=True):
    """Long DataFrame: one row per (match, seed, method, config). keep: set of match names whose
    Features are kept in the returned dict (for saving / plotting)."""
    paths = dfl_paths.match_paths() if paths is None else paths
    rl = ball_run_lengths()
    rows, kept = [], {}
    cfgs = grid_configs(grid)
    Ts = sorted(set(grid["T"]) | set(M4_T))
    for m_i, path in enumerate(paths):
        match = dfl_paths.match_name(path)
        d, gt = load(path)
        for seed in seeds:
            s = simulate(d, seed, rl[:2])
            dead = s["dead"]
            if dead_source in ("b1", "auto"):
                b1 = dead_from_b1(match)
                if b1 is not None and len(b1) == len(dead):
                    dead = b1
                elif dead_source == "b1":
                    raise FileNotFoundError("no B1 deadball output for match {}".format(m_i + 1))
            # baseline +/-50 on the Metrica ball
            near_m, _, _, _ = team_distances(s["players"], s["team"], s["metrica"])
            codes = np.where(near_m == "H", 0, np.where(near_m == "A", 1, -1))
            w50 = smooth(np.where(dead, -1, codes), 50).copy()
            w50[dead] = "D"
            base = dict(match=match, seed=seed)
            rows.append({**base, "method": "baseline +/-50", "cfg": "", **evaluate(d, gt, w50)})
            f = Features.from_tracking(s["players"], s["team"], s["ball"], dead, s["src"])
            tcs = {T: detect_touches(s["ball"], f.dmin, s["src"], T, gap_touches) for T in Ts}
            for T in M4_T:
                rows.append({**base, "method": "metodo 4 (solo tocco)", "cfg": "T{:g}".format(T),
                             **evaluate(d, gt, method4_labels(f, tcs[T]))})
            for c in cfgs:
                lab, _, info = run_touch_control(f, tcs[c["T"]], c["T1"], c["R"], c["N"], c["fallback_n"], soft=False)
                rows.append({**base, "method": "tocco + controllo", "cfg": cfg_name(c),
                             "n_unconfirmed": len(info["unconfirmed"]), **evaluate(d, gt, lab)})
            if keep and match in keep and seed == seeds[0]:
                f._runs.clear()  # keep only per-frame vectors, not the tracking
                kept[match] = dict(gt=gt, f=f, tcs=tcs)
        if verbose:
            print("partita {}/{} fatta".format(m_i + 1, len(paths)), flush=True)
    return pd.DataFrame(rows), kept, rl[2]


def lomo(df):
    """Leave-one-match-out: for each method, choose cfg on the other matches (mean spellF1 over
    matches and seeds), report the held-out match. Returns (held-out rows, chosen cfg per match)."""
    per = df.groupby(["method", "cfg", "match"], sort=False).mean(numeric_only=True).reset_index()
    out, chosen = [], {}
    for method, g in per.groupby("method", sort=False):
        for m in g.match.unique():
            train = g[g.match != m].groupby("cfg").spell_f1.mean()
            best = train.idxmax()
            row = g[(g.match == m) & (g.cfg == best)].iloc[0].to_dict()
            out.append(row)
            chosen[(method, m)] = best
    return pd.DataFrame(out), chosen


def summary(held):
    cols = ["spell_f1", "macro_f1", "n_pred", "n_gt", "missed", "spurious", "n_changes_gt", "lat_median", "abs_lat_median"]
    g = held.groupby("method", sort=False)
    t = g[cols].mean()
    t.insert(1, "spell_min", g.spell_f1.min())
    t.insert(2, "spell_max", g.spell_f1.max())
    return t.round(3)


def top_configs(df, method="tocco + controllo", k=10):
    """In-sample ranking over all matches (indicative only: the honest number is the LOMO one)."""
    g = df[df.method == method].groupby("cfg")[["spell_f1", "macro_f1", "n_pred", "missed", "spurious"]].mean()
    return g.sort_values("spell_f1", ascending=False).head(k).round(3)


def parse_cfg(name):
    """Inverse of cfg_name."""
    import re
    m = re.fullmatch(r"T([\d.]+) T1=([\d.]+)s R([\d.]+) N(\d+) fb(\d+)", name)
    if not m:
        raise ValueError(name)
    T, T1, R, N, fb = m.groups()
    return dict(T=float(T), T1=float(T1), R=float(R), N=int(N), fallback_n=int(fb) or None)


def save_outputs(kept, chosen, seed=SEEDS[0], tag="realistic"):
    """<match>_rule.npz per kept match, with the cfg chosen leave-one-match-out."""
    written = []
    for match, k in kept.items():
        name = chosen[("tocco + controllo", match)]
        c = parse_cfg(name)
        lab, p_home, _ = run_touch_control(k["f"], k["tcs"][c["T"]], c["T1"], c["R"], c["N"], c["fallback_n"])
        path = dfl_paths.out_path(match, "rule")
        np.savez_compressed(path, labels=lab, p_home=p_home,
                            variant=np.array("B2 touch+control {} | {} seed{} | LOMO".format(name, tag, seed)))
        written.append(path)
    return written


# -- plot ----------------------------------------------------------------------------------------

COLORS = {"H": "#2a6fdb", "A": "#e8833a", "D": "#c8c8c8"}


def plot_segment(k, cfg=None, start_s=None, seconds=60, ax=None):
    """GT vs method 4 vs touch+control on one stretch of a kept match, touches marked
    (filled = confirmed change, hollow = unconfirmed, small = same team)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    c = dict(DEFAULTS, **(cfg or {}))
    f, tc = k["f"], k["tcs"][c["T"]]
    lab, _, info = run_touch_control(f, tc, c["T1"], c["R"], c["N"], c["fallback_n"], soft=False)
    m4 = method4_labels(f, tc)
    gt = k["gt"]
    if start_s is None:  # first minute-long stretch, after 10 minutes, with many true changes
        ch = np.flatnonzero((gt[1:] != gt[:-1]) & np.isin(gt[1:], ["H", "A"]) & np.isin(gt[:-1], ["H", "A"]))
        ch = ch[ch > 10 * 60 * FPS]
        win = seconds * FPS
        cnt = np.searchsorted(ch, ch + win) - np.arange(len(ch))
        a = int(ch[cnt.argmax()]) - 5 * FPS if len(ch) else 10 * 60 * FPS
    else:
        a = int(start_s * FPS)
    z = min(a + seconds * FPS, len(gt))
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.2))
    rows = [("verità DFL", gt), ("metodo 4", m4), ("tocco + controllo", lab)]
    t = np.arange(a, z) / FPS
    for r, (_, y) in enumerate(rows):
        seg = y[a:z]
        cuts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            ax.broken_barh([(t[s0], (s1 - s0) / FPS)], (r - 0.38, 0.76), color=COLORS[seg[s0]], lw=0)
    confirmed = {i for i, _, kind, _ in info["changes"]}
    unconf = {i for i, _ in info["unconfirmed"]}
    for i in np.flatnonzero(tc[a:z]) + a:
        X = f.near[i]
        if i in confirmed:
            ax.plot(i / FPS, 2.62, "v", ms=9, color=COLORS.get(X, "k"), mec="k", mew=0.6)
        elif i in unconf:
            ax.plot(i / FPS, 2.62, "v", ms=9, mfc="none", mec=COLORS.get(X, "k"), mew=1.5)
        else:
            ax.plot(i / FPS, 2.62, "v", ms=4, color="#555555")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, 2.9)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("tempo (s)")
    ax.set_title("{} · {}".format(cfg_name(c), "tocchi: pieno = cambio confermato, vuoto = non confermato"),
                 fontsize=10, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color=COLORS["H"], label="H"), Patch(color=COLORS["A"], label="A"),
                       Patch(color=COLORS["D"], label="D"),
                       Line2D([], [], marker="v", ls="", color="#555555", label="tocco")],
              ncol=4, loc="upper right", bbox_to_anchor=(1, -0.18), frameon=False, fontsize=9)
    return ax


# -- CLI -----------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--quick", action="store_true", help="small grid")
    ap.add_argument("--dead", default="metrica", choices=("metrica", "b1", "auto"),
                    help="dead ball: Metrica-like ball missing, or B1's <match>_deadball.npz")
    ap.add_argument("--gap-touches", action="store_true", help="touches_across_gaps instead of touches")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    paths = dfl_paths.match_paths()
    keep = None if a.no_save else {dfl_paths.match_name(p) for p in paths}
    df, kept, rl_kind = evaluate_all(paths, tuple(a.seeds), QUICK_GRID if a.quick else GRID,
                                     a.dead, a.gap_touches, keep)
    held, chosen = lomo(df)
    pd.set_option("display.width", 200)
    print("\nlunghezze dei vuoti della palla:", rl_kind, "| palla ferma:", a.dead,
          "| partite:", len(paths), "| semi:", a.seeds)
    print("\nLeave-one-match-out (scelta sulle altre partite, misura sulla tenuta fuori):")
    print(summary(held).to_string())
    print("\nConfigurazioni scelte (quante volte):")
    print(pd.Series(chosen).groupby(level=0).value_counts().to_string())
    print("\nTocco + controllo, migliori configurazioni su tutte le partite (in-sample, indicativo):")
    print(top_configs(df).to_string())
    if not a.no_save:
        written = save_outputs(kept, chosen, a.seeds[0])
        print("\nscritti {} file _rule.npz in {}".format(len(written), dfl_paths.out_dir()))
    return df, held, chosen


if __name__ == "__main__":
    main()
