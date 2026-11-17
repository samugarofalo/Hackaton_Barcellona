"""B1: dead-ball detector from player positions (+ the ball when it is there), trained on DFL.

Why: our real tracking loses the ball on 41% of graded frames, almost always with the game stopped,
but not always. "Ball missing = D" (the baseline) breaks possessions on every gap in play and
misses the stoppages where the ball is still tracked. Here P(dead) per frame is learned from the
collective motion and shape of the players (Lang et al. 2022, Sci. Rep.: positions only, temporal
context of +/- 0.4 ... 80 s, ~92% per-frame accuracy) plus ball features where available.

Training data: the 7 converted DFL matches (other games, CC BY 4.0), degraded to look like the
provided tracking by dfl_sim.inputs (the simulation shared with B2, C1 and A1; level in $DFL_SIM):
the ball also missing in live play and still there in some stoppages, players with noise, drop-outs
and tracks that change slot. Fixed seeds.

Model: LightGBM binary (sklearn HistGradientBoosting if lightgbm is missing). Leave-one-match-out:
each match is predicted by a model trained on the other 6 (out-of-fold p_dead). Post-processing:
median filter on p_dead (window in frames), threshold, minimum stop length; window and threshold are
chosen by nested LOMO (for each held-out match, on the other 6 matches' out-of-fold predictions).

Outputs (dfl_paths contract):
    <out>/<match>_deadball.npz   p_dead (F,) float32 (out-of-fold, median-filtered), variant,
                                 threshold, window, min_dead
    <models>/deadball_lgbm.txt (or deadball_hgb.joblib) + deadball_meta.json: final model trained
    on all 7 matches, feature names and the chosen post-processing, to apply to another match with
    predict_p_dead().

    python pipeline/deadball.py [--stride 5] [--no-long] [--quick]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dfl_paths  # noqa: E402
from dfl_bench import SCALE, load, score  # noqa: E402
from dfl_sim import inputs, level  # noqa: E402
from baseline import smooth  # noqa: E402

FPS = 25
VMAX = 12.0          # m/s, faster than any player: a slot switch or a tracking jump
AMAX = 15.0          # m/s^2
SHIFTS_S = (0.4, 0.8, 2, 4, 20)
LONG_SHIFTS_S = (40, 80)
WINDOWS = (1, 51, 91, 151)            # median filter, frames (91 = 3.64 s, Lang et al.)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
MIN_DEAD = 25                          # frames: DFL stops <= 1 s are almost absent (2 of 711)
STOP_MIN = 2 * FPS                     # segment metric: stops >= 2 s
STOP_IOU = 0.5
NUM_THREADS = 4
SEED = 0

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  deterministic=True, force_col_wise=True, num_threads=NUM_THREADS, seed=SEED,
                  verbose=-1)
N_ROUNDS = 400
EDGE_S = 3.0   # s: bordi delle soste pesati con $DFL_B1_PESI (14/11)

# Savitzky-Golay, polynomial 2, window 7: smoothing and first-derivative coefficients.
SG_SMOOTH = np.array([-2, 3, 6, 7, 6, 3, -2], np.float64) / 21.0
SG_DERIV = np.array([-3, -2, -1, 0, 1, 2, 3], np.float64) / 28.0  # per frame


def models_dir():
    path = os.environ.get("DEADBALL_MODELS") or (
        "/kaggle/working/models" if dfl_paths.on_kaggle() else os.path.join(ROOT, "models"))
    os.makedirs(path, exist_ok=True)
    return path


# -- small numpy helpers -----------------------------------------------------------------------

def _conv_time(x, kernel):
    """Convolve along axis 0 (time); NaN wherever any frame of the window is NaN or out of range."""
    k = len(kernel)
    h = k // 2
    F = x.shape[0]
    out = np.zeros(x.shape, np.float64)
    for j, c in enumerate(kernel):
        s = j - h  # out[t] += c * x[t + s]
        lo, hi = max(0, -s), min(F, F - s)
        out[lo:hi] += c * x[lo + s:hi + s]
    out[:h] = np.nan
    out[F - h:] = np.nan
    return out


def _sg_smooth(x):
    """SG smoothing that keeps the raw value where the 7-frame window is incomplete."""
    s = _conv_time(x, SG_SMOOTH)
    return np.where(np.isfinite(s), s, x)


def _window_mean(x, w, side):
    """NaN-safe mean of x over the past w frames [t-w, t) or the future (t, t+w]. x: (F,)."""
    F = len(x)
    ok = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    t = np.arange(F)
    if side == "past":
        lo, hi = np.maximum(t - w, 0), t
    else:
        lo, hi = t + 1, np.minimum(t + 1 + w, F)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _row_quantiles(v, qs):
    """Per-row quantiles of v (F, P) ignoring NaN (sort puts NaN last)."""
    s = np.sort(v, axis=1)
    n = np.isfinite(v).sum(1)
    out = []
    for q in qs:
        idx = np.clip(np.floor(q * (n - 1)).astype(int), 0, v.shape[1] - 1)
        val = s[np.arange(len(v)), idx]
        out.append(np.where(n > 0, val, np.nan))
    return out


def _nanmean(a, axis):
    ok = np.isfinite(a)
    n = ok.sum(axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(ok, a, 0).sum(axis) / np.maximum(n, 1), np.nan)


def _runs(mask):
    edge = np.diff(np.r_[0, np.asarray(mask).astype(np.int8), 0])
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def _half_bounds(half):
    cut = np.flatnonzero(np.diff(half) != 0) + 1
    b = np.r_[0, cut, len(half)]
    return list(zip(b[:-1], b[1:]))


def _dist_to_rect(x, y, x0, x1, y0, y1):
    dx = np.maximum(np.maximum(x0 - x, x - x1), 0)
    dy = np.maximum(np.maximum(y0 - y, y - y1), 0)
    return np.hypot(dx, dy)


def _run_info(mask):
    """For each frame inside a True run: run length, frames since start, frames to end (0 outside)."""
    F = len(mask)
    length, since, until = np.zeros(F), np.zeros(F), np.zeros(F)
    for a, z in _runs(mask):
        length[a:z] = z - a
        since[a:z] = np.arange(z - a)
        until[a:z] = np.arange(z - a)[::-1]
    return length, since, until


# -- features ----------------------------------------------------------------------------------

BASE_PLAYER = ["n_on", "spd_mean", "spd_med", "spd_p10", "spd_p90", "spd_max", "frac_walk",
               "frac_run", "frac_sprint", "acc_mean", "coherence", "spd_H", "spd_A", "cent_dist",
               "cxabs_H", "cxabs_A", "spread_H", "spread_A", "spread_all", "x_extent", "y_extent",
               "n_out", "any_out", "max_out", "min_d_corner", "min_d_center", "n_center_circle",
               "n_in_box_max", "own_half_frac", "min_d_penspot"]
BASE_BALL = ["ball_present", "ball_speed", "ball_still", "ball_out", "ball_out_flag",
             "ball_d_corner", "ball_d_center", "ball_d_penspot", "ball_d_goalarea",
             "ball_d_player", "miss_len_s", "miss_since_s", "miss_until_s", "seen_len_s"]
SHIFTED = ["spd_mean", "spd_med", "spd_p90", "frac_walk", "frac_run", "acc_mean", "coherence",
           "cent_dist", "spread_all", "any_out", "n_in_box_max", "min_d_corner", "ball_present",
           "ball_speed"]


# 14/11: cosa succede attorno e dentro ogni buco della palla (NaN dove la palla si vede). Accese con
# $DFL_B1_BUCHI=1 (salvato nei metadati come gap_features); spente = feature di prima.
GAP = ["gap_spd", "gap_spd_prima", "gap_spd_dopo", "gap_run", "gap_coer", "gap_salto_m", "gap_vel_impl",
       "gap_bordo_inizio", "gap_bordo_fine", "gap_vpalla_prima"]
GAP_CTX_S = 2.0


def gap_on():
    return os.environ.get("DFL_B1_BUCHI") == "1"


def feature_names(long_shifts=True, gap=False):
    shifts = SHIFTS_S + (LONG_SHIFTS_S if long_shifts else ())
    names = BASE_PLAYER + BASE_BALL
    for s in shifts:
        for side in ("past", "fut"):
            names += ["{}_{}{:g}s".format(f, side, s) for f in SHIFTED]
    return names + (GAP if gap else [])


def _gap_features(f, b, present, L, W):
    """Per ogni buco della palla: velocita' dei giocatori dentro, nei GAP_CTX_S prima e dopo, corsa e
    coerenza dentro; salto tra l'ultima palla vista prima e la prima dopo (e velocita' implicita);
    distanza dalla linea di fondo/laterale piu' vicina all'inizio e alla fine (negativa fuori); velocita'
    della palla all'ultimo fotogramma visto."""
    F = len(present)
    g = {n: np.full(F, np.nan) for n in GAP}
    n = int(GAP_CTX_S * FPS)

    def bordo(i):
        x, y = b[i]
        return min(x, L - x, y, W - y)

    for a, z in _runs(~present):
        g["gap_spd"][a:z] = np.nanmean(f["spd_mean"][a:z]) if np.isfinite(f["spd_mean"][a:z]).any() else np.nan
        g["gap_run"][a:z] = np.nanmean(f["frac_run"][a:z]) if np.isfinite(f["frac_run"][a:z]).any() else np.nan
        g["gap_coer"][a:z] = np.nanmean(f["coherence"][a:z]) if np.isfinite(f["coherence"][a:z]).any() else np.nan
        pre, post = f["spd_mean"][max(0, a - n):a], f["spd_mean"][z:z + n]
        if np.isfinite(pre).any():
            g["gap_spd_prima"][a:z] = np.nanmean(pre)
        if np.isfinite(post).any():
            g["gap_spd_dopo"][a:z] = np.nanmean(post)
        if a > 0:
            g["gap_bordo_inizio"][a:z] = bordo(a - 1)
            vp = f["ball_speed"][max(0, a - n // 4):a]   # ultimo mezzo secondo visto (bordi del filtro NaN)
            g["gap_vpalla_prima"][a:z] = np.nanmean(vp) if np.isfinite(vp).any() else np.nan
        if z < F:
            g["gap_bordo_fine"][a:z] = bordo(z)
        if a > 0 and z < F:
            g["gap_salto_m"][a:z] = np.hypot(*(b[z] - b[a - 1]))
            g["gap_vel_impl"][a:z] = g["gap_salto_m"][a] / ((z - a + 1) / FPS)
    return g


def _features_half(ball, players, team, pitch, gap=False):
    L, W = float(pitch[0]), float(pitch[1])
    F = len(players)
    pos = players.astype(np.float64) * np.array([L, W])                 # (F, P, 2) metres
    on = np.isfinite(pos[..., 0])
    sm = _sg_smooth(pos)
    vel = _conv_time(sm, SG_DERIV) * FPS                                 # m/s
    spd = np.linalg.norm(vel, axis=2)
    bad = ~np.isfinite(spd) | (spd > VMAX)
    spd[bad] = np.nan
    vel[bad] = np.nan
    acc = np.abs(_conv_time(spd, SG_DERIV * 1.0) * FPS)
    acc[acc > AMAX] = np.nan
    x, y = sm[..., 0], sm[..., 1]

    f = {}
    f["n_on"] = on.sum(1).astype(np.float64)
    f["spd_mean"] = _nanmean(spd, 1)
    f["spd_p10"], f["spd_med"], f["spd_p90"], f["spd_max"] = _row_quantiles(spd, (0.1, 0.5, 0.9, 1.0))
    nv = np.isfinite(spd).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, cond in (("frac_walk", spd < 1.0), ("frac_run", spd > 4.0), ("frac_sprint", spd > 7.0)):
            f[name] = np.where(nv > 0, cond.sum(1) / np.maximum(nv, 1), np.nan)
        f["acc_mean"] = _nanmean(acc, 1)
        mv = np.stack([_nanmean(vel[..., 0], 1), _nanmean(vel[..., 1], 1)], 1)
        f["coherence"] = np.linalg.norm(mv, axis=1) / np.maximum(f["spd_mean"], 0.1)

    cents = {}
    for k in ("H", "A"):
        m = (team == k) & on
        xs, ys = np.where(m, x, np.nan), np.where(m, y, np.nan)
        cx, cy = _nanmean(xs, 1), _nanmean(ys, 1)
        cents[k] = (cx, cy)
        f["spd_" + k] = _nanmean(np.where(m, spd, np.nan), 1)
        f["cxabs_" + k] = np.abs(cx - L / 2)
        f["spread_" + k] = _nanmean(np.hypot(xs - cx[:, None], ys - cy[:, None]), 1)
    f["cent_dist"] = np.hypot(cents["H"][0] - cents["A"][0], cents["H"][1] - cents["A"][1])
    cx, cy = _nanmean(x, 1), _nanmean(y, 1)
    f["spread_all"] = _nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), 1)
    with np.errstate(invalid="ignore"):
        f["x_extent"] = np.nanmax(np.where(on, x, -np.inf), 1) - np.nanmin(np.where(on, x, np.inf), 1)
        f["y_extent"] = np.nanmax(np.where(on, y, -np.inf), 1) - np.nanmin(np.where(on, y, np.inf), 1)
    for k in ("x_extent", "y_extent"):
        f[k] = np.where(np.isfinite(f[k]), f[k], np.nan)

    out_d = _dist_to_rect(x, y, 0, L, 0, W)                              # NaN where off
    f["n_out"] = np.nansum(out_d > 1.0, 1).astype(np.float64)
    f["any_out"] = (f["n_out"] > 0).astype(np.float64)
    f["max_out"] = np.nanmax(np.where(on, out_d, 0), 1)
    corners = [(0, 0), (0, W), (L, 0), (L, W)]
    dc = np.min([np.hypot(x - a, y - b) for a, b in corners], axis=0)
    f["min_d_corner"] = np.nanmin(np.where(on, dc, np.inf), 1)
    dcen = np.hypot(x - L / 2, y - W / 2)
    f["min_d_center"] = np.nanmin(np.where(on, dcen, np.inf), 1)
    f["n_center_circle"] = np.nansum(dcen < 9.15, 1).astype(np.float64)
    boxes = [(0, 16.5), (L - 16.5, L)]
    yb0, yb1 = W / 2 - 20.16, W / 2 + 20.16
    f["n_in_box_max"] = np.max([np.nansum((x >= a) & (x <= b) & (y >= yb0) & (y <= yb1), 1)
                                for a, b in boxes], axis=0).astype(np.float64)
    fr = []
    for k in ("H", "A"):
        m = (team == k) & on
        side = np.sign(cents[k][0] - L / 2)[:, None]
        same = np.sign(x - L / 2) == side
        n = m.sum(1)
        fr.append(np.where(n > 0, (same & m).sum(1) / np.maximum(n, 1), np.nan))
    f["own_half_frac"] = _nanmean(np.stack(fr, 1), 1)
    spots = [(11.0, W / 2), (L - 11.0, W / 2)]
    dp = np.min([np.hypot(x - a, y - b) for a, b in spots], axis=0)
    f["min_d_penspot"] = np.nanmin(np.where(on, dp, np.inf), 1)
    for k in ("min_d_corner", "min_d_center", "min_d_penspot", "max_out"):
        f[k] = np.where(np.isfinite(f[k]), f[k], np.nan)

    # ball
    b = ball[:, :2].astype(np.float64) * np.array([L, W])
    present = np.isfinite(b[:, 0])
    bs = _sg_smooth(b)
    bv = _conv_time(bs, SG_DERIV) * FPS
    bspd = np.linalg.norm(bv, axis=1)
    f["ball_present"] = present.astype(np.float64)
    f["ball_speed"] = np.where(bspd > 60, np.nan, bspd)
    f["ball_still"] = np.where(np.isfinite(f["ball_speed"]), (f["ball_speed"] < 0.5).astype(float), np.nan)
    bx, by = b[:, 0], b[:, 1]
    f["ball_out"] = _dist_to_rect(bx, by, 0, L, 0, W)
    f["ball_out_flag"] = np.where(present, (f["ball_out"] > 0.5).astype(float), np.nan)
    f["ball_d_corner"] = np.min([np.hypot(bx - a, by - c) for a, c in corners], axis=0)
    f["ball_d_center"] = np.hypot(bx - L / 2, by - W / 2)
    f["ball_d_penspot"] = np.min([np.hypot(bx - a, by - c) for a, c in spots], axis=0)
    ga0, ga1 = W / 2 - 9.16, W / 2 + 9.16
    f["ball_d_goalarea"] = np.minimum(_dist_to_rect(bx, by, 0, 5.5, ga0, ga1),
                                      _dist_to_rect(bx, by, L - 5.5, L, ga0, ga1))
    dpb = np.hypot(x - bx[:, None], y - by[:, None])
    f["ball_d_player"] = np.min(np.where(np.isfinite(dpb), dpb, np.inf), 1)
    f["ball_d_player"] = np.where(np.isfinite(f["ball_d_player"]), f["ball_d_player"], np.nan)
    length, since, until = _run_info(~present)
    f["miss_len_s"], f["miss_since_s"], f["miss_until_s"] = length / FPS, since / FPS, until / FPS
    f["seen_len_s"] = _run_info(present)[0] / FPS
    if gap:
        f.update(_gap_features(f, b, present, L, W))
    return f


def features(d, ball, players, team=None, pitch=SCALE, long_shifts=True, gap=None):
    """(F, n_feat) float32 matrix; columns as feature_names(long_shifts).

    d: anything with d["half"] (F,) and, if team is None, d["team"] (P,).
    ball: (F, 2|3) normalised, NaN where missing. players: (F, P, 2) normalised, NaN where missing.
    team: (P,) or (F, P) 'H'/'A' (the slot's team), default d["team"].
    Halves are processed separately so no window crosses half time.
    """
    team = d["team"] if team is None else team
    gap = gap_on() if gap is None else gap
    half = np.asarray(d["half"])
    team = np.broadcast_to(np.asarray(team), players.shape[:2])
    shifts = SHIFTS_S + (LONG_SHIFTS_S if long_shifts else ())
    names = feature_names(long_shifts, gap)
    X = np.empty((len(half), len(names)), np.float32)
    for a, z in _half_bounds(half):
        f = _features_half(ball[a:z], players[a:z], team[a:z], pitch, gap)
        cols = [f[n] for n in BASE_PLAYER + BASE_BALL]
        for s in shifts:
            w = max(1, int(round(s * FPS)))
            for side in ("past", "fut"):
                cols += [_window_mean(f[n], w, side) for n in SHIFTED]
        cols += [f[n] for n in GAP] if gap else []
        X[a:z] = np.stack(cols, 1).astype(np.float32)
    return X


# -- realistic inputs --------------------------------------------------------------------------

def degraded_inputs(d, seed=0):
    """(ball (F,2), players (F,S,2), team (F,S)) shaped like the provided tracking, fixed seeds."""
    s = inputs(d, seed)
    return s["ball"], s["players"], s["team"]


# -- model -------------------------------------------------------------------------------------

def _has_lgb():
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def fit(X, y, n_rounds=N_ROUNDS, w=None):
    if _has_lgb():
        import lightgbm as lgb
        return lgb.train(LGB_PARAMS, lgb.Dataset(X, y, weight=w, free_raw_data=True), num_boost_round=n_rounds)
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=n_rounds, learning_rate=0.05, max_leaf_nodes=63,
                                       min_samples_leaf=200, l2_regularization=1.0,
                                       early_stopping=False, random_state=SEED)
    return m.fit(X, y, sample_weight=w)


def predict(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    return model.predict(X).astype(np.float32)


def importance(model, names):
    if hasattr(model, "feature_importance"):
        g = model.feature_importance("gain").astype(float)
        return pd.Series(g / max(g.sum(), 1e-12), index=names)
    return None  # HistGradientBoosting has no built-in importance


# -- post-processing ---------------------------------------------------------------------------

def median_filter(p, window, half):
    if window <= 1:
        return p.astype(np.float32)
    out = np.empty(len(p), np.float32)
    try:
        from scipy.ndimage import median_filter as mf
        for a, z in _half_bounds(half):
            out[a:z] = mf(p[a:z], size=window, mode="nearest")
    except ImportError:
        for a, z in _half_bounds(half):
            out[a:z] = pd.Series(p[a:z]).rolling(window, center=True, min_periods=1).median().to_numpy()
    return out


def to_dead(p_filt, thr, half, min_dead=MIN_DEAD):
    """Boolean dead per frame; dead runs shorter than min_dead frames are dropped."""
    dead = p_filt > thr
    if min_dead > 1:
        for a, z in _half_bounds(half):
            for s, e in _runs(dead[a:z]):
                if e - s < min_dead:
                    dead[a + s:a + e] = False
    return dead


# -- metrics -----------------------------------------------------------------------------------

def frame_metrics(dead_pred, dead_true):
    tp = float(np.sum(dead_pred & dead_true))
    fp = float(np.sum(dead_pred & ~dead_true))
    fn = float(np.sum(~dead_pred & dead_true))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return float(np.mean(dead_pred == dead_true)), f1


def stop_f1(dead_pred, dead_true, half, min_len=STOP_MIN, iou=STOP_IOU):
    """F1 over stops >= min_len frames, matched 1-to-1 (greedy, IoU-descending) at IoU >= iou."""
    tp = n_gt = n_pr = 0
    for a, z in _half_bounds(half):
        g = [r for r in _runs(dead_true[a:z]) if r[1] - r[0] >= min_len]
        p = [r for r in _runs(dead_pred[a:z]) if r[1] - r[0] >= min_len]
        n_gt += len(g)
        n_pr += len(p)
        pairs = []
        j0 = 0
        for i, (gs, ge) in enumerate(g):
            while j0 < len(p) and p[j0][1] <= gs:
                j0 += 1
            j = j0
            while j < len(p) and p[j][0] < ge:
                ps, pe = p[j]
                inter = min(ge, pe) - max(gs, ps)
                if inter > 0:
                    pairs.append((inter / float(max(ge, pe) - min(gs, ps)), i, j))
                j += 1
        pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
        ug, up = set(), set()
        for v, i, j in pairs:
            if v < iou:
                break
            if i in ug or j in up:
                continue
            ug.add(i)
            up.add(j)
        tp += len(ug)
    return 2.0 * tp / (n_gt + n_pr) if (n_gt + n_pr) else 0.0


def _nearest_codes(players, team, ball):
    dist = np.linalg.norm((players - ball[:, None, :2]) * SCALE, axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    k = dist.argmin(1)
    t = team[np.arange(len(k)), k]
    codes = np.where(t == "H", 0, 1).astype(np.int8)
    codes[~np.isfinite(dist.min(1))] = -1
    return codes


def possession_labels(codes, dead, half):
    """Starter baseline (nearest, majority +/- 50) with D where dead; alive frames the vote leaves
    empty (long ball gaps called alive) keep the last team of the half."""
    lab = smooth(np.where(dead, -1, codes), 50).copy()
    lab[dead] = "D"
    hole = (lab == "D") & ~dead
    if hole.any():
        s = pd.Series(np.where(lab == "D", None, lab), dtype=object)
        filled = np.empty(len(lab), object)
        for a, z in _half_bounds(half):
            filled[a:z] = s.iloc[a:z].ffill().bfill().fillna("H").to_numpy()
        lab[hole] = filled[hole]
    return lab


# -- the experiment ----------------------------------------------------------------------------

def _weights(y, missing, half):
    """Pesi di addestramento da $DFL_B1_PESI="<vivo con palla mancante>,<bordi>" (14/11): peso piu' alto
    dove B1 sbaglia, cioe' gioco vivo con la palla mancante ed entro EDGE_S dai bordi delle soste (dai
    due lati); 1 altrove. Vuoto = nessun peso (come prima)."""
    spec = os.environ.get("DFL_B1_PESI")
    if not spec:
        return None
    w_miss, w_edge = (float(x) for x in spec.split(","))
    w = np.ones(len(y), np.float32)
    w[(y == 0) & missing] = w_miss
    edge = np.zeros(len(y), bool)
    n = int(EDGE_S * FPS)
    for i in np.flatnonzero((y[1:] != y[:-1]) & (np.asarray(half)[1:] == np.asarray(half)[:-1])) + 1:
        edge[max(0, i - n):i + n] = True
    w[edge] = np.maximum(w[edge], w_edge)
    return w


def _prepare(paths, seed, long_shifts, verbose):
    data = []
    for path in paths:
        t0 = time.time()
        d, gt = load(path)
        d = {k: d[k] for k in d.files}
        ball, p, t = degraded_inputs(d, seed)
        X = features(d, ball, p, t, long_shifts=long_shifts)
        data.append(dict(name=dfl_paths.match_name(path), d=d, gt=gt, X=X,
                         y=(d["alive"] == 0).astype(np.int8), ball=ball, players=p, team=t,
                         missing=~np.isfinite(ball[:, 0]), codes=_nearest_codes(p, t, ball)))
        data[-1]["w"] = _weights(data[-1]["y"], data[-1]["missing"], d["half"])
        if verbose:
            print("{}: {} frames, {} features, {:.0f} s".format(data[-1]["name"], len(X), X.shape[1],
                                                                 time.time() - t0), flush=True)
    return data


def run(stride=5, long_shifts=True, seed=SEED, windows=WINDOWS, thresholds=THRESHOLDS,
        min_dead=MIN_DEAD, n_rounds=N_ROUNDS, paths=None, save=True, verbose=True):
    """Full B1 experiment. Returns a dict of tables and per-match arrays (see the notebook)."""
    paths = paths or dfl_paths.match_paths()
    names = feature_names(long_shifts, gap_on())
    data = _prepare(paths, seed, long_shifts, verbose)

    # 1. out-of-fold raw p_dead: leave-one-match-out, or $DFL_FOLDS groups of matches (match i in
    #    group i % k: with many matches one model per match is too slow)
    imps = []
    k = int(os.environ.get("DFL_FOLDS") or 0) or len(data)
    for g in range(k):
        t0 = time.time()
        test = [m for i, m in enumerate(data) if i % k == g]
        ok = set(dfl_paths.for_training([o["name"] for o in data]))
        tr = [o for i, o in enumerate(data) if i % k != g and o["name"] in ok]
        Xtr = np.concatenate([o["X"][::stride] for o in tr])
        ytr = np.concatenate([o["y"][::stride] for o in tr])
        wtr = None if tr[0]["w"] is None else np.concatenate([o["w"][::stride] for o in tr])
        model = fit(Xtr, ytr, n_rounds, wtr)
        for m in test:
            m["p_raw"] = predict(model, m["X"])
        imp = importance(model, names)
        if imp is not None:
            imps.append(imp)
        if verbose:
            print("fold {}/{} ({} partite fuori): trained on {} frames, {:.0f} s".format(
                g + 1, k, len(test), len(ytr), time.time() - t0), flush=True)

    # 2. score every (window, threshold) on every match, for the detector-based variants
    grid = {}
    for m in data:
        half = m["d"]["half"]
        for w in windows:
            pf = median_filter(m["p_raw"], w, half)
            m.setdefault("p_filt", {})[w] = pf
            for thr in thresholds:
                dead = to_dead(pf, thr, half, min_dead)
                grid[(m["name"], "detector", w, thr)] = score(
                    m["d"], m["gt"], possession_labels(m["codes"], dead, half))[1]
                grid[(m["name"], "missing&detector", w, thr)] = score(
                    m["d"], m["gt"], possession_labels(m["codes"], dead & m["missing"], half))[1]
        if verbose:
            print("grid scored:", m["name"], flush=True)

    def choose(variant, exclude):
        best, key = -1, None
        for w in windows:
            for thr in thresholds:
                v = np.mean([grid[(o["name"], variant, w, thr)] for o in data if o["name"] != exclude])
                if v > best:
                    best, key = v, (w, thr)
        return key

    # 3. nested choice (held-out match excluded) and final table
    rows = []
    for m in data:
        d, gt, half = m["d"], m["gt"], m["d"]["half"]
        true_dead = m["y"].astype(bool)
        w, thr = choose("detector", m["name"])
        w2, thr2 = choose("missing&detector", m["name"])
        m["window"], m["threshold"] = w, thr
        m["p_dead"] = m["p_filt"][w]
        dead_det = to_dead(m["p_dead"], thr, half, min_dead)
        dead_and = to_dead(m["p_filt"][w2], thr2, half, min_dead) & m["missing"]
        m["dead_pred"] = dead_det
        variants = [("ball missing (baseline)", m["missing"]),
                    ("detector", dead_det),
                    ("missing & detector", dead_and),
                    ("oracle D (gt)", true_dead)]
        for vname, dead in variants:
            acc, f1 = frame_metrics(dead, true_dead)
            macro, spell = score(d, gt, possession_labels(m["codes"], dead, half))
            rows.append(dict(match=m["name"], variant=vname, frame_acc=acc, f1_dead=f1,
                             stop_f1=stop_f1(dead, true_dead, half), macro_f1=macro, spell_f1=spell,
                             window=w if vname == "detector" else (w2 if vname.startswith("missing &") else None),
                             threshold=thr if vname == "detector" else (thr2 if vname.startswith("missing &") else None)))
        acc_raw, f1_raw = frame_metrics(m["p_raw"] > 0.5, true_dead)
        rows.append(dict(match=m["name"], variant="detector raw (no filter, thr 0.5)", frame_acc=acc_raw,
                         f1_dead=f1_raw, stop_f1=stop_f1(m["p_raw"] > 0.5, true_dead, half)))
        if save:
            variant = "lgbm_lomo_seed{}_stride{}_med{}_thr{:g}_min{}{}".format(
                seed, stride, w, thr, min_dead, "" if long_shifts else "_short") if _has_lgb() else \
                "hgb_lomo_seed{}_stride{}_med{}_thr{:g}_min{}".format(seed, stride, w, thr, min_dead)
            variant += "_sim:" + level()
            np.savez_compressed(dfl_paths.out_path(m["name"], "deadball"),
                                p_dead=m["p_dead"].astype(np.float32), p_raw=m["p_raw"].astype(np.float32),
                                variant=np.array(variant),
                                threshold=np.float32(thr), window=np.int32(w), min_dead=np.int32(min_dead))
    table = pd.DataFrame(rows)
    summary = table.groupby("variant", sort=False)[["frame_acc", "f1_dead", "stop_f1", "macro_f1",
                                                     "spell_f1"]].mean()

    imp_table = None
    if imps:
        I = pd.concat(imps, axis=1)
        imp_table = pd.DataFrame({"gain_mean": I.mean(1), "gain_min": I.min(1), "gain_max": I.max(1)})
        imp_table = imp_table.sort_values("gain_mean", ascending=False)

    # 4. final model on all matches, post-processing chosen on all out-of-fold predictions
    final = None
    if save:
        w, thr = choose("detector", None)
        w2, thr2 = choose("missing&detector", None)
        ok = set(dfl_paths.for_training([o["name"] for o in data]))
        X = np.concatenate([o["X"][::stride] for o in data if o["name"] in ok])
        y = np.concatenate([o["y"][::stride] for o in data if o["name"] in ok])
        wts = None if data[0]["w"] is None else np.concatenate([o["w"][::stride] for o in data if o["name"] in ok])
        model = fit(X, y, n_rounds, wts)
        final = save_model(model, names, dict(window=int(w), threshold=float(thr), min_dead=int(min_dead),
                                              and_window=int(w2), and_threshold=float(thr2),
                                              long_shifts=bool(long_shifts), stride=int(stride),
                                              seed=int(seed), n_rounds=int(n_rounds), sim=level(), pesi=os.environ.get("DFL_B1_PESI") or None, gap_features=gap_on(),
                                              matches=[o["name"] for o in data]))
        if verbose:
            print("final model ->", final)

    grid_df = pd.Series(grid).rename_axis(["match", "variant", "window", "threshold"]).rename("spell_f1")
    grid_df = grid_df.reset_index().groupby(["variant", "window", "threshold"]).spell_f1.mean().unstack()
    return dict(table=table, summary=summary, importance=imp_table, grid=grid_df, final=final,
                per_match={m["name"]: dict(p_raw=m["p_raw"], p_dead=m["p_dead"], alive=m["d"]["alive"],
                                           missing=m["missing"], half=m["d"]["half"],
                                           window=m["window"], threshold=m["threshold"],
                                           dead_pred=m["dead_pred"]) for m in data})


# -- save / apply ------------------------------------------------------------------------------

def save_model(model, names, post):
    out = models_dir()
    if hasattr(model, "save_model"):
        path = os.path.join(out, "deadball_lgbm.txt")
        model.save_model(path)
        kind = "lightgbm"
    else:
        import joblib
        path = os.path.join(out, "deadball_hgb.joblib")
        joblib.dump(model, path)
        kind = "sklearn_hgb"
    meta = dict(kind=kind, model_file=os.path.basename(path), features=names, fps=FPS,
                pitch_default=[float(SCALE[0]), float(SCALE[1])], **post)
    with open(os.path.join(out, "deadball_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return path


def load_model(folder=None):
    folder = folder or models_dir()
    with open(os.path.join(folder, "deadball_meta.json")) as fh:
        meta = json.load(fh)
    path = os.path.join(folder, meta["model_file"])
    if meta["kind"] == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=path)
    else:
        import joblib
        model = joblib.load(path)
    return model, meta


def predict_p_raw(ball, players, team, half, folder=None, pitch=SCALE):
    """Unfiltered P(dead) of the saved final model on any match. Returns (p_raw (F,), meta)."""
    model, meta = load_model(folder)
    X = features({"half": half}, ball, players, team, pitch=pitch, long_shifts=meta["long_shifts"],
                 gap=meta.get("gap_features", False))
    return predict(model, X), meta


def predict_p_dead(ball, players, team, half, folder=None, pitch=SCALE):
    """Apply the saved final model to any match (e.g. our own tracking, run locally).
    ball (F,2) normalised with NaN, players (F,P,2) normalised with NaN, team (P,) or (F,P) 'H'/'A',
    half (F,). Returns (p_dead filtered (F,), dead bool (F,), meta)."""
    raw, meta = predict_p_raw(ball, players, team, half, folder, pitch)
    p = median_filter(raw, meta["window"], np.asarray(half))
    return p, to_dead(p, meta["threshold"], np.asarray(half), meta["min_dead"]), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stride", type=int, default=5, help="training subsample, frames")
    ap.add_argument("--no-long", action="store_true", help="skip the 40 s and 80 s shifts")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--quick", action="store_true", help="smaller grid (window 91, thr 0.4-0.6)")
    a = ap.parse_args()
    kw = dict(windows=(91,), thresholds=(0.4, 0.5, 0.6)) if a.quick else {}
    res = run(stride=a.stride, long_shifts=not a.no_long, seed=a.seed, n_rounds=a.rounds, **kw)
    pd.set_option("display.width", 200)
    print("\nper match:\n", res["table"].round(3).to_string())
    print("\nmean over matches:\n", res["summary"].round(3).to_string())
    print("\nspellF1 grid (mean over matches, before nested choice):\n", res["grid"].round(3).to_string())
    if res["importance"] is not None:
        print("\ntop 20 features (gain share, mean over LOMO folds):\n",
              res["importance"].head(20).round(4).to_string())


if __name__ == "__main__":
    main()
