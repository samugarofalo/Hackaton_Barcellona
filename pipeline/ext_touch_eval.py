"""Touches seen in the camera view vs touches inferred on the pitch, and possession end to end, on
SoccerTrack v2 (CC BY 4.0): first 15 minutes of the 1st half of two matches, our whole chain on the
video (ext_players_eval.py --long: persons + our ball detector, colour, keepers, method 4 with the
provider tracking as maestro, fix 2). No file of the competition match is opened.

Truth: the hand-labelled ball actions (bas/, 12 classes, frame + acting team). Possession per frame
follows CHALLENGE.md as far as the events allow: a controlled action (pass, drive, cross, shot,
header, throw-in, free kick, successful tackle) gives the ball to the actor's team; a block does not
change it; OUT / GOAL make it dead (D) until the next action restarts play. Events mark the start of
a drive, not every touch, so this is a proxy: good to compare variants, rough in absolute terms.

The dead ball comes from that truth for every variant (oracle D), so only the attribution differs:
  nearest_w50        starter baseline on our 2D ball and players
  control            2D: another team's player within 1 m for 3 frames
  touch2d_T          2D: sudden change of ball velocity (> T m/s) with a player within 2 m (+control)
  sight_T            2D, only the change across the camera's line of sight (+control)
  img_touch_T        camera view: the ball box meets a player box when its image velocity (converted
                     to m/s with the player's height, 1.8 m) changes by > T; that player's team (+control)
  img_touch_T+imgctl the same, with control also from contact in the image (3 frames)
  events             the true actions as touches: ceiling of the rule with our players' absence

    .venv/bin/python pipeline/ext_touch_eval.py
Writes data/external/soccertrack-v2/eval/touch_eval.json and prints the tables.
"""

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")]
import ext_players_eval as E  # noqa: E402
import possession_metrics as pm  # noqa: E402
from baseline import smooth  # noqa: E402
from ball_link import link, select, MIN_CONF  # noqa: E402
from dfl_bench import SCALE  # noqa: E402
from dfl_control import run_rule, touches, sight_touches  # noqa: E402

CONTROL = {"PASS", "DRIVE", "HIGH PASS", "CROSS", "SHOT", "HEADER", "THROW IN", "FREE KICK",
           "PLAYER SUCCESSFUL TACKLE"}
DEAD = {"OUT", "GOAL"}
SIDE = {"left": "H", "right": "A", "L": "H", "R": "A"}
JUMP, MIN_LEN = 4, 8          # ball_track.py
BALL_GAP = 50                 # fuse.py: interpolate shorter gaps
STATIC_PX, STATIC_FRAMES = 8, 1500  # 60 s in one 8 px cell: the real ball never rests that long
BALL_OUT_PX = 10              # ball centre at most this far outside the image outline of the pitch
X_MARGIN, Y_TOP, Y_BOTTOM = 0.3, 0.1, 0.15   # contact: ball centre in the box grown by these shares
HEIGHT_M = 1.8
TOL = 5                       # frames: a detected touch matches a true action this close
T_GRID = (3.0, 5.0, 8.0)


def truth(match, n):
    """Per-frame possession labels and the list of true touches (frame, team)."""
    ev = json.load(open(os.path.join(E.DS, "bas", match, "{}_12_class_events.json".format(match))))["actions"]
    ev = sorted(((int(e["frame"]) - 1, e["label"], SIDE.get(e.get("team"), "")) for e in ev), key=lambda e: e[0])
    ev = [e for e in ev if 0 <= e[0] < n]
    lab, state = np.full(n, "D", "<U1"), "D"
    for k, (f, label, team) in enumerate(ev):
        if label in CONTROL and team:
            state = team
        elif label in DEAD:
            state = "D"
        lab[f:ev[k + 1][0] if k + 1 < len(ev) else n] = state
    touch = [(f, team) for f, label, team in ev if label not in DEAD and team]
    return lab, touch


def ball(match, n, to_pitch, outline):
    """Our ball: detections of the three windows in panorama pixels, kept on the pitch outline,
    linked + selected (no jumps), short gaps interpolated. Returns pitch metres and pixels per frame."""
    parts = [np.load(E.ball_path(match, w)) for w in range(len(E.WIN_X0))]
    f = np.concatenate([p["frame"] for p in parts])
    c = np.concatenate([p["conf"] for p in parts]).astype(np.float32)
    bb = np.concatenate([p["xyxy"] for p in parts])
    px = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
    on = np.array([cv2.pointPolygonTest(outline, (float(x), float(y)), True) >= -BALL_OUT_PX for x, y in px])
    f, c, bb, px = f[on], c[on], bb[on], px[on]
    # static objects (cones, spare balls, ad marks) fire in the same pixels all match long; the
    # panorama camera is fixed, so a cell busy in more than STATIC_FRAMES distinct frames is not the ball
    cell = np.round(px / STATIC_PX).astype(np.int64)
    key = (cell[:, 0] << 20) + cell[:, 1]
    _, first = np.unique(np.c_[key, f], axis=0, return_index=True)  # windows overlap: count frames once
    busy = pd.Series(key[first]).value_counts()
    busy = set(busy.index[busy > STATIC_FRAMES])
    near_busy = np.zeros(len(f), bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            near_busy |= np.isin(((cell[:, 0] + dx) << 20) + cell[:, 1] + dy, list(busy))
    f, c, bb, px = f[~near_busy], c[~near_busy], bb[~near_busy], px[~near_busy]
    size = np.median((bb[:, 2] - bb[:, 0] + bb[:, 3] - bb[:, 1]) / 2)
    keep, tid = link(f, px, c, size, JUMP, MIN_LEN)
    keep &= c >= MIN_CONF
    pick = select(f, px, c, tid, keep, size, JUMP)
    pix = np.full((n, 2), np.nan)
    pix[f[pick]] = px[pick]
    seen = np.isfinite(pix[:, 0])
    pix = pd.DataFrame(pix).interpolate(limit_area="inside").to_numpy(copy=True)
    gap = pd.Series(~seen).groupby(seen.cumsum()).transform("sum").to_numpy()
    pix[~seen & (gap > BALL_GAP)] = np.nan
    xy = np.full((n, 2), np.nan)
    ok = np.isfinite(pix[:, 0])
    xy[ok] = to_pitch(pix[ok, 0], pix[ok, 1])
    return xy, pix, seen, size


def players(ours, n):
    """Method-4 players per frame: pitch metres, team H/A/'' and panorama box."""
    o = ours[ours.frame < n].copy()
    o["k"] = o.groupby("frame").cumcount()
    K = int(o.k.max()) + 1
    p = np.full((n, K, 2), np.nan)
    t = np.full((n, K), "", "<U1")
    b = np.full((n, K, 4), np.nan)
    p[o.frame, o.k] = o[["px", "py"]].to_numpy()
    t[o.frame, o.k] = o.team_m.map(SIDE).fillna("").to_numpy()
    b[o.frame, o.k] = o[["x1", "y1", "x2", "y2"]].to_numpy()
    return p, t, b


def nearest(p, t, xy):
    d = np.linalg.norm(p - xy[:, None], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    k = d.argmin(1)
    dmin = d.min(1)
    return np.where(np.isfinite(dmin), t[np.arange(len(k)), k], ""), dmin


def image_contact(pix, boxes, team):
    """Per frame: team of the player whose box the ball meets ('' if none), and the image velocity
    change in m/s (pixels scaled by that player's height = 1.8 m; nearest box when no contact)."""
    x1, y1, x2, y2 = (boxes[..., i] for i in range(4))
    w, h = x2 - x1, y2 - y1
    bx, by = pix[:, 0:1], pix[:, 1:2]
    inside = ((bx >= x1 - X_MARGIN * w) & (bx <= x2 + X_MARGIN * w)
              & (by >= y1 - Y_TOP * h) & (by <= y2 + Y_BOTTOM * h))
    # among contacts, the box whose centre line and feet are nearest the ball (in box heights)
    score = np.hypot((bx - (x1 + x2) / 2) / np.where(h > 0, h, np.nan), (by - y2) / np.where(h > 0, h, np.nan))
    score = np.where(np.isnan(score), np.inf, score)
    k = np.where(inside, score, np.inf).argmin(1)
    has = inside.any(1)
    kk = np.where(has, k, score.argmin(1))
    hh = h[np.arange(len(kk)), kk]
    s = pd.DataFrame(pix).rolling(5, center=True, min_periods=1).mean().to_numpy()
    v = np.gradient(s, axis=0) * 25
    dv = np.full(len(s), 0.0)
    dv[2:-2] = np.linalg.norm(v[4:] - v[:-4], axis=1)
    dv = np.nan_to_num(dv * HEIGHT_M / np.where(np.isfinite(hh) & (hh > 0), hh, np.inf))
    tm = np.where(has, team[np.arange(len(k)), k], "")
    return tm, dv


def touch_scores(flags, who, true_touch):
    """Detected touches (bool per frame + team) against the true actions: recall, team right among
    the matched, precision (a lower bound: dribble touches are not labelled)."""
    det = np.flatnonzero(flags & (who != ""))
    used = np.zeros(len(det), bool)
    hit = right = 0
    for f, team in true_touch:
        j = np.flatnonzero(~used & (np.abs(det - f) <= TOL))
        if len(j):
            j = j[np.argmin(np.abs(det[j] - f))]
            used[j] = True
            hit += 1
            right += who[det[j]] == team
    return {"true_actions": len(true_touch), "detected": int(len(det)),
            "recall": round(hit / max(len(true_touch), 1), 3), "team_right_of_matched": round(right / max(hit, 1), 3),
            "precision_lower_bound": round(hit / max(len(det), 1), 3)}


def camera_ground(det, to_pitch):
    """Ground point under the camera: least-squares meeting of head lines (triangulate.camera_centre,
    with the pixel -> pitch polynomial instead of a homography). The panorama is stitched, so this
    is an approximation; only the horizontal position is used."""
    d = det[det.conf > 0.6].iloc[::25]
    x = ((d.x1 + d.x2) / 2).to_numpy()
    F = np.c_[to_pitch(x, d.y2.to_numpy()), np.full(len(d), HEIGHT_M)]
    G = np.c_[to_pitch(x, d.y1.to_numpy()), np.zeros(len(d))]
    ok = np.all(np.isfinite(F), 1) & np.all(np.isfinite(G), 1) & (np.linalg.norm(F[:, :2] - G[:, :2], axis=1) > 0.5)
    u = F[ok] - G[ok]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    P = np.eye(3)[None] - u[:, :, None] * u[:, None, :]
    return np.linalg.solve(P.sum(0), np.einsum("nij,nj->i", P, G[ok]))


def score(gt, pred):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    return round(float(r["spell_f1_mean"]), 4), round(float(r["macro_f1"]), 4)


def run_match(match, rng):
    for k in E.VARIANTS["fix2"]:
        E.FIX[k] = True
    R, det = E.evaluate_match(match, False, rng, long=True)
    ch = R.pop("_chain")
    n, to_pitch = E.LONG_FRAMES, ch["to_pitch"]
    gt, true_touch = truth(match, n)
    dead = gt == "D"
    xy, pix, seen, size = ball(match, n, to_pitch, E.pitch_outline(match))
    p, t, boxes = players(ch["ours"], n)
    near, dmin = nearest(p, t, xy)
    cam = camera_ground(det, to_pitch)
    img_team, img_dv = image_contact(pix, boxes, t)
    live = ~dead
    out = {"frames": n, "live_share": round(float(live.mean()), 3), "camera_m": np.round(cam, 1).tolist(),
           "ball": {"seen_live": round(float(seen[live].mean()), 3),
                    "after_interpolation_live": round(float(np.isfinite(xy[live, 0]).mean()), 3),
                    "size_px": round(float(size), 1)},
           "players_team_known_share": round(float((t != "").sum() / max(np.isfinite(p[..., 0]).sum(), 1)), 3),
           "image_contact_live": round(float((img_team[live] != "").mean()), 3),
           "possession": {}, "touches": {}}

    codes = np.where(near == "H", 0, np.where(near == "A", 1, -1)).astype(np.int8)
    preds = {}
    w50 = smooth(np.where(dead, -1, codes), 50)
    w50[dead] = "D"
    preds["nearest_w50"] = w50
    preds["control"] = run_rule(near, dmin, dead, 1.0, 3)
    xn = xy / SCALE  # dfl_control works in pitch-normalised units (times SCALE inside)
    for T in T_GRID:
        tc = touches(xn, dmin, T)
        preds["touch2d_T{:g}".format(T)] = run_rule(near, dmin, dead, 1.0, 3, tc)
        out["touches"]["touch2d_T{:g}".format(T)] = touch_scores(tc, near, true_touch)
        sc, sd, sk = sight_touches(xn, p / SCALE, T, cam[:2], 1.5)
        steam = np.where(np.isfinite(sd), t[np.arange(n), sk], "")
        preds["sight_T{:g}".format(T)] = run_rule(steam, sd, dead, 1.0, 3, sc)
        out["touches"]["sight_T{:g}".format(T)] = touch_scores(sc, steam, true_touch)
        peak = (img_dv >= np.roll(img_dv, 1)) & (img_dv >= np.roll(img_dv, -1))
        ic = (img_dv > T) & peak & (img_team != "")
        preds["img_touch_T{:g}".format(T)] = run_rule(np.where(ic, img_team, near), dmin, dead, 1.0, 3, ic)
        out["touches"]["img_touch_T{:g}".format(T)] = touch_scores(ic, img_team, true_touch)
        img_d = np.where(img_team != "", 0.0, np.inf)
        preds["img_touch_T{:g}+imgctl".format(T)] = run_rule(img_team, img_d, dead, 0.5, 3, ic)
    ev = np.zeros(n, bool)
    ev_team = np.full(n, "", "<U1")
    for f, team in true_touch:
        ev[f], ev_team[f] = True, team
    preds["events (ceiling)"] = run_rule(np.where(ev, ev_team, ""), np.full(n, np.inf), dead, None, 0, ev)
    for k, v in preds.items():
        out["possession"][k] = score(gt, v)
    return out


def main():
    rng = np.random.default_rng(0)
    res = {m: run_match(m, rng) for m in E.MATCHES}
    with open(os.path.join(E.OUT, "touch_eval.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for m, r in res.items():
        print("\n==", m, {k: v for k, v in r.items() if k not in ("possession", "touches")})
    print("\nspellF1 / macroF1 (dead ball from the truth for all)")
    rows = {k: [res[m]["possession"][k] for m in E.MATCHES] for k in res[E.MATCHES[0]]["possession"]}
    for k, v in rows.items():
        print("{:24} {}   mean spellF1 {:.3f}".format(k, "  ".join("{:.3f}/{:.3f}".format(*x) for x in v),
                                                    np.mean([x[0] for x in v])))
    print("\ntouches vs true actions (+-{} frames)".format(TOL))
    for k in res[E.MATCHES[0]]["touches"]:
        print("{:24} {}".format(k, " | ".join(str(res[m]["touches"][k]) for m in E.MATCHES)))


if __name__ == "__main__":
    main()
