"""Players and ball per frame in three versions, for comparison and for the possession rules.

  metrica   the supplied ATD as it is.
  fused     Metrica as the skeleton, our evidence on top:
              team   our boxes are paired one-to-one with Metrica points (carrier_fix.pair, 1.5 m).
                     A segment = consecutive frames (gaps <= GAP) in which a Metrica slot stays paired
                     with the same detector track, so it never spans two people. If the segment's
                     mean shirt colour says the other team with probability >= SURE, the slot's
                     team is flipped over that segment.
              extra  an unpaired box nearer the ball than every Metrica player (ball >= MISS_M from
                     them, box < MISS_M and 1 m nearer), whose detector track is confidently H or A
                     and not an official, is added as a player.
              ball   ours (ball_track.py) where present, else Metrica's; `source` says which.
  ours      our detections only: detector tracks (ByteTrack) with continuity -- gaps up to HOLD
            frames interpolated -- and team = track colour averaged over +/- VOTE frames of the same
            track (unsure if < SURE); officials (track mostly "other") dropped; our ball only.
  keepers   (keepers.py) a box flagged as keeper takes its keeper team and source "keeper" in
            "ours" and "ours_m"; Metrica never overrules it.
  ours_m    "ours", but where our colour is unsure the team comes from the Metrica players around
            that position, each weighted exp(-d^2 / 2 SIGMA_M^2) within NEAR_M; still unsure if no
            Metrica player is that close. Metrica only fills our doubts, it never overrules a sure
            colour.

All constants come from measurements on other data or from carrier_fix.py; none from frames of this
match. Import and call load(start, n).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import GAME, GAMES, require  # noqa: E402
from baseline import team_of_each_track, HOME  # noqa: E402

SCALE = np.array(GAMES[GAME]["pitch"], np.float32)
SURE, GAP, HOLD, VOTE, MISS_M = 0.8, 3, 25, 12, 2.0
BALL_GAP = 50  # interpolate our ball over gaps up to 2 s; longer ones stay empty
SIGMA_M, NEAR_M = 1.0, 3.0  # Metrica vs our feet: median 0.19 m, so 1 m is generous, 3 m is far


def _detections():
    foot, box = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "night", "tactical", "*.npz"))):
        c = np.load(path)
        keep = (c["cls"] == 0) & np.all((c["foot"] > -0.02) & (c["foot"] < 1.02), axis=1)
        foot.append(c["foot"][keep])
        box.append(c["xyxy"][keep])
    z = np.load(os.path.join(ROOT, "data", "cleanup", "carrier_codes.npz"))
    gk = os.path.join(ROOT, "data", "cleanup", "keepers.npz")
    gk = np.load(gk)["team"] if os.path.exists(gk) else np.full(len(z["det_frame"]), "")
    return (z["det_frame"], z["det_slot"], z["det_track"], z["det_p"].astype(np.float32),
            np.concatenate(foot), np.concatenate(box), gk)


def _our_ball(start, n):
    for path in glob.glob(os.path.join(ROOT, "data", "cleanup", "ball_track_*.npz")):
        t = np.load(path)
        a = int(t["start"])
        if a <= start and start + n <= a + len(t["xy"]):
            return t["xy"][start - a:start - a + n]
    return np.full((n, 2), np.nan)


def load(start, n):
    atd = np.load(os.path.join(ROOT, "data", "diagnostics", "broadcast", "atd_tactical.npz"))
    mp, mb = atd["players"][start:start + n], atd["ball"][start:start + n]
    slot_team = np.where(team_of_each_track(require(["atd_meta"])[0]) == HOME, "H", "A")
    f, slot, track, p, foot, box, gk = _detections()
    w = (f >= start) & (f < start + n)
    d = pd.DataFrame({"f": f[w] - start, "slot": slot[w], "track": track[w],
                      "pH": p[w, 0], "pA": p[w, 1], "pO": p[w, 2], "x": foot[w, 0], "y": foot[w, 1],
                      "x1": box[w, 0], "y1": box[w, 1], "x2": box[w, 2], "y2": box[w, 3],
                      "gk": np.where(gk[w] == "", None, gk[w])})
    ob = _our_ball(start, n)

    # Per detector track: whole-window mean colour (for "extra" players and officials).
    tr = d[d.track >= 0].groupby("track")[["pH", "pA", "pO"]].mean()

    # ---- fused: team flips over (slot, track) segments
    team = np.broadcast_to(slot_team, mp.shape[:2]).copy()
    s = d[(d.slot >= 0) & (d.track >= 0)].sort_values(["slot", "f"])
    brk = (s.slot.diff() != 0) | (s.track.diff() != 0) | (s.f.diff() > GAP + 1)
    s = s.assign(seg=brk.cumsum())
    g = s.groupby("seg").agg(slot=("slot", "first"), a=("f", "min"), z=("f", "max"),
                             pH=("pH", "mean"), pA=("pA", "mean"))
    flips = 0
    for r in g.itertuples():
        colour = "H" if r.pH >= SURE else "A" if r.pA >= SURE else None
        if colour and colour != slot_team[r.slot]:
            team[r.a:r.z + 1, r.slot] = colour
            flips += r.z - r.a + 1

    ball = np.where(np.isfinite(ob[:, :1]), ob, mb)
    source = np.where(np.isfinite(ob[:, 0]), "ours", np.where(np.isfinite(mb[:, 0]), "metrica", ""))

    # ---- fused: extra player at the ball
    extra = {}
    un = d[(d.slot < 0) & (d.track >= 0)]
    un = un.join(tr, on="track", rsuffix="_t")
    un = un[(un.pO_t < 0.5) & ((un.pH_t >= SURE) | (un.pA_t >= SURE))]
    if len(un):
        fi = un.f.to_numpy()
        bxy = ball[fi] * SCALE
        bd = np.linalg.norm(un[["x", "y"]].to_numpy() * SCALE - bxy, axis=1)
        dm = np.linalg.norm((mp[fi] - ball[fi][:, None]) * SCALE, axis=2)
        dm = np.where(np.isnan(dm), np.inf, dm).min(axis=1)
        ok = (bd < MISS_M) & (dm >= MISS_M) & (bd < dm - 1.0)
        for r, k in zip(un[ok].itertuples(), np.flatnonzero(ok)):
            extra.setdefault(int(r.f), []).append((np.array([r.x, r.y]), "H" if r.pH_t >= SURE else "A"))

    # ---- ours: detector tracks with continuity
    ours = {}  # frame -> list of (xy, team or None)
    for t, h in d[d.track >= 0].sort_values("f").groupby("track"):
        if tr.loc[t, "pO"] >= 0.5:
            continue  # official
        h = h.drop_duplicates("f")
        frames = np.arange(h.f.min(), h.f.max() + 1)
        full = h.set_index("f").reindex(frames)
        gap_len = full.x.isna().astype(int).groupby(full.x.notna().cumsum()).transform("sum")
        fill = full.x.isna() & (gap_len <= HOLD)
        xy = full[["x", "y", "x1", "y1", "x2", "y2"]].interpolate(limit_area="inside")
        col = full[["pH", "pA"]].rolling(2 * VOTE + 1, center=True, min_periods=1).mean()
        keep_gk = full.gk.ffill()
        for fr, x, y, b, ph, pa, seen, filled, g in zip(frames, xy.x, xy.y, xy[["x1", "y1", "x2", "y2"]].to_numpy(),
                                                        col.pH, col.pA, full.x.notna(), fill, keep_gk):
            if seen or filled:
                tm = g if isinstance(g, str) else "H" if ph >= SURE else "A" if pa >= SURE else None
                ours.setdefault(int(fr), []).append((np.array([x, y]), tm, b, bool(seen), isinstance(g, str)))

    # ---- ours_m: unsure colour -> Metrica players around, weighted by distance
    ours_m, borrowed = {}, 0
    for fr, items in ours.items():
        on = np.isfinite(mp[fr, :, 0])
        pts, tms = mp[fr, on] * SCALE, np.broadcast_to(slot_team, mp.shape[:2])[fr, on]
        out = []
        for q, tm, b, seen, is_gk in items:
            src = "keeper" if is_gk else "colour" if tm else None
            if tm is None and len(pts):
                dist = np.linalg.norm(pts - q * SCALE, axis=1)
                wgt = np.where(dist <= NEAR_M, np.exp(-dist ** 2 / (2 * SIGMA_M ** 2)), 0.0)
                if wgt.sum() > 0:
                    wh = wgt[tms == "H"].sum()
                    tm, src = ("H" if wh >= wgt.sum() - wh else "A"), "metrica"
                    borrowed += 1
            out.append((q, tm, b, seen, src))
        ours_m[fr] = out

    # ---- ball variants for "ours_m": gaps interpolated (<= BALL_GAP), or filled with Metrica
    bdf = pd.DataFrame(ob, columns=["x", "y"])
    run = bdf.x.isna().astype(int).groupby(bdf.x.notna().cumsum()).transform("sum")
    interp = bdf.interpolate(limit_area="inside").to_numpy(copy=True)
    interp[(bdf.x.isna() & (run > BALL_GAP)).to_numpy()] = np.nan
    mfill = np.where(np.isfinite(ob[:, :1]), ob, mb)

    return {"metrica": (mp, np.broadcast_to(slot_team, mp.shape[:2]), mb),
            "ball_interp": interp, "ball_mfill": mfill,
            "fused": (mp, team, ball, source, extra),
            "ours": (ours, ob), "ours_m": (ours_m, ob), "flipped_slot_frames": flips,
            "borrowed": borrowed, "n_ours": sum(len(v) for v in ours.values()),
            "unsure_m": sum(t is None for v in ours_m.values() for _, t, *_ in v)}
