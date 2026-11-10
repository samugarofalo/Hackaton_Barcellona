"""Clean the Metrica tracking only where possession is decided: the player at the ball.

DFL degradation (dfl_degrade.py) showed two errors there cost spellF1 beyond the 0.04 noise band:
the player at the ball carrying the other team's slot ("carrier swap", a duel mis-tracked) and the
player at the ball missing. Both are checked against the overnight detections, frame by frame,
with no label and no constant tied to a frame of this match:

  team      a shirt-colour classifier (H / A / other) is learned from detections paired one-to-one
            with a Metrica player (the supplied tracking is the teacher; its few % of errors are
            outvoted). "other" = detector tracks that Metrica almost never pairs (officials).
            Where the slot nearest the ball is paired with boxes whose colour says the other
            team, confidently and over +/- VOTE frames, that frame's team is flipped.
  missing   where no Metrica player is within MISS_M of the ball but an unpaired detection (not
            "other") is, and closer, that detection's colour team takes the frame.

    .venv/bin/python pipeline/carrier_fix.py
Writes data/cleanup/carrier_codes.npz (original and corrected per-frame codes) and
data/cleanup/carrier_episodes.csv, and prints how much changes.
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
from baseline import team_of_each_track, smooth, HOME, AWAY, NOBODY  # noqa: E402
from team_colour import train  # noqa: E402

OUT = os.path.join(ROOT, "data", "cleanup")
SCALE = np.array(GAMES[GAME]["pitch"], np.float32)
PAIR_M = 1.5      # detection <-> Metrica pairing radius (the two agree to ~0.2 m median)
VOTE = 3          # +/- frames of colour votes: a duel swap can last < 1 s, a wider window mixes the two players
MIN_BOXES = 4
SURE = 0.8        # mean colour probability needed to overrule the tracking
MISS_M = 2.0
OTHER_SHARE = 0.1  # a detector track paired with Metrica less than this is an official


def load_detections():
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(ROOT, "data", "night", "tactical", "*.npz")))]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("frame", "cls", "track", "foot", "hist")}
    keep = (cat["cls"] == 0) & np.all((cat["foot"] > -0.02) & (cat["foot"] < 1.02), axis=1)
    return {k: v[keep] for k, v in cat.items()}


def pair(det, players):
    """Slot paired with each detection (-1 = none), one-to-one per frame within PAIR_M."""
    import lap
    slot = np.full(len(det["frame"]), -1, np.int32)
    starts = np.flatnonzero(np.r_[True, np.diff(det["frame"]) != 0])
    for a, z in zip(starts, np.r_[starts[1:], len(slot)]):
        f = det["frame"][a]
        on = np.flatnonzero(np.isfinite(players[f, :, 0]))
        if not len(on):
            continue
        cost = np.linalg.norm(det["foot"][a:z, None] * SCALE - players[f, on][None] * SCALE, axis=2)
        _, d_to_p, _ = lap.lapjv(cost, extend_cost=True, cost_limit=PAIR_M)
        ok = d_to_p >= 0
        slot[a:z][ok] = on[d_to_p[ok]]
    return slot


def colour_model(det, slot, team):
    """p (n,3) over H, A, other for every detection; prints a held-out check."""
    x = det["hist"].astype(np.float32)
    y = np.full(len(x), -1)
    y[slot >= 0] = np.where(team[slot[slot >= 0]] == HOME, 0, 1)
    tr = pd.DataFrame({"t": det["track"], "paired": slot >= 0}).groupby("t").paired.agg(["mean", "size"])
    officials = tr[(tr["mean"] < OTHER_SHARE) & (tr["size"] >= 50)].index
    y[(slot < 0) & np.isin(det["track"], officials) & (det["track"] >= 0)] = 2
    rng = np.random.default_rng(0)
    half2 = det["frame"] >= 100000

    def fit(mask):
        idx = rng.choice(np.flatnonzero(mask), min(300000, mask.sum()), replace=False)
        import torch
        xt, yt = torch.tensor(x[idx]), torch.tensor(y[idx])
        model = torch.nn.Linear(x.shape[1], 3)
        opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(xt), yt).backward()
            opt.step()
        return lambda z: torch.softmax(model(torch.tensor(z)), 1).detach().numpy()

    p = fit((y >= 0) & ~half2)(x[(y >= 0) & half2])
    yt = y[(y >= 0) & half2]
    sure = p.max(1) >= SURE
    print("colour held-out (trained 1st half, tested 2nd): {} boxes, agree {:.4f}; "
          "confident {:.3f} of boxes, agree there {:.4f}; officials class {} boxes".format(
              len(yt), (p.argmax(1) == yt).mean(), sure.mean(), (p.argmax(1) == yt)[sure].mean(),
              int((y == 2).sum())), flush=True)
    return fit(y >= 0)(x)


def window_mean(values, counts, vote):
    """Centred +/- vote sums along axis 0 via cumulative sums."""
    cv = np.concatenate([np.zeros_like(values[:1]), np.cumsum(values, 0)])
    cc = np.concatenate([np.zeros_like(counts[:1]), np.cumsum(counts, 0)])
    i = np.arange(len(values))
    lo, hi = np.maximum(0, i - vote), np.minimum(len(values), i + vote + 1)
    return cv[hi] - cv[lo], cc[hi] - cc[lo]


def main():
    os.makedirs(OUT, exist_ok=True)
    atd = np.load(os.path.join(ROOT, "data", "diagnostics", "broadcast", "atd_tactical.npz"))
    players, ball = atd["players"], atd["ball"]
    team = team_of_each_track(require(["atd_meta"])[0])
    F, S = players.shape[:2]

    det = load_detections()
    print("detections:", len(det["frame"]), flush=True)
    slot = pair(det, players)
    print("paired to Metrica: {:.3f}".format((slot >= 0).mean()), flush=True)
    p = colour_model(det, slot, team)

    # Original nearest-player team, as the baseline does.
    dist = np.linalg.norm((players - ball[:, None]) * SCALE, axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    k = dist.argmin(1)
    dk = dist.min(1)
    has = np.isfinite(dk)
    orig = np.where(team[k] == HOME, HOME, AWAY).astype(np.int8)
    orig[~has] = NOBODY
    fixed = orig.copy()

    # Team check: colour votes of boxes paired with each slot, +/- VOTE frames.
    sel = slot >= 0
    votes = np.zeros((F, S, 3), np.float32)
    cnt = np.zeros((F, S), np.float32)
    np.add.at(votes, (det["frame"][sel], slot[sel]), p[sel])
    np.add.at(cnt, (det["frame"][sel], slot[sel]), 1)
    rows = np.flatnonzero(has)
    vs, cs = window_mean(votes[:, :, :2], cnt, VOTE)
    v, c = vs[rows, k[rows]], cs[rows, k[rows]]
    mean = v / np.maximum(c, 1)[:, None]
    other = np.where(team[k[rows]] == HOME, 1, 0)  # column of the other team
    swap = (c >= MIN_BOXES) & (mean[np.arange(len(rows)), other] >= SURE)
    fixed[rows[swap]] = np.where(other[swap] == 0, HOME, AWAY)
    del votes, vs, cs

    # Missing carrier: nobody of Metrica near the ball, an unpaired non-official box closer.
    un = (slot < 0) & (p[:, 2] < 0.5)
    bd = np.linalg.norm((det["foot"] - ball[det["frame"]]) * SCALE, axis=1)
    cand = un & (bd < MISS_M) & (bd < dk[det["frame"]] - 1.0) & (dk[det["frame"]] >= MISS_M)
    # Colour of the candidate's detector track over +/- VOTE frames.
    tdf = pd.DataFrame({"f": det["frame"], "t": det["track"], "ph": p[:, 0], "pa": p[:, 1]})
    missing = 0
    for i in np.flatnonzero(cand):
        f, t = det["frame"][i], det["track"][i]
        if t < 0:
            continue
        g = tdf[(tdf.t == t) & (tdf.f >= f - VOTE) & (tdf.f <= f + VOTE)]
        if len(g) >= MIN_BOXES:
            ph, pa = g.ph.mean(), g.pa.mean()
            if max(ph, pa) >= SURE and fixed[f] == orig[f]:
                fixed[f] = HOME if ph > pa else AWAY
                missing += fixed[f] != orig[f]

    alive = has.sum()
    changed = fixed != orig
    edges = np.diff(np.r_[0, changed.astype(np.int8), 0])
    ep = pd.DataFrame({"first": np.flatnonzero(edges == 1), "last": np.flatnonzero(edges == -1) - 1})
    ep["frames"] = ep["last"] - ep["first"] + 1
    ep["from"] = np.array(["H", "A"])[orig[ep["first"]]]
    ep["to"] = np.array(["H", "A"])[fixed[ep["first"]]]
    ep["video_time"] = ["{}:{:02d}".format(f // 1500, f // 25 % 60) for f in ep["first"]]
    ep.to_csv(os.path.join(OUT, "carrier_episodes.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "carrier_codes.npz"), orig=orig, fixed=fixed,
                        swap_rows=rows[swap], det_frame=det["frame"], det_slot=slot,
                        det_track=det["track"], det_p=p.astype(np.float16))
    print("frames with ball {}: team flipped by colour {} ({:.2%}), missing carrier added {} ({:.2%})".format(
        alive, int(swap.sum() - 0), swap.sum() / alive, missing, missing / alive))
    print("episodes {} (median {} frames, >= 25 frames: {})".format(
        len(ep), int(ep.frames.median()) if len(ep) else 0, int((ep.frames >= 25).sum())))
    for w in (25, 50):
        a, b = smooth(orig, w), smooth(fixed, w)
        spells = lambda s: int((s[1:] != s[:-1]).sum())  # noqa: E731
        print("smooth{}: output frames changed {} ({:.2%}); label changes {} -> {}".format(
            w, int((a != b).sum()), (a != b).mean(), spells(a), spells(b)))


if __name__ == "__main__":
    main()
