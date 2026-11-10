"""Goalkeepers: kit colour says "keeper", position says how likely, the half says whose.

1. Position prior, learned on DFL (keepers flagged, other games): for a player, depth = metres he
   stands behind the deepest teammate (towards his own goal). p_pos = logistic(depth), fitted on
   data/external/bassek2025/keeper_depth.npz. On DFL: depth >= 2 m -> keeper 99.5% of the time.
2. Colour, 4 classes H / A / GK / other: same histograms and teacher as carrier_fix.py, plus GK =
   boxes paired with a Metrica player whose p_pos >= TEACH (the two kits are alike, so one class).
3. Per detection, keeper log-odds = colour log-odds (GK vs H+A) + position log-likelihood ratio,
   the position computed for the box itself against the Metrica players of the team defending that
   half. Averaged over the detector track (continuity): a track is a keeper if the mean posterior
   >= 0.5, and each of its boxes must itself be the deepest of that half (depth >= MIN_DEPTH: on
   DFL 98.4% of keeper frames), so a track that briefly jumps to an opponent does not carry the
   label over. Its team = the team defending the half it stands in. At most one keeper per side per
   frame (highest posterior wins).

    .venv/bin/python pipeline/keepers.py
Writes data/cleanup/keepers.npz (per detection, same order as carrier_codes.npz: keeper flag,
keeper team, posterior) and prints only aggregate checks.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import require, half_video_blocks, GAME  # noqa: E402
from baseline import team_of_each_track, HOME  # noqa: E402
from carrier_fix import load_detections, SCALE  # noqa: E402

TEACH = 0.99
MIN_DEPTH = 0.0
OTHER_SHARE = 0.1


def fit_depth_prior():
    import torch
    z = np.load(os.path.join(ROOT, "data", "external", "bassek2025", "keeper_depth.npz"))
    g, k = torch.tensor(z["g"]).clamp(-60, 60), torch.tensor(z["keeper"].astype(np.float32))
    a, b = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.05)
    for _ in range(500):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(a * g / 10 + b, k).backward()
        opt.step()
    a, b, prior = float(a) / 10, float(b), float(k.mean())
    print("DFL depth prior: logit = {:.3f} * depth_m + {:.3f}; keeper share {:.3f}".format(a, b, prior))
    return lambda depth: 1 / (1 + np.exp(-(a * np.clip(depth, -60, 60) + b))), prior


def defending_left(players, team):
    """{(half, 'H'/'A'): True if that team defends the low-x goal}, from mean Metrica positions."""
    out = {}
    for half, a, z in half_video_blocks(GAME):
        x = players[a:z + 1, :, 0]
        for t in ("H", "A"):
            out[(half, t)] = np.nanmean(x[:, team == t]) < 0.5
    return out


def half_of(frames):
    h = np.zeros(len(frames), int)
    for half, a, z in half_video_blocks(GAME):
        h[(frames >= a) & (frames <= z)] = half
    return h


def main():
    p_pos, prior = fit_depth_prior()
    atd = np.load(os.path.join(ROOT, "data", "diagnostics", "broadcast", "atd_tactical.npz"))
    players = atd["players"]
    team = np.where(team_of_each_track(require(["atd_meta"])[0]) == HOME, "H", "A")
    side = defending_left(players, team)
    cols = {t: np.flatnonzero(team == t) for t in ("H", "A")}

    det = load_detections()
    z = np.load(os.path.join(ROOT, "data", "cleanup", "carrier_codes.npz"))
    slot, track = z["det_slot"], z["det_track"]
    f, foot = det["frame"], det["foot"]
    half = half_of(f)
    n = len(f)

    # Depth of every detection, against the Metrica players of the team defending its half.
    depth = np.full(n, np.nan)
    owner = np.full(n, "", dtype="<U1")
    for hf in (1, 2):
        for t in ("H", "A"):
            left = side[(hf, t)]
            sel = np.flatnonzero((half == hf) & ((foot[:, 0] < 0.5) == left))
            owner[sel] = t
            for a in range(0, len(sel), 200000):
                i = sel[a:a + 200000]
                mates = players[f[i]][:, cols[t]]  # (m, k, 2)
                same = np.hypot((mates[..., 0] - foot[i, None, 0]) * SCALE[0],
                                (mates[..., 1] - foot[i, None, 1]) * SCALE[1]) <= 1.5
                mx = np.where(same | np.isnan(mates[..., 0]), np.nan, mates[..., 0])
                dist = (lambda v: v * SCALE[0]) if left else (lambda v: (1 - v) * SCALE[0])
                with np.errstate(all="ignore"):
                    depth[i] = np.nanmin(dist(mx), axis=1) - dist(foot[i, 0])
    pp = np.where(np.isfinite(depth), p_pos(np.nan_to_num(depth, nan=-60)), prior)
    print("position: detections with p_keeper >= 0.5: {:.2%}".format((pp >= 0.5).mean()), flush=True)

    # Colour teacher: H / A / GK / other.
    y = np.full(n, -1)
    paired = slot >= 0
    y[paired] = np.where(team[slot[paired]] == "H", 0, 1)
    y[paired & (pp >= TEACH)] = 2
    tr = pd.DataFrame({"t": track, "paired": paired}).groupby("t").paired.agg(["mean", "size"])
    officials = tr[(tr["mean"] < OTHER_SHARE) & (tr["size"] >= 50)].index
    y[~paired & np.isin(track, officials) & (track >= 0)] = 3
    print("teacher boxes: H {} A {} GK {} other {}".format(*[(y == c).sum() for c in range(4)]), flush=True)

    import torch
    x = det["hist"].astype(np.float32)
    rng = np.random.default_rng(0)

    def fit(mask):
        idx = np.flatnonzero(mask)
        # GK is rare: keep all of its boxes, subsample the rest
        gk = idx[y[idx] == 2]
        rest = rng.choice(idx[y[idx] != 2], min(300000, (y[idx] != 2).sum()), replace=False)
        idx = np.r_[gk, rest]
        xt, yt = torch.tensor(x[idx]), torch.tensor(y[idx])
        model = torch.nn.Linear(x.shape[1], 4)
        opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(xt), yt).backward()
            opt.step()
        return lambda q: torch.softmax(model(torch.tensor(q)), 1).detach().numpy()

    h2 = half == 2
    pc = fit((y >= 0) & ~h2)(x[(y >= 0) & h2])
    yt = y[(y >= 0) & h2]
    for c, name in enumerate(("H", "A", "GK", "other")):
        m = yt == c
        print("held-out (2nd half) {:>5}: {} boxes, recalled {:.3f}; of boxes called {}, right {:.3f}".format(
            name, m.sum(), (pc[m].argmax(1) == c).mean(), name, (yt[pc.argmax(1) == c] == c).mean()))
    pc = fit(y >= 0)(x)

    # Combine colour and position, then average over the detector track.
    eps = 1e-4
    lo_colour = np.log(pc[:, 2] + eps) - np.log(pc[:, 0] + pc[:, 1] + eps)
    lo_pos = np.log(pp + eps) - np.log(1 - pp + eps) - np.log(prior / (1 - prior))
    lo = lo_colour + lo_pos
    df = pd.DataFrame({"t": track, "lo": lo, "f": f, "owner": owner, "o": pc[:, 3]})
    track_lo = df[df.t >= 0].groupby("t").lo.mean()
    df["post"] = 1 / (1 + np.exp(-df.t.map(track_lo).fillna(df.lo)))
    df.loc[df.o >= 0.5, "post"] = 0.0
    keeper = ((df.post >= 0.5) & (np.nan_to_num(depth, nan=-99) >= MIN_DEPTH)).to_numpy(copy=True)
    # at most one keeper per side (owner team) per frame
    cand = df[keeper].sort_values("post", ascending=False)
    dup = cand.duplicated(["f", "owner"])
    keeper[cand.index[dup]] = False
    gk_team = np.where(keeper, owner, "")
    np.savez_compressed(os.path.join(ROOT, "data", "cleanup", "keepers.npz"), keeper=keeper,
                        team=gk_team, post=df.post.to_numpy(np.float32), p_colour=pc.astype(np.float16),
                        depth=depth.astype(np.float32))
    per_frame = pd.Series(keeper).groupby(f).sum()
    print("keeper boxes {:.2%} of detections; frames with 0/1/2 keepers: {}".format(
        keeper.mean(), per_frame.reindex(np.unique(f), fill_value=0).value_counts().sort_index().to_dict()))
    print("keepers by team: H {}  A {}".format((gk_team == "H").sum(), (gk_team == "A").sum()))


if __name__ == "__main__":
    main()
