"""Map broadcast frames to tactical frames from the two ATDs alone (no images).

Same instant => same players in the same places on the pitch. For each continuous broadcast
stretch with >= MIN_PLAYERS tracked players and >= MIN_SECONDS long, three frames spread over it
are compared with every tactical frame by a one-sided Chamfer distance: the mean, over broadcast
players, of the distance in metres to the nearest tactical player. The best tactical frame for each
gives an offset (tactical - broadcast); the three must agree within AGREE frames, and the stretch's
offset is then refined frame by frame over the whole stretch within +/- REFINE.

    python pipeline/broadcast_align.py
Needs data/diagnostics/broadcast/atd_{tactical,broadcast}.npz (cached raw ATD positions).
Writes data/diagnostics/broadcast/alignment.csv: one row per stretch.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
from games import GAME, GAMES  # noqa: E402

OUT = os.path.join(ROOT, "data", "diagnostics", "broadcast")
MIN_PLAYERS, MIN_SECONDS, AGREE, REFINE = 10, 10, 5, 12
SCALE = np.array(GAMES[GAME]["pitch"], dtype=np.float32)


def chamfer_all(query, tac):
    """query (k,2) metres; tac (N,46,2) metres with NaN -> (N,) mean nearest distance."""
    out = np.empty(len(tac), np.float32)
    for a in range(0, len(tac), 20000):
        t = tac[a:a + 20000]
        d = np.linalg.norm(t[:, :, None, :] - query[None, None], axis=3)  # (n,46,k)
        d = np.where(np.isnan(d), np.inf, d).min(axis=1)
        out[a:a + 20000] = d.mean(axis=1)
    return out


def chamfer_pairs(bq, tq):
    """Per-pair mean nearest distance; bq (m,51,2) tq (m,46,2)."""
    d = np.linalg.norm(bq[:, :, None] - tq[:, None], axis=3)
    d = np.where(np.isnan(d), np.inf, d).min(axis=2)  # (m,51): nearest tactical per bcast
    d = np.where(np.isfinite(d), d, np.nan)
    return np.nanmean(d, axis=1)


def main():
    tac_npz = np.load(os.path.join(OUT, "atd_tactical.npz"))
    bro_npz = np.load(os.path.join(OUT, "atd_broadcast.npz"))
    tac = tac_npz["players"] * SCALE
    bro = bro_npz["players"] * SCALE
    count = np.isfinite(bro[:, :, 0]).sum(axis=1)
    edge = np.diff(np.r_[0, (count >= MIN_PLAYERS).astype(int), 0])
    starts, ends = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1

    # Orientation check on the longest stretch: plain vs pitch rotated 180 degrees.
    longest = int(np.argmax(ends - starts))
    mid = (starts[longest] + ends[longest]) // 2
    q = bro[mid][np.isfinite(bro[mid, :, 0])]
    plain = chamfer_all(q, tac).min()
    flipped = chamfer_all(SCALE - q, tac).min()
    flip = flipped < plain
    print("orientation: plain {:.2f} m, rotated {:.2f} m -> {}".format(
        plain, flipped, "rotated" if flip else "plain"))
    if flip:
        bro = SCALE - bro

    rows = []
    for a, z in zip(starts, ends):
        if (z - a + 1) < MIN_SECONDS * 25:
            continue
        picks = np.linspace(a + 12, z - 12, 3).astype(int)
        offsets, scores = [], []
        for b in picks:
            c = chamfer_all(bro[b][np.isfinite(bro[b, :, 0])], tac)
            t = int(np.nanargmin(c))
            offsets.append(t - b)
            scores.append(float(c[t]))
        agree = max(offsets) - min(offsets) <= AGREE
        best, best_cost = None, np.inf
        if agree:
            frames = np.arange(a, z + 1)
            for off in range(int(np.median(offsets)) - REFINE, int(np.median(offsets)) + REFINE + 1):
                t = frames + off
                ok = (t >= 0) & (t < len(tac))
                cost = np.nanmedian(chamfer_pairs(bro[frames[ok]], tac[t[ok]]))
                if cost < best_cost:
                    best, best_cost = off, cost
        rows.append({"b_first": int(a), "b_last": int(z), "seconds": round((z - a + 1) / 25, 1),
                     "coarse_offsets": offsets, "coarse_cost_m": [round(s, 2) for s in scores],
                     "agree": agree, "offset": best,
                     "t_first": int(a + best) if best is not None else None,
                     "t_last": int(z + best) if best is not None else None,
                     "median_cost_m": round(float(best_cost), 3) if best is not None else None})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "alignment.csv"), index=False)


if __name__ == "__main__":
    main()
