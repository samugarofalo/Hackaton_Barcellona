"""Ball in 3D from the two cameras: tactical and broadcast see the same ball from two points.

Cameras: each camera's centre is fixed through the match (it pans and zooms), and is found from the
players -- the line through a player's head point projected on the grass (homography) and his head
in space (foot point, 1.8 m up) passes through the camera; the least-squares meeting point of tens
of thousands of such lines is the centre. Both estimates are stable on independent halves of the
data (`cameras()` prints them).

Ball: our tactical ball (ball_track.py) and our broadcast ball (broadcast_ball.py, linked with the
same trajectory filter), each put on the grass by its homography, give two rays. Where both exist
in the same tactical frame, the ball is the midpoint of the shortest segment between the rays;
`gap` is that segment's length (how well the rays really meet).

Checks with no ground truth, printed by main():
  - gap: the rays must nearly meet;
  - ground balls (fitted height < 0.3 m): the two grass points must coincide;
  - flights: the fitted height must fall with g = 9.81 m/s^2;
  - time alignment: the broadcast -> tactical offset is refined within +/- 3 frames by the best
    agreement, per half.

    .venv/bin/python pipeline/triangulate.py
Writes data/cleanup/cameras.npz and data/cleanup/ball_3d.npz (tactical frame, xyz in metres, gap).
Prints only aggregates.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from load_homography import load as load_homography, unproject  # noqa: E402
from ball_link import link, select, MIN_CONF  # noqa: E402

SCALE = np.array(GAMES[GAME]["pitch"], np.float64)
HEAD_M = 1.8
OUT = os.path.join(ROOT, "data", "cleanup")


def camera_centre(source, files, step=25):
    mat, _, _ = load_homography(source=source)
    A, b = np.zeros((3, 3)), np.zeros(3)
    for p in files:
        c = np.load(p)
        m = (c["cls"] == 0) & (c["conf"] > 0.6)
        f, bb = c["frame"][m], c["xyxy"][m]
        for k in np.unique(f)[::step]:
            j = f == k
            x = (bb[j, 0] + bb[j, 2]) / 2
            fx, fy = unproject(mat[k], x, bb[j, 3])
            hx, hy = unproject(mat[k], x, bb[j, 1])
            F = np.c_[fx * SCALE[0], fy * SCALE[1], np.full(len(x), HEAD_M)]
            Gh = np.c_[hx * SCALE[0], hy * SCALE[1], np.zeros(len(x))]
            ok = np.all(np.isfinite(F), 1) & np.all(np.isfinite(Gh), 1) & (np.linalg.norm(F[:, :2] - Gh[:, :2], axis=1) > 0.5)
            u = F[ok] - Gh[ok]
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            P = np.eye(3)[None] - u[:, :, None] * u[:, None, :]
            A += P.sum(0)
            b += np.einsum("nij,nj->i", P, Gh[ok])
    return np.linalg.solve(A, b)


def cameras():
    path = os.path.join(OUT, "cameras.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["tactical"], z["broadcast"]
    out = {}
    for name in ("tactical", "broadcast"):
        files = sorted(glob.glob(os.path.join(ROOT, "data", "night", name, "*.npz")))
        out[name] = camera_centre(name, files[::2])
        other = camera_centre(name, files[1::2])
        print("{} camera: {} m (independent half of the data: {} m)".format(
            name, np.round(out[name], 1), np.round(other, 1)), flush=True)
    np.savez(path, **out)
    return out["tactical"], out["broadcast"]


def broadcast_track():
    """Linked + selected broadcast ball, one detection per broadcast frame: (broadcast frame,
    mapped tactical frame, ground xy in metres)."""
    z = np.load(os.path.join(OUT, "ball_det_broadcast.npz"))
    f, c, bb, xy, tac = z["frame"], z["conf"], z["xyxy"], z["xy"], z["tactical"]
    on = np.all((xy * SCALE > -1) & (xy * SCALE < SCALE + 1), axis=1)  # spare balls off the pitch
    f, c, bb, xy, tac = f[on], c[on], bb[on], xy[on], tac[on]
    px = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
    size = np.median((bb[:, 2] - bb[:, 0] + bb[:, 3] - bb[:, 1]) / 2)
    keep, tid = link(f, px, c, size, 4, 8)
    keep &= c >= MIN_CONF
    pick = select(f, px, c, tid, keep, size, 4)
    return f[pick], tac[pick], xy[pick] * SCALE


def tactical_track():
    out = {}
    for path in glob.glob(os.path.join(OUT, "ball_track_*.npz")):
        t = np.load(path)
        a = int(t["start"])
        for i in np.flatnonzero(np.isfinite(t["xy"][:, 0])):
            out[a + i] = t["xy"][i] * SCALE
    return out


def meet(c1, g1, c2, g2):
    """Closest points of rays c1->g1 and c2->g2 (g on the grass, z = 0). Returns midpoint, gap."""
    p1, p2 = np.c_[g1, np.zeros(len(g1))], np.c_[g2, np.zeros(len(g2))]
    d1, d2 = p1 - c1, p2 - c2
    w = c1 - c2
    a, b, c = np.sum(d1 * d1, 1), np.sum(d1 * d2, 1), np.sum(d2 * d2, 1)
    d, e = np.sum(d1 * w, 1), np.sum(d2 * w, 1)
    den = a * c - b * b
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    q1, q2 = c1 + s[:, None] * d1, c2 + t[:, None] * d2
    return (q1 + q2) / 2, np.linalg.norm(q1 - q2, axis=1)


def main():
    ct, cb = cameras()
    tac = tactical_track()
    bf, bt, bxy = broadcast_track()
    print("broadcast ball after the trajectory filter: {} frames".format(len(bf)), flush=True)

    # offset refinement per half: tactical frame = mapped + delta
    best_delta = {}
    for half, a, z in half_video_blocks(GAME):
        sel = (bt >= a) & (bt <= z)
        scores = {}
        for delta in range(-3, 4):
            frames = bt[sel] + delta
            ok = np.array([f in tac for f in frames])
            if ok.sum() < 100:
                continue
            g1 = np.array([tac[f] for f in frames[ok]])
            _, gap = meet(ct, g1, cb, bxy[sel][ok])
            scores[delta] = np.median(gap)
        best_delta[half] = min(scores, key=scores.get)
        print("half {}: median ray gap by offset {}".format(half, {k: round(v, 2) for k, v in scores.items()}))

    frames, pts, gaps, gt_, gb_ = [], [], [], [], []
    for half, a, z in half_video_blocks(GAME):
        sel = (bt >= a) & (bt <= z)
        fr = bt[sel] + best_delta[half]
        ok = np.array([f in tac for f in fr])
        g1 = np.array([tac[f] for f in fr[ok]])
        p, gap = meet(ct, g1, cb, bxy[sel][ok])
        frames.append(fr[ok]); pts.append(p); gaps.append(gap); gt_.append(g1); gb_.append(bxy[sel][ok])
    frames, pts, gaps = np.concatenate(frames), np.concatenate(pts), np.concatenate(gaps)
    gt_, gb_ = np.concatenate(gt_), np.concatenate(gb_)
    o = np.argsort(frames)
    frames, pts, gaps, gt_, gb_ = frames[o], pts[o], gaps[o], gt_[o], gb_[o]
    np.savez_compressed(os.path.join(OUT, "ball_3d.npz"), frame=frames, xyz=pts, gap=gaps,
                        ground_tactical=gt_, ground_broadcast=gb_)

    graded = sum(z - a + 1 for _, a, z in half_video_blocks(GAME))
    print("frames with both balls: {} ({:.1%} of graded frames)".format(len(frames), len(frames) / graded))
    print("ray gap: p50 {:.2f} m, p90 {:.2f} m".format(*np.percentile(gaps, [50, 90])))
    good = gaps < 1.0
    zz = pts[good, 2]
    print("rays meeting within 1 m: {:.1%}; height there: p10 {:.2f} p50 {:.2f} p90 {:.2f} m; below 0 m: {:.1%}".format(
        good.mean(), *np.percentile(zz, [10, 50, 90]), (zz < -0.3).mean()))
    low = good & (pts[:, 2] < 0.3)
    print("ground balls: the two grass points differ p50 {:.2f} m, p90 {:.2f} m".format(
        *np.percentile(np.linalg.norm(gt_[low] - gb_[low], axis=1), [50, 90])))
    # gravity check on flights: >= 10 consecutive frames with good rays and height > 1 m
    acc = []
    run = good & (pts[:, 2] > 1.0)
    i = 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and run[j + 1] and frames[j + 1] == frames[j] + 1 and run[i]:
            j += 1
        if run[i] and j - i + 1 >= 10:
            t = (frames[i:j + 1] - frames[i]) / 25.0
            acc.append(2 * np.polyfit(t, pts[i:j + 1, 2], 2)[0])
        i = j + 1
    if acc:
        print("flights (>= 10 frames above 1 m): {}; vertical acceleration p25/p50/p75 {:.1f} / {:.1f} / {:.1f} m/s^2 "
              "(gravity -9.8)".format(len(acc), *np.percentile(acc, [25, 50, 75])))


if __name__ == "__main__":
    main()
