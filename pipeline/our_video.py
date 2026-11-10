"""Debug video: our detections on the tactical video, players and ball together. To watch, not to label.

  players   night-run boxes (yolo26x), coloured by the shirt-colour classifier of carrier_fix.py:
            blue = H (Barcelona), red = A (Athletic), yellow = official, grey = unsure (< SURE)
  ball      ball_track.py (our detector, trajectory filter, no jumps): green circle + 1 s trail
  inset     4x zoom on the ball (top right)

    .venv/bin/python pipeline/our_video.py --start 20000 --frames 4500
Writes data/cleanup/our_video_<start>_<last>.mp4. Prints only aggregate counts.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import asset_paths  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402
from compare_tracking import decode, encoder  # noqa: E402

SURE = 0.8
COLOURS = {0: (255, 120, 0), 1: (40, 40, 230), 2: (0, 230, 255)}  # BGR: H blue, A red, official yellow
UNSURE = (170, 170, 170)
TRAIL, ZOOM, CROP = 25, 4, 90


def players(start, n):
    """Boxes + colour probabilities, in the order carrier_fix.load_detections saved them."""
    xyxy = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "night", "tactical", "*.npz"))):
        c = np.load(p)
        keep = (c["cls"] == 0) & np.all((c["foot"] > -0.02) & (c["foot"] < 1.02), axis=1)
        xyxy.append(c["xyxy"][keep])
    xyxy = np.concatenate(xyxy)
    z = np.load(os.path.join(ROOT, "data", "cleanup", "carrier_codes.npz"))
    f, p = z["det_frame"], z["det_p"].astype(np.float32)
    assert len(f) == len(xyxy)
    m = (f >= start) & (f < start + n)
    return f[m], xyxy[m], p[m]


def ball(start, n):
    for path in glob.glob(os.path.join(ROOT, "data", "cleanup", "ball_track_*.npz")):
        t = np.load(path)
        a = int(t["start"])
        if a <= start and start + n <= a + len(t["xy"]):
            return t["xy"][start - a:start - a + n]
    raise SystemExit("no ball_track_*.npz covers the range: run ball_track.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=4500)
    a = ap.parse_args()
    start, n = a.start, a.frames
    pf, pbox, pp = players(start, n)
    bxy = ball(start, n)
    matrices, _, _ = load_homography()
    rows = {}
    for i, f in enumerate(pf):
        rows.setdefault(int(f), []).append(i)

    out = os.path.join(ROOT, "data", "cleanup", "our_video_{}_{}.mp4".format(start, start + n - 1))
    enc = encoder(out)
    trail, n_ball, unsure = [], 0, 0
    for k, img in enumerate(decode(asset_paths()["video"], start, n)):
        f = start + k
        counts = [0, 0, 0]
        for i in rows.get(f, []):
            x1, y1, x2, y2 = map(int, pbox[i])
            cls = int(np.argmax(pp[i]))
            sure = pp[i, cls] >= SURE
            col = COLOURS[cls] if sure else UNSURE
            counts[cls] += sure
            unsure += not sure
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, "HAR"[cls] if sure else "?", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        pt = None
        if np.isfinite(bxy[k, 0]):
            x, y = project(matrices[f], bxy[k, 0], bxy[k, 1])
            if np.isfinite(x):
                pt = (int(x), int(y))
                n_ball += 1
        trail = (trail + [pt])[-TRAIL:]
        pts = [q for q in trail if q]
        for q0, q1 in zip(pts, pts[1:]):
            cv2.line(img, q0, q1, (0, 255, 0), 1)
        if pt:
            cv2.circle(img, pt, 14, (0, 255, 0), 2)
            x0 = int(np.clip(pt[0] - CROP, 0, img.shape[1] - 2 * CROP))
            y0 = int(np.clip(pt[1] - CROP, 0, img.shape[0] - 2 * CROP))
            inset = cv2.resize(img[y0:y0 + 2 * CROP, x0:x0 + 2 * CROP], None, fx=ZOOM / 2, fy=ZOOM / 2,
                               interpolation=cv2.INTER_NEAREST)
            h, w = inset.shape[:2]
            img[10:10 + h, img.shape[1] - w - 10:img.shape[1] - 10] = inset
            cv2.rectangle(img, (img.shape[1] - w - 10, 10), (img.shape[1] - 10, 10 + h), (255, 255, 255), 1)
        text = "frame {}  {}:{:02d}   H {}  A {}  ref {}   ball {}".format(
            f, f // 1500, f // 25 % 60, counts[0], counts[1], counts[2], "yes" if pt else "-")
        cv2.rectangle(img, (0, 0), (720, 34), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    print("{}\n{} frames; ball shown in {:.1%}; unsure player boxes {:.1%}".format(
        out, n, n_ball / n, unsure / max(len(pf), 1)))


if __name__ == "__main__":
    main()
