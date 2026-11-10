"""Debug video of method 4 (fuse.py "ours_m") on the tactical video. To watch, not to label.

  box colour    blue = H, red = A, from our shirt colour (thick) or borrowed from the Metrica players
                around (thin, label "m"); grey = still unsure; keepers (keepers.py): light blue
                "GK H" / orange "GK A", thick
  dashed box    position held through an occlusion gap (no detection that frame, interpolated)
  ball          ours: green circle + 1 s trail, 4x zoom inset top right; where ours is missing:
                cyan circle = interpolated (variant 4a), magenta cross = Metrica's ball (variant 4b)

    .venv/bin/python pipeline/method4_video.py --start 20000 --frames 4500
Writes data/cleanup/method4_video_<start>_<last>.mp4.
"""

import argparse
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
from fuse import load  # noqa: E402

COL = {"H": (255, 120, 0), "A": (40, 40, 230), None: (170, 170, 170)}
GK_COL = {"H": (255, 220, 120), "A": (0, 150, 255)}
TRAIL, ZOOM, CROP = 25, 4, 90


def dashed(img, p1, p2, col):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, 8):
        cv2.line(img, (x, y1), (min(x + 4, x2), y1), col, 1)
        cv2.line(img, (x, y2), (min(x + 4, x2), y2), col, 1)
    for y in range(y1, y2, 8):
        cv2.line(img, (x1, y), (x1, min(y + 4, y2)), col, 1)
        cv2.line(img, (x2, y), (x2, min(y + 4, y2)), col, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=4500)
    a = ap.parse_args()
    start, n = a.start, a.frames
    v = load(start, n)
    ours_m, ob = v["ours_m"]
    interp, mb = v["ball_interp"], v["metrica"][2]
    matrices, _, _ = load_homography()
    out = os.path.join(ROOT, "data", "cleanup", "method4_video_{}_{}.mp4".format(start, start + n - 1))
    enc = encoder(out)
    trail = []
    for k, img in enumerate(decode(asset_paths()["video"], start, n)):
        for q, tm, b, seen, src in ours_m.get(k, []):
            if not np.all(np.isfinite(b)):
                continue
            x1, y1, x2, y2 = map(int, b)
            col = GK_COL[tm] if src == "keeper" else COL[tm]
            if seen:
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if src == "keeper" else 2 if src == "colour" else 1)
            else:
                dashed(img, (x1, y1), (x2, y2), col)
            label = "?" if tm is None else "GK " + tm if src == "keeper" else tm + ("m" if src == "metrica" else "")
            cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        pt = None
        if np.isfinite(ob[k, 0]):
            x, y = project(matrices[start + k], ob[k, 0], ob[k, 1])
            if np.isfinite(x):
                pt = (int(x), int(y))
        if pt is None:
            for b, kind in ((interp[k], "i"), (mb[k], "m")):
                if np.isfinite(b[0]):
                    x, y = project(matrices[start + k], b[0], b[1])
                    if np.isfinite(x):
                        if kind == "i":
                            cv2.circle(img, (int(x), int(y)), 14, (255, 255, 0), 2)
                        else:
                            cv2.drawMarker(img, (int(x), int(y)), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
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
        f = start + k
        cv2.rectangle(img, (0, 0), (420, 34), (0, 0, 0), -1)
        cv2.putText(img, "method 4   frame {}  {}:{:02d}".format(f, f // 1500, f // 25 % 60), (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    print(out)


if __name__ == "__main__":
    main()
