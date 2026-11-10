"""Debug video: our ball detector on the tactical video, next to the Metrica ball. To watch, not to label.

Drawn on each frame:
  thin yellow circles   every detection with conf >= SHOW_CONF (number = confidence)
  thick green circle    the ball kept by ball_track.py (trajectory filter, no Metrica gate) when a
                        ball_track_*.npz covers the range; else highest confidence within GATE_M of Metrica
  magenta cross         the Metrica ball projected into the image
  inset (top right)     4x zoom around the kept ball (or the Metrica ball if none)

    .venv/bin/python pipeline/ball_video.py --start 110808 --frames 3000
Reads the newest data/cleanup/ball_det_*.npz that covers the range; writes
data/cleanup/ball_video_<start>_<last>.mp4.
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
from games import GAME, GAMES, asset_paths  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402
from compare_tracking import decode, encoder  # noqa: E402

SCALE = np.array(GAMES[GAME]["pitch"], np.float32)
SHOW_CONF, GATE_M, ZOOM, CROP = 0.2, 4.0, 4, 90


def detections(start, n):
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "cleanup", "ball_det_*.npz")), key=os.path.getmtime)[::-1]:
        a, z = map(int, os.path.basename(path)[9:-4].split("_"))
        if a <= start and start + n - 1 <= z:
            return np.load(path)
    raise SystemExit("no ball_det_*.npz covers {}..{}: run ball_eval.py first".format(start, start + n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=110808)
    ap.add_argument("--frames", type=int, default=3000)
    a = ap.parse_args()
    start, n = a.start, a.frames
    z = detections(start, n)
    metrica = np.load(os.path.join(ROOT, "data", "diagnostics", "broadcast", "atd_tactical.npz"))["ball"]
    matrices, _, _ = load_homography()
    tracked = None
    for path in glob.glob(os.path.join(ROOT, "data", "cleanup", "ball_track_*.npz")):
        t = np.load(path)
        if int(t["start"]) <= start and start + n <= int(t["start"]) + len(t["xy"]):
            tracked = (int(t["start"]), t["xy"])
    by_frame = {}
    for i in np.flatnonzero((z["frame"] >= start) & (z["frame"] < start + n) & (z["conf"] >= SHOW_CONF)):
        by_frame.setdefault(int(z["frame"][i]), []).append(i)

    out = os.path.join(ROOT, "data", "cleanup", "ball_video_{}_{}{}.mp4".format(
        start, start + n - 1, "_linked" if tracked else ""))
    enc = encoder(out)
    kept = 0
    for k, img in enumerate(decode(asset_paths()["video"], start, n)):
        f = start + k
        m = metrica[f]
        mpx = None
        if np.isfinite(m[0]):
            px, py = project(matrices[f], m[0], m[1])
            if np.isfinite(px):
                mpx = (int(px), int(py))
        best, best_c = None, -1
        for i in by_frame.get(f, []):
            x1, y1, x2, y2 = z["xyxy"][i]
            c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(img, c, 10, (0, 220, 255), 1)
            cv2.putText(img, "{:.2f}".format(z["conf"][i]), (c[0] + 12, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
            d = np.linalg.norm((z["xy"][i] - m) * SCALE) if np.isfinite(m[0]) else np.inf
            pick = np.allclose(z["xy"][i], tracked[1][f - tracked[0]]) if tracked else \
                d < GATE_M and z["conf"][i] > best_c
            if pick:
                best, best_c = (c, d), z["conf"][i]
        if best:
            kept += 1
            cv2.circle(img, best[0], 14, (0, 255, 0), 2)
        if mpx:
            cv2.drawMarker(img, mpx, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
        centre = best[0] if best else mpx
        if centre:
            x0 = int(np.clip(centre[0] - CROP, 0, img.shape[1] - 2 * CROP))
            y0 = int(np.clip(centre[1] - CROP, 0, img.shape[0] - 2 * CROP))
            inset = cv2.resize(img[y0:y0 + 2 * CROP, x0:x0 + 2 * CROP], None, fx=ZOOM / 2, fy=ZOOM / 2,
                               interpolation=cv2.INTER_NEAREST)
            h, w = inset.shape[:2]
            img[10:10 + h, img.shape[1] - w - 10:img.shape[1] - 10] = inset
            cv2.rectangle(img, (img.shape[1] - w - 10, 10), (img.shape[1] - 10, 10 + h), (255, 255, 255), 1)
        text = "frame {}  {}:{:02d}  ".format(f, f // 1500, f // 25 % 60)
        text += "ball conf {:.2f}, {:.1f} m from Metrica".format(best_c, best[1]) if best and np.isfinite(best[1]) \
            else "ball conf {:.2f}, Metrica: no ball".format(best_c) if best else \
            ("no detection near Metrica" if mpx else "Metrica: no ball")
        cv2.rectangle(img, (0, 0), (760, 34), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    print("{}  ({} frames, ball kept in {:.1%})".format(out, n, kept / n))


if __name__ == "__main__":
    main()
