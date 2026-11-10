"""Ball detector trained on other games (ball_dataset.py), run on a stretch of our tactical video.

No labels of this match: the detections are compared with the supplied Metrica ball and with the
generic COCO detector of the night run, to see coverage (frames with a ball) and agreement.

    .venv/bin/python pipeline/ball_eval.py [--start 110808] [--frames 10000] [--weights ...]
Writes data/cleanup/ball_det_<start>_<last>.npz (frame, conf, xyxy, pitch xy) and prints the table.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import GAME, GAMES, asset_paths  # noqa: E402
from load_homography import load as load_homography, unproject  # noqa: E402
from compare_tracking import decode  # noqa: E402

SCALE = np.array(GAMES[GAME]["pitch"], np.float32)
WEIGHTS = os.path.join(ROOT, "data", "models", "runs", "external", "ball_yolo", "runs", "s640", "weights", "best.pt")
AGREE_M = 2.0


def summary(name, frames, conf, xy, metrica, start, n, c):
    keep = conf >= c
    f, p = frames[keep], xy[keep]
    seen = np.zeros(n, bool)
    best = np.full(n, np.inf)
    has_m = np.isfinite(metrica[start:start + n, 0])
    d = np.linalg.norm((p - metrica[f]) * SCALE, axis=1)
    seen[f - start] = True
    np.minimum.at(best, f - start, np.where(np.isfinite(d), d, np.inf))
    print("{:<26} conf>={:.2f}  frames with a ball {:5.1%}  | where Metrica has it: found {:5.1%}, "
          "within {:.0f} m {:5.1%}, median dist {:.2f} m | where Metrica has none: found {:5.1%}".format(
              name, c, seen.mean(), seen[has_m].mean(), AGREE_M, (best[has_m] < AGREE_M).mean(),
              np.median(best[has_m & np.isfinite(best)]), seen[~has_m].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=110808)
    ap.add_argument("--frames", type=int, default=10000)
    ap.add_argument("--weights", default=WEIGHTS)
    a = ap.parse_args()
    start, n = a.start, a.frames
    out = os.path.join(ROOT, "data", "cleanup", "ball_det_{}_{}.npz".format(start, start + n - 1))
    if not os.path.exists(out):
        from ultralytics import YOLO
        model = YOLO(a.weights)
        matrices, _, _ = load_homography()
        rows = {"frame": [], "conf": [], "xyxy": [], "xy": []}
        for i, image in enumerate(decode(asset_paths()["video"], start, n)):
            b = model.predict(image, imgsz=1920, conf=0.05, half=True, device="0", verbose=False)[0].boxes
            if not len(b):
                continue
            xyxy = b.xyxy.cpu().numpy()
            px, py = unproject(matrices[start + i], (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2)
            rows["frame"].append(np.full(len(b), start + i))
            rows["conf"].append(b.conf.cpu().numpy())
            rows["xyxy"].append(xyxy)
            rows["xy"].append(np.c_[px, py])
        np.savez_compressed(out, **{k: np.concatenate(v) for k, v in rows.items()})
    z = np.load(out)
    metrica = np.load(os.path.join(ROOT, "data", "diagnostics", "broadcast", "atd_tactical.npz"))["ball"]

    g = {"frame": [], "conf": [], "xy": []}
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "night", "tactical", "*.npz"))):
        c = np.load(p)
        m = (c["cls"] == 32) & (c["frame"] >= start) & (c["frame"] < start + n)
        g["frame"].append(c["frame"][m]); g["conf"].append(c["conf"][m].astype(float)); g["xy"].append(c["foot"][m])
    g = {k: np.concatenate(v) for k, v in g.items()}
    print("frames {}..{}: Metrica has the ball in {:.1%}".format(
        start, start + n - 1, np.isfinite(metrica[start:start + n, 0]).mean()))
    for c in (0.05, 0.2, 0.4):
        summary("generic COCO (night)", g["frame"], g["conf"], g["xy"], metrica, start, n, c)
        summary("ball detector (ours)", z["frame"], z["conf"], z["xy"], metrica, start, n, c)


if __name__ == "__main__":
    main()
