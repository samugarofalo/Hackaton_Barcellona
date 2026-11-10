"""Our ball detector on the broadcast frames that show live play (the night run's plan), for
triangulation with the tactical camera.

The broadcast ball is ~14 px at 1920 (generic COCO boxes), twice the tactical one, so the detector
(trained with balls of ~6-9 px, see ball_dataset.py) runs at IMGSZ = 1280. Frames and their tactical
frame come from the night run (data/night/broadcast, column mapped_tactical).

    .venv/bin/python pipeline/broadcast_ball.py
Writes data/cleanup/ball_det_broadcast.npz: frame (broadcast), tactical (mapped tactical frame),
conf, xyxy (broadcast pixels), xy (normalised pitch via the broadcast homography).
"""

import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import asset_paths  # noqa: E402
from load_homography import load as load_homography, unproject  # noqa: E402
from compare_tracking import decode  # noqa: E402
from night_run import runs  # noqa: E402
from ball_eval import WEIGHTS  # noqa: E402

IMGSZ = 1280
OUT = os.path.join(ROOT, "data", "cleanup", "ball_det_broadcast.npz")


def plan():
    frames, mapped = [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "night", "broadcast", "*.npz"))):
        c = np.load(p)
        f, i = np.unique(c["frame"], return_index=True)
        frames.append(f)
        mapped.append(c["mapped_tactical"][i])
    f, m = np.concatenate(frames), np.concatenate(mapped)
    o = np.argsort(f)
    return f[o], m[o].astype(int)


def main():
    from ultralytics import YOLO
    frames, mapped = plan()
    to_tac = dict(zip(frames.tolist(), mapped.tolist()))
    model = YOLO(WEIGHTS)
    matrices, _, _ = load_homography(source="broadcast")
    video = asset_paths(source="broadcast")["video"]
    rows = {k: [] for k in ("frame", "tactical", "conf", "xyxy", "xy")}
    done, t0 = 0, time.time()
    for first, n in runs(frames):
        for k, img in enumerate(decode(video, first, n)):
            f = first + k
            b = model.predict(img, imgsz=IMGSZ, conf=0.05, half=True, device="0", verbose=False)[0].boxes
            if len(b):
                xyxy = b.xyxy.cpu().numpy()
                px, py = unproject(matrices[f], (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2)
                rows["frame"].append(np.full(len(b), f))
                rows["tactical"].append(np.full(len(b), to_tac[f]))
                rows["conf"].append(b.conf.cpu().numpy())
                rows["xyxy"].append(xyxy)
                rows["xy"].append(np.c_[px, py])
        done += n
        print("{}/{} frames, {:.0f} fps".format(done, len(frames), done / (time.time() - t0)), flush=True)
    np.savez_compressed(OUT, **{k: np.concatenate(v) for k, v in rows.items()})
    print(OUT)


if __name__ == "__main__":
    main()
