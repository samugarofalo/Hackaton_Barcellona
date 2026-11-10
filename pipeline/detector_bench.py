"""Benchmark person detectors on one clip: speed and agreement with known people.

References, all from data already produced for the clip (run compare_tracking.py and
identities.py --kickoff first):
  - Metrica points in the image, except slots known to be frozen (--exclude-slots);
  - the fixed-roster identities (players the detector found that Metrica lacks, and officials).
Metrica is not ground truth, and the roster came from yolo26x@1920, which biases "unexplained"
in its favour. Read the numbers as relative, not absolute.

    python pipeline/detector_bench.py
Writes data/diagnostics/tracking-comparison/bench_<tag>.csv.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import GAME, GAMES  # noqa: E402
from load_homography import load as load_homography, unproject  # noqa: E402
from compare_tracking import OUT_DIR, MODEL_DIR, PITCH_MARGIN_M, MATCH_M, decode, one_to_one  # noqa: E402

VARIANTS = [  # (model, imgsz, half)
    ("yolo26x.pt", 1920, False),
    ("yolo26x.pt", 1920, True),
    ("yolo26x.pt", 2560, True),
    ("yolo26x.pt", 3200, True),
    ("yolo11x.pt", 1920, True),
    ("yolo12x.pt", 1920, True),
    ("rtdetr-x.pt", 1920, True),
]
# Hard clips (--episode): the variants that survived the kickoff bench, plus a lower threshold.
HARD = [  # (model, imgsz, half, conf)
    ("yolo26x.pt", 1920, True, 0.25),
    ("yolo26x.pt", 1920, True, 0.10),
    ("yolo26x.pt", 2560, True, 0.25),
    ("yolo26x.pt", 3200, True, 0.25),
    ("yolo11x.pt", 1920, True, 0.25),
]


def iou_pairs(xyxy, thr=0.5):
    """Number of box pairs overlapping with IoU > thr (likely duplicates)."""
    if len(xyxy) < 2:
        return 0
    x1 = np.maximum(xyxy[:, None, 0], xyxy[None, :, 0])
    y1 = np.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
    x2 = np.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
    y2 = np.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    iou = inter / (area[:, None] + area[None] - inter)
    return int((np.triu(iou, 1) > thr).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, default=11185)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--exclude-slots", type=int, nargs="*", default=[46])
    parser.add_argument("--missing-identity", type=int, default=6,
                        help="roster identity absent from Metrica (A6 on the kickoff clip)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--episode", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="hard clip: also measure boxes near the Metrica ball in FIRST..LAST")
    args = parser.parse_args()
    if args.episode:
        return hard(args)

    from ultralytics import YOLO, RTDETR
    import torch

    length, width = GAMES[GAME]["pitch"]
    scale = np.array([length, width])
    start, n = args.start, args.frames
    tag = "{}_{}".format(start, start + n - 1)
    met = pd.read_csv(os.path.join(OUT_DIR, "metrica_{}.csv".format(tag)))
    ident = pd.read_csv(os.path.join(OUT_DIR, "identities_{}.csv".format(tag)))
    met = met[(met.state != "out_of_view") & ~met.slot.isin(args.exclude_slots)]
    met_by = {f: g[["x", "y"]].values * scale for f, g in met.groupby("frame")}
    id_by = {f: g for f, g in ident.groupby("frame")}
    matrices, _, _ = load_homography()

    frames = list(decode(os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical.mp4"), start, n))
    rows = []
    for name, imgsz, half in VARIANTS:
        cls = RTDETR if name.startswith("rtdetr") else YOLO
        model = cls(os.path.join(MODEL_DIR, name))
        opts = dict(classes=[0], imgsz=imgsz, conf=args.conf, half=half, device="0",
                    verbose=False)
        model.predict(frames[0], **opts)  # warm-up
        paired = far_p = far_n = near_p = near_n = total = 0
        missing_hit = unexplained = dups = n_boxes = 0
        torch.cuda.synchronize()
        t0 = time.time()
        for i, image in enumerate(frames):
            frame = start + i
            boxes = model.predict(image, **opts)[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            fx, fy = unproject(matrices[frame], (xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3])
            pos = np.c_[fx, fy] * scale
            on = ((pos > -PITCH_MARGIN_M) & (pos < scale + PITCH_MARGIN_M)).all(axis=1)
            pos, xyxy = pos[on], xyxy[on]
            n_boxes += len(pos)
            dups += iou_pairs(xyxy)

            m = met_by.get(frame, np.zeros((0, 2)))
            m_to_b, _ = one_to_one(m, pos, MATCH_M)
            ok = m_to_b >= 0
            paired += ok.sum()
            total += len(m)
            far, near = m[:, 1] < 20, m[:, 1] > 48
            far_p, far_n = far_p + (ok & far).sum(), far_n + far.sum()
            near_p, near_n = near_p + (ok & near).sum(), near_n + near.sum()

            ids = id_by[frame]
            target = ids[ids.identity == args.missing_identity][["x_m", "y_m"]].values
            if len(target) and len(pos):
                missing_hit += np.linalg.norm(pos - target[0], axis=1).min() <= 1.5
            known = np.r_[m, ids[["x_m", "y_m"]].values]
            if len(pos):
                d = np.linalg.norm(pos[:, None] - known[None], axis=2).min(axis=1)
                unexplained += (d > MATCH_M).sum()
        torch.cuda.synchronize()
        seconds = time.time() - t0
        row = {"model": name, "imgsz": imgsz, "fp16": half,
               "fps": round(n / seconds, 2),
               "full_match_hours": round(158857 / (n / seconds) / 3600, 2),
               "metrica_recall": round(paired / total, 4),
               "far_side_recall": round(far_p / max(far_n, 1), 4),
               "near_side_recall": round(near_p / max(near_n, 1), 4),
               "missing_player_found": round(missing_hit / n, 3),
               "boxes_per_frame": round(n_boxes / n, 2),
               "unexplained_per_frame": round(unexplained / n, 3),
               "duplicate_pairs": dups}
        print(row, flush=True)
        rows.append(row)
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "bench_{}.csv".format(tag)), index=False)


def hard(args):
    """Per variant: Metrica recall in the clip and boxes near the ball during the episode."""
    from ultralytics import YOLO
    import torch
    from compare_tracking import read_atd
    from games import require

    length, width = GAMES[GAME]["pitch"]
    scale = np.array([length, width])
    start, n = args.start, args.frames
    tag = "{}_{}".format(start, start + n - 1)
    first, last = args.episode
    met = pd.read_csv(os.path.join(OUT_DIR, "metrica_{}.csv".format(tag)))
    met = met[met.state != "out_of_view"]
    met_by = {f: g[["x", "y"]].values * scale for f, g in met.groupby("frame")}
    _, ball = read_atd(require(["atd_raw"])[0], start, n)
    matrices, _, _ = load_homography()
    frames = list(decode(require(["video"])[0], start, n))
    rows = []
    for name, imgsz, half, conf in HARD:
        model = YOLO(os.path.join(MODEL_DIR, name))
        opts = dict(classes=[0], imgsz=imgsz, conf=conf, half=half, device="0", verbose=False)
        model.predict(frames[0], **opts)
        paired = total = near2 = near5 = ep = 0
        nearest = []
        t0 = time.time()
        for i, image in enumerate(frames):
            frame = start + i
            xyxy = model.predict(image, **opts)[0].boxes.xyxy.cpu().numpy()
            fx, fy = unproject(matrices[frame], (xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3])
            pos = np.c_[fx, fy] * scale
            pos = pos[((pos > -PITCH_MARGIN_M) & (pos < scale + PITCH_MARGIN_M)).all(axis=1)]
            m = met_by.get(frame, np.zeros((0, 2)))
            paired += (one_to_one(m, pos, MATCH_M)[0] >= 0).sum()
            total += len(m)
            if first <= frame <= last and np.isfinite(ball[i, 0]):
                d = np.linalg.norm(pos - ball[i] * scale, axis=1).min() if len(pos) else np.inf
                ep += 1
                near2 += d <= 2
                near5 += d <= 5
                nearest.append(d)
        row = {"clip": tag, "model": name, "imgsz": imgsz, "conf": conf,
               "fps": round(n / (time.time() - t0), 2),
               "metrica_recall": round(paired / max(total, 1), 4),
               "episode_frames": ep, "box_within_2m_of_ball": round(near2 / max(ep, 1), 3),
               "box_within_5m_of_ball": round(near5 / max(ep, 1), 3),
               "median_nearest_box_m": round(float(np.median(nearest)), 1) if nearest else None}
        print(row, flush=True)
        rows.append(row)
        del model
        torch.cuda.empty_cache()
    out = os.path.join(OUT_DIR, "bench_hard.csv")
    pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)


if __name__ == "__main__":
    main()
