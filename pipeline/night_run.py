"""Overnight extraction: everything the detector can tell, saved once, decided on later.

Nothing here makes a decision (no team, no roster, no correction): it measures and saves, so that
every later choice -- corrections, possession rules, segmentation, the metric -- is recomputed from
these files in minutes without decoding video again.

Steps (each resumable: a finished chunk file is skipped):
  tactical   persons + ball (COCO 0, 32), conf >= 0.05, ByteTrack ids, foot point on the pitch,
             shirt/shorts Lab histograms (team_colour.py's features) and torso median Lab.
             Both graded halves, 158,857 frames.
  broadcast  same, on broadcast frames that show live play: homography valid, mapped to a graded
             tactical frame by the per-half linear fit of broadcast_align.py, and not a replay
             (its players match the tactical ones at that frame: Chamfer <= REPLAY_M).
  pose       yolo26x-pose on the tactical frames: 17 keypoints per person; foot point = midpoint
             of the visible ankles, on the pitch.

    python pipeline/night_run.py tactical [--limit N]
    python pipeline/night_run.py broadcast [--limit N]
    python pipeline/night_run.py pose [--limit N]

Output: data/night/<step>/chunk_<first>_<last>.npz, plus data/night/<step>.log (fps, ETA).
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
from load_homography import load as load_homography, unproject  # noqa: E402
from compare_tracking import MODEL_DIR, W, decode, shirt_colour  # noqa: E402
from team_colour import BINS, REGIONS, histogram  # noqa: E402
from broadcast_align import chamfer_pairs  # noqa: E402

OUT = os.path.join(ROOT, "data", "night")
CHUNK = 2500
CONF = 0.05
IMGSZ = 1920
REPLAY_M = 1.5
MIN_CHECK_PLAYERS = 5
SCALE = np.array(GAMES[GAME]["pitch"], dtype=np.float32)


def log(step, text):
    with open(os.path.join(OUT, step + ".log"), "a") as f:
        f.write(time.strftime("%H:%M:%S ") + text + "\n")
    print(text, flush=True)


def runs(frames):
    """Contiguous runs of sorted frame indices -> [(first, n)]."""
    frames = np.asarray(frames)
    cut = np.flatnonzero(np.diff(frames) != 1)
    return [(int(a), int(z - a + 1)) for a, z in zip(np.r_[frames[0], frames[cut + 1]],
                                                     np.r_[frames[cut], frames[-1]])]


def broadcast_plan():
    """(broadcast frames, mapped tactical frames) for live play, from alignment.csv."""
    al = pd.read_csv(os.path.join(ROOT, "data", "diagnostics", "broadcast", "alignment.csv"))
    al = al[al.agree.astype(bool)]
    tac = np.load(os.path.join(ROOT, "data", "diagnostics", "broadcast", "atd_tactical.npz"))["players"] * SCALE
    bro_npz = np.load(os.path.join(ROOT, "data", "diagnostics", "broadcast", "atd_broadcast.npz"))
    bro, valid = bro_npz["players"] * SCALE, bro_npz["valid"]
    b_all = np.arange(len(bro))
    mapped = np.full(len(bro), -1)
    for half, first, last in half_video_blocks(GAME):
        rows = al[(al.t_first >= first - 5000) & (al.t_last <= last + 5000)]
        if len(rows) < 2:
            continue
        # Median, not a line fit: a replay stretch can agree with itself on a wrong offset
        # (25692 instead of ~26920 in the second half) and a fit follows it. The drift within a
        # half is <= 3 frames.
        offset = int(np.median(rows.offset))
        t_of_b = b_all + offset
        inside = (t_of_b >= first) & (t_of_b <= last)
        mapped[inside] = t_of_b[inside]
        log("broadcast", "half {}: t = b + {}  (stretches {}, offsets {}..{})".format(
            half, offset, len(rows), int(rows.offset.min()), int(rows.offset.max())))
    keep = (mapped >= 0) & valid
    idx = np.flatnonzero(keep)
    cost = np.full(len(bro), np.nan)
    for a in range(0, len(idx), 20000):
        sel = idx[a:a + 20000]
        cost[sel] = chamfer_pairs(bro[sel], tac[mapped[sel]])
    count = np.isfinite(bro[:, :, 0]).sum(axis=1)
    # Over 1 s, so one badly tracked frame does not cut a live stretch in two.
    smooth = pd.Series(np.where(keep, cost, np.nan)).rolling(
        25, center=True, min_periods=5).median().values
    replay = (count >= MIN_CHECK_PLAYERS) & (smooth > REPLAY_M)
    live = keep & ~replay
    for a, m in runs(np.flatnonzero(live)):  # drop slivers shorter than 1 s
        if m < 25:
            live[a:a + m] = False
    log("broadcast", "frames: mapped {}  valid hom {}  replay-like {}  kept {}".format(
        int((mapped >= 0).sum()), int(keep.sum()), int((keep & replay).sum()), int(live.sum())))
    return np.flatnonzero(live), mapped, cost


def detect_chunk(model, video, matrices, first, n, pose=False):
    out = {k: [] for k in ("frame", "cls", "conf", "xyxy", "track", "foot", "lab", "hist", "kpt")}
    for i, image in enumerate(decode(video, first, n)):
        frame = first + i
        if pose:
            res = model.predict(image, imgsz=IMGSZ, conf=CONF, half=True, device="0",
                                verbose=False)[0]
        else:
            res = model.track(image, persist=True, tracker="bytetrack.yaml", classes=[0, 32],
                              imgsz=IMGSZ, conf=CONF, half=True, device="0", verbose=False)[0]
        b = res.boxes
        if not len(b):
            continue
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(np.int8)
        track = b.id.cpu().numpy().astype(np.int32) if b.id is not None else np.full(len(b), -1, np.int32)
        fx, fy = (xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]
        if pose:
            kpt = res.keypoints.data.cpu().numpy()  # (n,17,3) x,y,conf
            ank = kpt[:, 15:17]
            seen = ank[:, :, 2] >= 0.5
            both = seen.sum(axis=1)
            ax = np.where(both > 0, (ank[:, :, 0] * seen).sum(1) / np.maximum(both, 1), fx)
            ay = np.where(both > 0, (ank[:, :, 1] * seen).sum(1) / np.maximum(both, 1), fy)
            fx, fy = ax, ay
            out["kpt"].append(kpt.astype(np.float16))
        px, py = unproject(matrices[frame], fx, fy)
        out["frame"].append(np.full(len(b), frame, np.int32))
        out["cls"].append(cls)
        out["conf"].append(b.conf.cpu().numpy().astype(np.float16))
        out["xyxy"].append(xyxy.astype(np.float32))
        out["track"].append(track)
        out["foot"].append(np.c_[px, py].astype(np.float32))
        if not pose:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            hist = np.zeros((len(b), BINS ** 3 * len(REGIONS)), np.float16)
            med = np.full((len(b), 3), np.nan, np.float16)
            for k in np.flatnonzero(cls == 0):
                x1, y1, x2, y2 = xyxy[k]
                h, w = y2 - y1, x2 - x1
                xs, xe = int(max(0, x1 + 0.2 * w)), int(min(W, x2 - 0.2 * w))
                parts = []
                for top, bottom in REGIONS.values():
                    ys, ye = int(y1 + top * h), int(y1 + bottom * h)
                    parts.append(histogram(lab[ys:ye, xs:xe].reshape(-1, 3),
                                           hsv[ys:ye, xs:xe].reshape(-1, 3)))
                hist[k] = np.concatenate(parts)
                med[k] = shirt_colour(hsv, lab, xyxy[k])
            out["hist"].append(hist)
            out["lab"].append(med)
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["tactical", "broadcast", "pose"])
    parser.add_argument("--limit", type=int, help="only the first N frames (speed test)")
    args = parser.parse_args()
    from ultralytics import YOLO

    os.makedirs(os.path.join(OUT, args.step), exist_ok=True)
    source = "broadcast" if args.step == "broadcast" else "tactical"
    video = asset_paths(source=source)["video"]
    matrices, _, _ = load_homography(source=source)

    extra = {}
    if args.step == "broadcast":
        frames, mapped, cost = broadcast_plan()
        extra = {"mapped": mapped, "replay_cost": cost}
    else:
        frames = np.concatenate([np.arange(a, z + 1) for _, a, z in half_video_blocks(GAME)])
    if args.limit:
        frames = frames[:args.limit]

    weights = "yolo26x-pose.pt" if args.step == "pose" else "yolo26x.pt"
    total, done, t0 = len(frames), 0, time.time()
    model = YOLO(os.path.join(MODEL_DIR, weights))
    for first, n in runs(frames):
        for tracker in getattr(getattr(model, "predictor", None), "trackers", None) or []:
            tracker.reset()  # fresh ByteTrack ids for each contiguous run
        for a in range(first, first + n, CHUNK):
            m = min(CHUNK, first + n - a)
            path = os.path.join(OUT, args.step, "chunk_{}_{}.npz".format(a, a + m - 1))
            if not os.path.exists(path):
                res = detect_chunk(model, video, matrices, a, m, pose=args.step == "pose")
                if extra:
                    res["mapped_tactical"] = extra["mapped"][res["frame"]] if len(res["frame"]) else np.zeros(0)
                    res["replay_cost"] = extra["replay_cost"][res["frame"]] if len(res["frame"]) else np.zeros(0)
                np.savez_compressed(path + ".tmp.npz", **res)
                os.replace(path + ".tmp.npz", path)
            done += m
            rate = done / (time.time() - t0)
            log(args.step, "{}/{} frames  {:.1f} fps  ETA {:.0f} min".format(
                done, total, rate, (total - done) / max(rate, 1e-6) / 60))


if __name__ == "__main__":
    main()
