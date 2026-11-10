"""Keep ball detections that form a trajectory; drop isolated ones (heads, boots, spots).

Detections (conf >= MIN_CONF) are linked frame to frame, greedily by confidence: a detection joins
the tracklet whose last position is within JUMP ball-sizes per elapsed frame (up to GAP frames
missing), else starts a new one. Tracklets with fewer than MIN_LEN detections are dropped. Units are
ball sizes (median box side), so the same values hold for the tactical video at 1920 and for
SoccerNet frames seen by the model at 640: the ball is 6-7 px in both.

Tuned on SoccerNet-Tracking test clips (other games, ball annotated on every frame):
    .venv/bin/python pipeline/ball_link.py
Writes data/external/ball_yolo/sn_test_dets.npz (cached detections) and prints the grid.
"""

import configparser
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "data", "models", "runs", "external", "ball_yolo", "runs", "s640", "weights", "best.pt")
MIN_CONF = 0.2
GAP = 3
RESET = 25  # frames without a ball after which the next one may appear anywhere


def link(frame, xy, conf, size, jump, min_len):
    """(keep mask, tracklet id) over detections. frame int, xy (n,2) pixels, size = ball px."""
    order = np.lexsort((-conf, frame))
    tid = np.full(len(frame), -1)
    last = {}  # tracklet -> (frame, xy)
    used = set()
    nxt = 0
    for i in order:
        if conf[i] < MIN_CONF:
            continue
        best, best_d = None, np.inf
        for t, (f0, p0) in last.items():
            gap = frame[i] - f0
            if 0 < gap <= GAP + 1 and (t, frame[i]) not in used:
                d = np.linalg.norm(xy[i] - p0)
                if d <= jump * size * gap and d < best_d:
                    best, best_d = t, d
        if best is None:
            best, nxt = nxt, nxt + 1
        tid[i] = best
        used.add((best, frame[i]))
        last[best] = (frame[i], xy[i])
        last = {t: v for t, v in last.items() if frame[i] - v[0] <= GAP + 1}
    ok = tid >= 0
    counts = np.bincount(tid[ok], minlength=max(nxt, 1))
    keep = np.zeros(len(frame), bool)
    keep[ok] = counts[tid[ok]] >= min_len
    return keep, tid


def select(frame, xy, conf, tid, keep, size, jump, prefer=None):
    """One ball per frame, no physical jumps: stay on the current tracklet while it lasts; when it
    ends, take only a kept detection reachable from the last ball (jump sizes per elapsed frame);
    after RESET frames without a ball, restart on the best candidate (`prefer`: lower is better,
    e.g. distance to the Metrica ball; default: higher confidence). Returns detection indices."""
    prefer = -conf if prefer is None else prefer
    picked, cur, last_f, last_p = [], -1, None, None
    idx = np.flatnonzero(keep)
    for fr in np.unique(frame[idx]):
        j = idx[frame[idx] == fr]
        same = j[tid[j] == cur]
        if len(same):
            i = same[0]
        elif last_f is not None and fr - last_f <= RESET:
            d = np.linalg.norm(xy[j] - last_p, axis=1)
            ok = d <= jump * size * (fr - last_f)
            if not ok.any():
                continue
            i = j[ok][np.argmin(d[ok])]
        else:
            i = j[np.argmin(prefer[j])]
        picked.append(i)
        cur, last_f, last_p = tid[i], fr, xy[i]
    return np.array(picked, int)


def sn_detections():
    cache = os.path.join(ROOT, "data", "external", "ball_yolo", "sn_test_dets.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    out = {k: [] for k in ("clip", "frame", "conf", "xy", "size", "gt_xy", "gt_size", "gt_frame", "gt_clip")}
    clips = sorted(glob.glob(os.path.join(ROOT, "data", "external", "soccernet-tracking", "test", "SNMOT-*")))
    for ci, clip in enumerate(clips):
        info = configparser.ConfigParser()
        info.read(os.path.join(clip, "gameinfo.ini"))
        ball = {int(k.split("_")[1]) for k, v in info["Sequence"].items()
                if k.startswith("trackletid_") and v.strip().startswith("ball")}
        for row in open(os.path.join(clip, "gt", "gt.txt")):
            f, t, x, y, w, h = map(float, row.split(",")[:6])
            if int(t) in ball:
                out["gt_clip"].append(ci); out["gt_frame"].append(int(f))
                out["gt_xy"].append((x + w / 2, y + h / 2)); out["gt_size"].append((w + h) / 2)
        images = sorted(glob.glob(os.path.join(clip, "img1", "*.jpg")))
        for image in images:
            res = model.predict(image, imgsz=640, conf=0.05, device="0", verbose=False)[0]
            f = int(os.path.basename(image)[:-4])
            b = res.boxes.xyxy.cpu().numpy()
            for k in range(len(b)):
                out["clip"].append(ci); out["frame"].append(f); out["conf"].append(float(res.boxes.conf[k]))
                out["xy"].append(((b[k, 0] + b[k, 2]) / 2, (b[k, 1] + b[k, 3]) / 2))
                out["size"].append((b[k, 2] - b[k, 0] + b[k, 3] - b[k, 1]) / 2)
        print(os.path.basename(clip), flush=True)
    out = {k: np.array(v) for k, v in out.items()}
    np.savez_compressed(cache, **out)
    return out


def score(d, keep_fn, jump=None):
    """Per frame one detection: the highest-confidence kept one, or select() when jump is given.
    Hit = within 2 ball sizes of the GT ball."""
    hits = picks = gts = 0
    for ci in np.unique(d["gt_clip"]):
        m = d["clip"] == ci
        f, xy, c = d["frame"][m], d["xy"][m], d["conf"][m]
        size = np.median(d["size"][m]) if m.any() else 1
        keep, tid = keep_fn(f, xy, c, size)
        keep &= c >= MIN_CONF
        g = d["gt_clip"] == ci
        gt = {int(fr): (p, s) for fr, p, s in zip(d["gt_frame"][g], d["gt_xy"][g], d["gt_size"][g])}
        gts += len(gt)
        if jump is None:
            chosen = [np.flatnonzero(keep & (f == fr))[np.argmax(c[keep & (f == fr)])] for fr in np.unique(f[keep])]
        else:
            chosen = select(f, xy, c, tid, keep, size, jump)
        for j in chosen:
            fr = f[j]
            picks += 1
            if fr in gt and np.linalg.norm(xy[j] - gt[fr][0]) <= 2 * max(gt[fr][1], 4):
                hits += 1
    return hits / max(picks, 1), hits / max(gts, 1), picks


def main():
    d = sn_detections()
    print("SoccerNet-Tracking test: {} clips, {} frames with a GT ball".format(
        len(np.unique(d["gt_clip"])), len(d["gt_frame"])))
    p, r, n = score(d, lambda f, xy, c, s: (np.ones(len(f), bool), np.arange(len(f))))
    print("no linking (top conf per frame):  precision {:.3f}  recall {:.3f}".format(p, r))
    for jump in (2, 4, 8):
        for min_len in (4, 8, 12, 25):
            p, r, n = score(d, lambda f, xy, c, s: link(f, xy, c, s, jump, min_len))
            p2, r2, n = score(d, lambda f, xy, c, s: link(f, xy, c, s, jump, min_len), jump)
            print("jump {:>2} sizes/frame, min {:>2} det:  precision {:.3f}  recall {:.3f} | no jumps: "
                  "precision {:.3f}  recall {:.3f}".format(jump, min_len, p, r, p2, r2))


if __name__ == "__main__":
    main()


def guided_fill(frame, xy, conf, picked, size, radius, min_conf, max_gap=50):
    """Detections below MIN_CONF that fill a gap of the selected ball: inside a gap of at most
    max_gap frames between two selected detections, take the strongest detection with
    conf >= min_conf within `radius` ball sizes of the straight line between the gap's ends.
    Returns extra detection indices (one per filled frame)."""
    picked = np.asarray(picked, int)
    if len(picked) < 2:
        return np.zeros(0, int)
    order = np.argsort(frame[picked])
    pf, pxy = frame[picked][order], xy[picked][order]
    by_frame = {}
    for i in np.flatnonzero(conf >= min_conf):
        by_frame.setdefault(int(frame[i]), []).append(i)
    extra = []
    for k in range(len(pf) - 1):
        a, b = pf[k], pf[k + 1]
        if b - a < 2 or b - a > max_gap + 1:
            continue
        for fr in range(a + 1, b):
            cand = by_frame.get(fr)
            if not cand:
                continue
            pred = pxy[k] + (pxy[k + 1] - pxy[k]) * (fr - a) / (b - a)
            cand = np.array(cand)
            d = np.linalg.norm(xy[cand] - pred, axis=1)
            ok = d <= radius * size
            if ok.any():
                extra.append(cand[ok][np.argmax(conf[cand[ok]])])
    return np.array(extra, int)
