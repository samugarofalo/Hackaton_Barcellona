"""Our player chain measured on an external dataset with ground truth: SoccerTrack v2 (CC BY 4.0).

Only data/external/soccertrack-v2 is read; no file of the competition match is opened. The chain
is the one of night_run.py + carrier_fix.py + keepers.py + fuse.py ("ours", "ours_m"), copied here
(same constants) and run on SoccerTrack panoramas brought to the conditions of our tactical video:

  scale     the panorama (4096x1080) already shows players at about the height of ours (GSR boxes:
            median ~50 px vs our 51): kept at 1:1, no resize.
  crop      three 1920x1080 windows (x0 = 0, 1088, 2176; 832 px overlaps), so the detector sees
            1920x1080 at imgsz 1920 exactly as on our video. Each window is a separate run
            (own ByteTrack, own colour model); boxes whose foot is within EDGE px of a window
            side are "don't care", so a player cut by the border is neither a miss nor a FP.
  bitrate   libx264 High yuv420p 25 fps at 5.0 Mbit/s per window (= ours, same pixels per frame).
            The original panorama is itself H.264 at 9.8 Mbit/s for 2.13x the pixels (~4.6 Mbit/s
            per 1920x1080 equivalent), so "recoded" is a second generation; --raw skips it.

Teacher ("maestro") = GSR pitch positions with slot team (they are the provider's BePro tracking
rounded to 1.05 x 0.68 m, the same numbers as raw/*_tracker_box_data.xml); truth = GSR boxes.

    .venv/bin/python pipeline/ext_players_eval.py gt          # GSR -> eval/gt_<seg>.npz (stream parse)
    .venv/bin/python pipeline/ext_players_eval.py prep        # ffmpeg: windows at 5 Mbit/s
    .venv/bin/python pipeline/ext_players_eval.py detect [--raw]   # yolo26x + ByteTrack + hists
    .venv/bin/python pipeline/ext_players_eval.py eval [--raw] [--variant fix12|fix123]
                                                  # metrics -> eval/metrics[_raw][_variant].json
Output in data/external/soccertrack-v2/eval/.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DS = os.path.join(ROOT, "data", "external", "soccertrack-v2")
OUT = os.path.join(DS, "eval")
MODEL = os.path.join(ROOT, "data", "models", "yolo26x.pt")

MATCHES = ("117093", "128057")  # M2 validation, M7 test: the two with a hand-boxed MOT clip
WIN_X0 = (0, 1088, 2176)
WW, WH = 1920, 1080
EDGE = 20
BITRATE = "5000k"

# chain constants, copied from night_run / team_colour / carrier_fix / fuse / keepers
CONF, IMGSZ = 0.05, 1920
BINS = 4
REGIONS = {"shirt": (0.15, 0.50), "shorts": (0.50, 0.70)}
PAIR_M, OTHER_SHARE, SURE = 1.5, 0.1, 0.8
HOLD, VOTE = 25, 12
SIGMA_M, NEAR_M = 1.0, 3.0
GK_A, GK_B, GK_PRIOR = 0.292, -0.578, 0.092   # DFL depth prior (keepers.py)
TEACH, MIN_DEPTH = 0.99, 0.0
OUTLINE_PX = 3  # on-pitch tolerance in the image (benches stand a few px beyond the far line)
POLY = 6  # pixel -> pitch polynomial degree (held-out 0.53/0.60 m median; 8 gains < 0.1 m)
# fixes measured as variants (eval --variant): 1 contrast colour, 2 careful filling, 3 track stitching
FIX = {"contrast": False, "fill": False, "stitch": False}
VARIANTS = {"base": (), "fix1": ("contrast",), "fix2": ("fill",), "fix3": ("stitch",),
            "fix12": ("contrast", "fill"), "fix123": ("contrast", "fill", "stitch")}
STITCH_GAP = 50       # frames: a fragment may resume up to 2 s after the previous one ended
STITCH_SPEED = 8.0    # m/s, a sprint: how far the player can have gone meanwhile
STITCH_SLACK = 1.0    # m, position noise at the two ends
STITCH_COLOUR = 1.0   # max |(pL - pR)_a - (pL - pR)_b|: two sure opposite teams differ by ~2


# ---------------------------------------------------------------- ground truth
def gt():
    """Stream each 2.7 GB GSR json once; keep the object records (no pitch lines) of every frame."""
    os.makedirs(OUT, exist_ok=True)
    for match in MATCHES:
        for half in ("1st", "2nd"):
            path = os.path.join(OUT, "gsr_{}_{}.npz".format(match, half))
            if os.path.exists(path):
                continue
            rows, buf, inside, t0 = [], [], False, time.time()
            with open(os.path.join(DS, "gsr", match, "{}_{}.json".format(match, half))) as fh:
                for line in fh:
                    if not inside:
                        inside = line.startswith('    "annotations"')
                        continue
                    if line.startswith("    ]"):
                        break  # end of annotations ("categories" follows)
                    if line.startswith("        {"):
                        buf = [line]
                    elif buf:
                        buf.append(line)
                        if len(buf) == 2 and line.rstrip().endswith('P",'):
                            buf = []  # pitch-line record
                        elif line.startswith("        }"):
                            a = json.loads("".join(buf).rstrip().rstrip(","))
                            buf = []
                            if a.get("supercategory") != "object":
                                continue  # camera record
                            b, p, at = a["bbox_image"], a["bbox_pitch"], a["attributes"]
                            rows.append((int(a["image_id"][-6:]), a["track_id"], a["category_id"],
                                         at.get("team") or "", at.get("role") or "", b["x"], b["y"], b["w"],
                                         b["h"], p["x_bottom_middle"], p["y_bottom_middle"]))
            d = pd.DataFrame(rows, columns=["frame", "tid", "cat", "team", "role", "x", "y", "w", "h", "px", "py"])
            np.savez_compressed(path, **{c: (lambda v: v.astype(str) if v.dtype == object else v)(d[c].to_numpy())
                                         for c in d.columns})
            print(match, half, len(d), "objects", d.cat.value_counts().to_dict(),
                  "{:.0f} s".format(time.time() - t0), flush=True)


def load_gsr(match, half):
    z = np.load(os.path.join(OUT, "gsr_{}_{}.npz".format(match, half)), allow_pickle=False)
    return pd.DataFrame({k: z[k] for k in z.files})


# ---------------------------------------------------------------- video
def clip(match):
    return os.path.join(DS, "mot", "clips", match + ".mp4")


def window_video(match, w):
    return os.path.join(OUT, "video", "{}_w{}.mp4".format(match, w))


def prep():
    """One decode per clip, split into the three windows, each re-encoded like our tactical video."""
    os.makedirs(os.path.join(OUT, "video"), exist_ok=True)
    for match in MATCHES:
        if all(os.path.exists(window_video(match, w)) for w in range(len(WIN_X0))):
            continue
        n = len(WIN_X0)
        graph = "[0:v]split={}{};".format(n, "".join("[s%d]" % i for i in range(n))) + ";".join(
            "[s{0}]crop={1}:{2}:{3}:0[o{0}]".format(i, WW, WH, x0) for i, x0 in enumerate(WIN_X0))
        for ps in (1, 2):  # two-pass ABR: the average lands on BITRATE (one pass undershoots by ~20%)
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", clip(match), "-filter_complex", graph]
            for i in range(n):
                log = os.path.join(OUT, "video", "pass_{}_w{}".format(match, i))
                cmd += ["-map", "[o%d]" % i, "-an", "-c:v", "libx264", "-profile:v", "high",
                        "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",  # no duplicated frames
                        "-b:v", BITRATE, "-pass", str(ps), "-passlogfile", log]
                cmd += ["-f", "null", os.devnull] if ps == 1 else [window_video(match, i)]
            subprocess.run(cmd, check=True)
        print(match, "windows written", flush=True)


def frames(match, w, raw):
    """BGR frames of window w: the re-encoded file, or (raw) a crop of the original clip."""
    if raw:
        # passthrough: no duplicated frames at timestamp gaps (rawvideo is CFR by default: 6000 vs 5976)
        cmd = ["ffmpeg", "-v", "error", "-i", clip(match), "-vf", "crop={}:{}:{}:0".format(WW, WH, WIN_X0[w]),
               "-fps_mode", "passthrough"]
    else:
        cmd = ["ffmpeg", "-v", "error", "-i", window_video(match, w)]
    proc = subprocess.Popen(cmd + ["-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size = WW * WH * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) != size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(WH, WW, 3).copy()
    proc.wait()


# ---------------------------------------------------------------- detector (night_run.detect_chunk)
def histogram(lab, hsv):
    """team_colour.histogram, copied."""
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 50) & (hsv[:, 2] < 200)
    keep = lab[~grass]
    if len(keep) < 5:
        return np.zeros(BINS ** 3, np.float32)
    q = (keep.astype(np.int32) * BINS) // 256
    h = np.bincount(q[:, 0] * BINS * BINS + q[:, 1] * BINS + q[:, 2], minlength=BINS ** 3)
    return (h / h.sum()).astype(np.float32)


def det_path(match, w, raw):
    return os.path.join(OUT, "det", "{}_w{}{}.npz".format(match, w, "_raw" if raw else ""))


def detect(raw):
    from ultralytics import YOLO
    os.makedirs(os.path.join(OUT, "det"), exist_ok=True)
    for match in MATCHES:
        for w in range(len(WIN_X0)):
            path = det_path(match, w, raw)
            if os.path.exists(path):
                continue
            model = YOLO(MODEL)  # fresh ByteTrack per window run
            out = {k: [] for k in ("frame", "conf", "xyxy", "track", "hist")}
            t0 = time.time()
            for frame, image in enumerate(frames(match, w, raw)):
                res = model.track(image, persist=True, tracker="bytetrack.yaml", classes=[0, 32],
                                  imgsz=IMGSZ, conf=CONF, half=True, device="0", verbose=False)[0]
                b = res.boxes
                if not len(b):
                    continue
                keep = b.cls.cpu().numpy() == 0  # persons; the ball only shares the tracker call
                if not keep.any():
                    continue
                xyxy = b.xyxy.cpu().numpy()[keep]
                track = b.id.cpu().numpy().astype(np.int32)[keep] if b.id is not None else np.full(keep.sum(), -1, np.int32)
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                hist = np.zeros((len(xyxy), BINS ** 3 * len(REGIONS)), np.float16)
                for k, (x1, y1, x2, y2) in enumerate(xyxy):
                    h, wd = y2 - y1, x2 - x1
                    xs, xe = int(max(0, x1 + 0.2 * wd)), int(min(WW, x2 - 0.2 * wd))
                    parts = []
                    for top, bottom in REGIONS.values():
                        ys, ye = int(y1 + top * h), int(y1 + bottom * h)
                        parts.append(histogram(lab[ys:ye, xs:xe].reshape(-1, 3), hsv[ys:ye, xs:xe].reshape(-1, 3)))
                    hist[k] = np.concatenate(parts)
                out["frame"].append(np.full(len(xyxy), frame, np.int32))
                out["conf"].append(b.conf.cpu().numpy()[keep].astype(np.float16))
                out["xyxy"].append((xyxy + [WIN_X0[w], 0, WIN_X0[w], 0]).astype(np.float32))  # panorama px
                out["track"].append(track)
                out["hist"].append(hist)
                if frame % 1000 == 0:
                    print(match, w, frame, "{:.1f} fps".format((frame + 1) / (time.time() - t0)), flush=True)
            np.savez_compressed(path, **{k: np.concatenate(v) for k, v in out.items()})


# ---------------------------------------------------------------- evaluation helpers
def released_calibration(match):
    """Panorama pixel -> pitch metres with the released calibration (fisheye + homography). It misses
    its own hand keypoints by 1.6-1.8 m median and up to 13 m at the far corners (stitched panorama,
    not one fisheye), so it is used only to link MOT ids to GSR ids."""
    r = os.path.join(DS, "raw", match, match + "_")
    c = np.load(r + "camera_intrinsics.npz")  # only K, D, Knew are read (plain arrays, no pickle)
    K, D, Kn = c["K"], c["D"], c["Knew"]
    H = np.linalg.inv(np.load(r + "homography.npy"))

    def to_pitch(fx, fy):
        u = cv2.fisheye.undistortPoints(np.c_[fx, fy][None].astype(np.float64), K, D, P=Kn)[0]
        return cv2.perspectiveTransform(u[None], H)[0] - [52.5, 34.0]
    return to_pitch


def assign(fa, fb, cost, limit):
    """One-to-one per frame (lap.lapjv with cost_limit, as carrier_fix.pair): index into b per a, or -1."""
    import lap
    out = np.full(len(fa), -1)
    gb = pd.Series(np.arange(len(fb))).groupby(fb).indices
    for f, ia in pd.Series(np.arange(len(fa))).groupby(fa).indices.items():
        ib = gb.get(f)
        if ib is None:
            continue
        _, x, _ = lap.lapjv(np.ascontiguousarray(cost(ia, ib), np.float64), extend_cost=True, cost_limit=limit)
        ok = x >= 0
        out[ia[ok]] = ib[x[ok]]
    return out


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def iou_assign(fa, boxa, fb, boxb):
    return assign(fa, fb, lambda i, j: 1 - iou(boxa[i], boxb[j]), 0.5)


def fit(x, y, mask, classes, rng):
    """carrier_fix.colour_model / keepers.fit: linear softmax on the histograms (torch)."""
    import torch
    idx = np.flatnonzero(mask)
    idx = rng.choice(idx, min(300000, len(idx)), replace=False)
    xt, yt = torch.tensor(x[idx]), torch.tensor(y[idx])
    model = torch.nn.Linear(x.shape[1], classes)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(xt), yt).backward()
        opt.step()
    return lambda z: torch.softmax(model(torch.tensor(z)), 1).detach().numpy()


def pitch_outline(match):
    """Image polygon of the pitch boundary from the hand keypoints (walk: near line, right goal line,
    far line, left goal line). The polynomial extrapolates wildly off the pitch, so the on-pitch filter
    of load_detections is done in the image against this outline."""
    kp = json.load(open(os.path.join(DS, "raw", match, match + "_keypoints.json")))
    pts = {tuple(float(v) for v in k.strip("()").split(",")): v for k, v in kp.items()}
    side = lambda cond, key, rev: sorted([k for k in pts if cond(k)], key=key, reverse=rev)  # noqa: E731
    walk = (side(lambda k: k[1] == 68, lambda k: k[0], False) + side(lambda k: k[0] == 105 and k[1] < 68, lambda k: k[1], True)
            + side(lambda k: k[1] == 0 and k[0] < 105, lambda k: k[0], True) + side(lambda k: k[0] == 0 and 0 < k[1] < 68, lambda k: k[1], False))
    return np.array([pts[k] for k in walk], np.float32)


def poly_terms(fx, fy):
    u, v = (np.asarray(fx, float) - 2048) / 2048, (np.asarray(fy, float) - 540) / 540
    return np.stack([u ** i * v ** j for i in range(POLY + 1) for j in range(POLY + 1 - i)], 1)


def truth(match):
    """MOT hand boxes (22 players + keepers per frame) with team / keeper / pitch point from GSR,
    and the pixel -> pitch mapping used by the chain.

    The MOT clip is the first 5,976 frames of the 1st-half panorama (decoded pixels identical), so
    MOT frame i (0-based) = GSR frame i+1. MOT ids are linked to GSR track ids by the assignment
    with the smallest mean foot distance over the clip (released calibration). Then pixel -> pitch
    is refitted as a degree-POLY polynomial from the hand-box feet to the GSR points of the FIRST
    half of the clip (3 trims of residuals > 3 m), so our feet live in the maestro's frame, as
    Metrica's homography does for its own tracking; position errors are reported on the second half."""
    import lap
    to_pitch = released_calibration(match)
    mot = pd.read_csv(os.path.join(DS, "mot", match + ".txt"), header=None, usecols=range(6),
                      names=["frame", "id", "x", "y", "w", "h"])
    mot[["fx", "fy"]] = to_pitch(mot.x + mot.w / 2, mot.y + mot.h)
    g = load_gsr(match, "1st")
    # side each team defends: mean maestro x over the whole half (keepers.defending_left)
    defend_left = {t[0].upper(): bool(v < 0) for t, v in g.groupby("team").px.mean().items()}
    g = g[g.frame <= mot.frame.max() + 1].assign(frame=lambda d: d.frame - 1)
    m = mot.merge(g[["frame", "tid", "px", "py"]], on="frame")
    m["d"] = np.hypot(m.fx - m.px, m.fy - m.py)
    cost = m.groupby(["id", "tid"]).d.mean().unstack()
    _, x, _ = lap.lapjv(cost.to_numpy(), extend_cost=True)
    link = dict(zip(cost.index, cost.columns[x]))
    mot["tid"] = mot.id.map(link)
    mot = mot.merge(g[["frame", "tid", "cat", "team", "px", "py"]], on=["frame", "tid"], how="left")
    mot["box"] = list(np.c_[mot.x, mot.y, mot.x + mot.w, mot.y + mot.h])
    A, Pm = poly_terms(mot.x + mot.w / 2, mot.y + mot.h), mot[["px", "py"]].to_numpy()
    first = keep = (mot.frame < (mot.frame.max() + 1) // 2).to_numpy()
    for _ in range(3):
        coef = np.linalg.lstsq(A[keep], Pm[keep], rcond=None)[0]
        keep = first & (np.hypot(*(A @ coef - Pm).T) < 3)
    to_pitch = lambda fx, fy: poly_terms(fx, fy) @ coef  # noqa: E731
    mot[["fx", "fy"]] = A @ coef
    link_d = np.hypot(mot.fx - mot.px, mot.fy - mot.py)[~first]
    per_id = pd.Series(link_d.to_numpy()).groupby(mot.id.to_numpy()[~first]).median()
    return mot.reset_index(drop=True), g, link_d, to_pitch, per_id, defend_left


def nearest_other(kq, xyq, tq, kd, xyd, td):
    """Distance (m) from each query point to the nearest detection with the same key (window, frame)
    on ANOTHER track."""
    out = np.full(len(kq), np.inf)
    gd = pd.Series(np.arange(len(kd))).groupby(kd).indices
    for k, iq in pd.Series(np.arange(len(kq))).groupby(kq).indices.items():
        idd = gd.get(k)
        if idd is None:
            continue
        d = np.linalg.norm(xyq[iq][:, None] - xyd[idd][None], axis=2)
        d[(tq[iq][:, None] == td[idd][None]) & (tq[iq][:, None] >= 0)] = np.inf
        out[iq] = d.min(1)
    return out


def stitch(det, skip):
    """Fix 3: join ByteTrack fragments of one player, per window. A track that starts 1..STITCH_GAP
    frames after another ended, within reach of a sprint, with a compatible shirt colour, continues
    it. One-to-one (lapjv) between ends and starts, so chains a -> b -> c follow. `skip`: tracks
    that are not players (officials). Returns the new track id per row."""
    import lap
    d = det[(det.track >= 0) & ~det.track.isin(skip)].sort_values("frame")
    g = d.groupby("track")
    t = pd.DataFrame({"w": g.w.first(), "f0": g.frame.min(), "f1": g.frame.max(),
                      "c": g.pL.mean() - g.pR.mean(),
                      "x0": g.px.apply(lambda v: v.iloc[:5].median()), "y0": g.py.apply(lambda v: v.iloc[:5].median()),
                      "x1": g.px.apply(lambda v: v.iloc[-5:].median()), "y1": g.py.apply(lambda v: v.iloc[-5:].median())})
    nxt = {}
    for _, tw in t.groupby("w"):
        gap = tw.f0.to_numpy()[None] - tw.f1.to_numpy()[:, None]          # end of a (row) -> start of b (col)
        dist = np.hypot(tw.x0.to_numpy()[None] - tw.x1.to_numpy()[:, None], tw.y0.to_numpy()[None] - tw.y1.to_numpy()[:, None])
        reach = STITCH_SLACK + STITCH_SPEED * gap / 25
        ok = (gap >= 1) & (gap <= STITCH_GAP) & (dist <= reach) & (np.abs(tw.c.to_numpy()[None] - tw.c.to_numpy()[:, None]) <= STITCH_COLOUR)
        cost = np.where(ok, dist / reach, 2.0)
        _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=1.0)
        for a, b in enumerate(x):
            if b >= 0 and ok[a, b]:
                nxt[tw.index[a]] = tw.index[b]
    root = {}
    for a in sorted(t.index, key=lambda k: t.f0[k]):  # by start time: a fragment's root is already known
        root.setdefault(a, a)
        if a in nxt:
            root[nxt[a]] = root[a]
    return det.track.map(root).fillna(det.track).astype(int).to_numpy(), len(nxt)


def window_valid(w, fx):
    return (fx >= WIN_X0[w] + EDGE) & (fx <= WIN_X0[w] + WW - EDGE)


def pct(a):
    return round(float(np.mean(a)), 4) if len(a) else None


# ---------------------------------------------------------------- the chain + metrics, one match
def evaluate_match(match, raw, rng):
    mot, g, link_d, to_pitch, per_id, defend_left = truth(match)
    gt_box = np.stack(mot.box.to_numpy())
    R = {"hand_foot_vs_gsr_heldout_m": {"median": round(float(np.median(link_d)), 2),
                                        "p90": round(float(np.percentile(link_d, 90)), 2),
                                        "worst_id_median": round(float(per_id.max()), 2)}}

    # maestro: GSR points per frame and slot (= the provider's tracking), team per slot
    tids = np.sort(g.tid.unique())
    F = int(mot.frame.max()) + 1
    P = np.full((F, len(tids), 2), np.nan)
    col = np.searchsorted(tids, g.tid)
    P[g.frame, col] = g[["px", "py"]].to_numpy()
    slot_team = g.groupby("tid").team.first().reindex(tids).map({"left": "L", "right": "R"}).to_numpy()
    R["defends_left"] = {t: bool(v) for t, v in defend_left.items()}

    # truth flags: a player with another within 1.5 m (feet from the hand boxes)
    near = np.zeros(len(mot), bool)
    for f, idx in mot.groupby("frame").indices.items():
        xy = mot[["fx", "fy"]].to_numpy()[idx]
        d = np.linalg.norm(xy[:, None] - xy[None], axis=2) + np.eye(len(idx)) * 99
        near[idx] = d.min(1) < PAIR_M
    mot["near"] = near

    # detections of the three windows, panorama pixels; foot on the pitch; matched to truth by IoU
    parts = []
    for w in range(len(WIN_X0)):
        z = np.load(det_path(match, w, raw))
        d = pd.DataFrame({"frame": z["frame"], "conf": z["conf"].astype(np.float32),
                          "track": np.where(z["track"] >= 0, z["track"] + 100000 * w, -1)})
        d[["x1", "y1", "x2", "y2"]] = z["xyxy"]
        d["w"] = w
        d["fxp"] = (d.x1 + d.x2) / 2
        d[["px", "py"]] = to_pitch(d.fxp, d.y2)
        d["hist_i"] = np.arange(len(d))
        d["valid"] = window_valid(w, d.fxp)
        gw = mot[window_valid(w, mot.x + mot.w / 2)].index.to_numpy()  # truth inside this window
        gin = window_valid(w, mot.x + mot.w / 2).to_numpy()
        # match against ALL truth boxes (a box cut by the border matches nothing and is then ignored)
        j = iou_assign(d.frame.to_numpy(), d[["x1", "y1", "x2", "y2"]].to_numpy(), mot.frame.to_numpy(), gt_box)
        d["gi"] = j
        d["gt_in"] = (j >= 0) & gin[np.maximum(j, 0)]
        parts.append((d, z["hist"].astype(np.float32), gw))
    outline = pitch_outline(match)
    on = lambda d: (np.array([cv2.pointPolygonTest(outline, (float(x), float(y)), True) for x, y in  # noqa: E731
                              zip(d.fxp, d.y2)]) >= -OUTLINE_PX) & (d.px.abs() <= 52.5 + 2.1) & (d.py.abs() <= 34 + 1.36)
    det = pd.concat([p[0][on(p[0])] for p in parts], ignore_index=True)
    X = np.concatenate([p[1][on(p[0]).to_numpy()] for p in parts])
    gt_in_w = {w: p[2] for w, p in enumerate(parts)}

    # ---- 1. detection, per window inside its don't-care border, summed
    det_rows, tp, fn, fp = [], 0, 0, 0
    rec_near, rec_far = [], []
    for w, gw in gt_in_w.items():
        dw = det[det.w == w]
        hit = np.zeros(len(mot), bool)
        hit[dw.gi[dw.gi >= 0]] = True
        tp += int(hit[gw].sum())
        fn += int((~hit[gw]).sum())
        fp += int(((dw.gi < 0) & dw.valid).sum())
        rec_near += list(hit[gw][mot.near.to_numpy()[gw]])
        rec_far += list(hit[gw][~mot.near.to_numpy()[gw]])
    gk_rows = np.concatenate([gw[mot.cat.to_numpy()[gw] == 2] for gw in gt_in_w.values()])
    hit_all = det[det.gi >= 0].groupby("w").gi.apply(set)
    R["detection"] = {"truth_boxes": tp + fn, "recall": round(tp / (tp + fn), 4),
                      "precision_all_unmatched_as_FP": round(tp / (tp + fp), 4), "FP_boxes": fp,
                      "recall_isolated": pct(rec_far), "recall_within_1.5m": pct(rec_near),
                      "share_truth_within_1.5m": round(len(rec_near) / (len(rec_near) + len(rec_far)), 4),
                      "recall_keepers": pct([r in hit_all.get(w, set()) for w, gw in gt_in_w.items()
                                            for r in gw[mot.cat.to_numpy()[gw] == 2]])}
    tpd = det[det.gt_in]
    iou_m = np.array([iou(np.array([[a, b, c, e]]), gt_box[k][None])[0, 0] for a, b, c, e, k in
                      zip(tpd.x1, tpd.y1, tpd.x2, tpd.y2, tpd.gi)][::20])
    R["detection"]["iou_median"] = round(float(np.median(iou_m)), 3)

    # ---- 2. pairing with the maestro (carrier_fix.pair, 1.5 m)
    # per window: each window is its own "video", so a player seen in two overlapping windows pairs twice
    nw = len(WIN_X0)
    slot = assign((det.w * F + det.frame).to_numpy(), np.repeat(np.arange(nw * F), len(tids)),
                  lambda i, j: np.nan_to_num(np.linalg.norm(det[["px", "py"]].to_numpy()[i][:, None]
                                                            - np.tile(P.reshape(-1, 2), (nw, 1))[j][None], axis=2), nan=99), PAIR_M)
    det["slot"] = np.where(slot >= 0, slot % len(tids), -1)
    true_team = np.where(det.gi >= 0, mot.team.to_numpy()[np.maximum(det.gi, 0)], "")
    det["true"] = pd.Series(true_team).map({"left": "L", "right": "R"}).fillna("").to_numpy()
    det["true_gk"] = (det.gi >= 0) & (mot.cat.to_numpy()[np.maximum(det.gi, 0)] == 2)
    paired = det.slot >= 0
    R["pairing"] = {"paired_share": pct(paired), "paired_and_truth_same_player": pct(
        (mot.tid.to_numpy()[det.gi[paired & (det.gi >= 0)]] == tids[det.slot[paired & (det.gi >= 0)]]))}

    # ---- 3. colour, 3 classes L / R / other (carrier_fix.colour_model)
    tr = det[det.track >= 0].groupby("track").slot.agg(lambda s: (s >= 0).mean()).rename("share").to_frame()
    tr["size"] = det[det.track >= 0].groupby("track").size()
    officials = tr[(tr.share < OTHER_SHARE) & (tr["size"] >= 50)].index
    y = np.full(len(det), -1)
    y[paired] = np.where(slot_team[det.slot[paired]] == "L", 0, 1)
    y[~paired & det.track.isin(officials)] = 2
    late = det.frame.to_numpy() >= F // 2
    held = fit(X, y, (y >= 0) & ~late, 3, rng)(X)
    test = late & (det.true != "").to_numpy()
    ph = held[test]
    truth_lr = (det.true.to_numpy()[test] == "R").astype(int)
    sure = ph[:, :2].max(1) >= SURE
    R["colour_box_heldout"] = {"boxes": int(test.sum()), "acc_argmax_LR": pct(ph[:, :2].argmax(1) == truth_lr),
                               "sure_share": pct(sure), "acc_when_sure": pct((ph[:, :2].argmax(1) == truth_lr)[sure]),
                               "called_other_share": pct(ph.argmax(1) == 2)}
    p3 = fit(X, y, y >= 0, 3, rng)(X)
    det[["pL", "pR", "pO"]] = p3
    t = (det.true != "").to_numpy()
    s3 = p3[t, :2].max(1) >= SURE
    ok3 = p3[t, :2].argmax(1) == (det.true.to_numpy()[t] == "R")
    R["colour_box_insample"] = {"acc_argmax_LR": pct(ok3), "sure_share": pct(s3), "acc_when_sure": pct(ok3[s3]),
                                "acc_when_sure_isolated": pct(ok3[s3 & ~mot.near.to_numpy()[det.gi[t]]]),
                                "acc_when_sure_within_1.5m": pct(ok3[s3 & mot.near.to_numpy()[det.gi[t]]])}

    if FIX["stitch"]:
        skip = det[det.track >= 0].groupby("track").pO.mean()
        n_before = det[det.track >= 0].track.nunique()
        det["track"], joins = stitch(det, skip[skip >= 0.5].index)
        R["stitch"] = {"tracks_before": int(n_before), "joins": int(joins),
                       "tracks_after": int(det[det.track >= 0].track.nunique())}

    # ---- 4. keepers (keepers.py), with two keeper colour classes
    own = np.where((det.px < 0).to_numpy() == defend_left["L"], "L", "R")  # team defending that half
    depth = np.full(len(det), np.nan)
    fr = det.frame.to_numpy()
    for tm in ("L", "R"):
        i = np.flatnonzero(own == tm)
        mates = P[fr[i]][:, slot_team == tm]
        same = np.hypot(mates[..., 0] - det.px.to_numpy()[i, None], mates[..., 1] - det.py.to_numpy()[i, None]) <= 1.5
        mx = np.where(same | np.isnan(mates[..., 0]), np.nan, mates[..., 0])
        goal = (lambda v: v + 52.5) if defend_left[tm] else (lambda v: 52.5 - v)
        with np.errstate(all="ignore"):
            depth[i] = np.nanmin(goal(mx), axis=1) - goal(det.px.to_numpy()[i])
    pp = np.where(np.isfinite(depth), 1 / (1 + np.exp(-(GK_A * np.clip(np.nan_to_num(depth, nan=-60), -60, 60) + GK_B))), GK_PRIOR)
    y5 = np.full(len(det), -1)
    y5[paired] = np.where(slot_team[det.slot[paired]] == "L", 0, 1)
    teach_gk = paired.to_numpy() & (pp >= TEACH)
    y5[teach_gk] = np.where(slot_team[det.slot[teach_gk]] == "L", 2, 3)
    y5[~paired & det.track.isin(officials)] = 4
    gk_teacher_true = pct(det.true_gk.to_numpy()[teach_gk])
    p5 = fit(X, y5, y5 >= 0, 5, rng)(X)
    eps = 1e-4
    lo = (np.log(p5[:, 2] + p5[:, 3] + eps) - np.log(p5[:, 0] + p5[:, 1] + eps)
          + np.log(pp + eps) - np.log(1 - pp + eps) - np.log(GK_PRIOR / (1 - GK_PRIOR)))
    kd = pd.DataFrame({"trk": det.track, "lo": lo, "f": fr, "w": det.w, "own": own})
    kd["post"] = 1 / (1 + np.exp(-kd.trk.map(kd[kd.trk >= 0].groupby("trk").lo.mean()).fillna(kd.lo)))
    kd.loc[p5[:, 4] >= 0.5, "post"] = 0.0
    keeper = ((kd.post >= 0.5) & (np.nan_to_num(depth, nan=-99) >= MIN_DEPTH)).to_numpy(copy=True)
    cand = kd[keeper].sort_values("post", ascending=False)
    keeper[cand.index[cand.duplicated(["w", "f", "own"])]] = False
    det["keeper"] = keeper
    det["gk_team"] = np.where(keeper, own, "")
    tg = det.true_gk.to_numpy()
    colour_side = np.where(p5[:, 2] >= p5[:, 3], "L", "R")
    R["keepers"] = {"teacher_boxes": int(teach_gk.sum()), "teacher_really_keeper": gk_teacher_true,
                    "flagged_boxes": int(keeper.sum()), "precision": pct(tg[keeper]),
                    "recall_of_detected_keepers": pct(keeper[tg]),
                    "team_right_when_flagged_true": pct((own == det.true.to_numpy())[keeper & tg]),
                    "colour_GK_side_agrees_with_half": pct((colour_side == own)[keeper]),
                    "false_keepers_are_outfield_of_team": pd.Series(det.true.to_numpy()[keeper & ~tg & (det.gi >= 0).to_numpy()]).value_counts().to_dict(),
                    "false_keepers_unmatched_boxes": int((keeper & (det.gi < 0).to_numpy()).sum())}

    # ---- 5. officials: who is dropped
    tro = det[det.track >= 0].groupby("track").agg(pO=("pO", "mean"), n=("pO", "size"),
                                                   matched=("gi", lambda s: (s >= 0).mean()))
    dropped = tro[tro.pO >= 0.5]
    R["officials"] = {"tracks_dropped": len(dropped), "boxes_dropped": int(dropped.n.sum()),
                      "dropped_boxes_that_are_players": int(det[det.track.isin(dropped.index)].gi.ge(0).sum()),
                      "player_boxes_dropped_share": pct(det[det.gi >= 0].track.isin(dropped.index)),
                      "unmatched_tracks_ge50_boxes_matched_lt10pct": int(((tro.matched < 0.1) & (tro.n >= 50)).sum()),
                      "of_which_dropped": int(((tro.matched < 0.1) & (tro.n >= 50) & (tro.pO >= 0.5)).sum())}
    R["officials"]["_dropped_track_ids"] = [int(t) for t in dropped.index]

    # ---- 6. continuity (raw ByteTrack tracks) per window: IDF1 approx, ID switches, fragments
    idtp = n_gt = n_det = sw = frag = 0
    per_id_tracks = []
    for w, gw in gt_in_w.items():
        dw = det[(det.w == w) & (det.track >= 0)]
        n_gt += len(gw)
        n_det += int((dw.gt_in | ((dw.gi < 0) & dw.valid)).sum())
        mm = dw[dw.gt_in].assign(gt_id=lambda v: mot.id.to_numpy()[v.gi])
        c = mm.groupby(["gt_id", "track"]).size().unstack(fill_value=0).to_numpy().astype(np.float64)
        import lap
        _, x, _ = lap.lapjv(c.max() - c, extend_cost=True)  # max-weight id assignment
        ok = (x >= 0) & (x < c.shape[1])
        idtp += int(c[np.flatnonzero(ok), x[ok]].sum())
        seq = mm.sort_values(["gt_id", "frame"])
        for _, s_ in seq.groupby("gt_id"):
            tk, fk = s_.track.to_numpy(), s_.frame.to_numpy()
            # a switch only between matches <= 1 s apart: leaving the window and coming back is not one
            sw += int(((tk[1:] != tk[:-1]) & (np.diff(fk) <= 25)).sum())
            per_id_tracks.append(len(np.unique(tk)))
        # fragments: a truth player's coverage breaks (matched -> unmatched -> matched) inside the window
        hit = np.zeros(len(mot), bool)
        hit[mm.gi] = True
        sub = mot.loc[gw, ["id", "frame"]].assign(h=hit[gw]).sort_values(["id", "frame"])
        for _, s_ in sub.groupby("id"):
            h = s_.h.to_numpy().astype(int)
            h = h[np.argmax(h):]  # from the first time he is found
            frag += int((np.diff(h) == 1).sum())  # each re-acquisition = one break
    tl = det[det.gt_in & (det.track >= 0)].groupby("track").size()
    pur = det[det.gt_in & (det.track >= 0)].assign(gid=lambda v: mot.id.to_numpy()[v.gi])
    major = pur.groupby("track").gid.agg(lambda v: v.value_counts().index[0])
    impure = pur.gid.to_numpy() != pur.track.map(major).to_numpy()
    n_ids = pur.groupby("track").gid.agg(lambda v: (v.value_counts() >= 10).sum())
    R["continuity_raw_tracks"] = {"IDF1_approx": round(2 * idtp / (n_gt + n_det), 4), "IDR": round(idtp / n_gt, 4),
                                  "IDP": round(idtp / n_det, 4), "id_switches": sw,
                                  "player_track_boxes_median_s": round(float(tl.median()) / 25, 1),
                                  "boxes_on_a_track_mostly_another_player": pct(impure),
                                  "tracks_spanning_2plus_players(>=10 boxes each)": int((n_ids >= 2).sum()),
                                  "tracks_on_players": int(len(n_ids)),
                                  "id_switches_per_visible_player_minute": round(sw / (n_gt / 1500), 3),
                                  "coverage_breaks_per_visible_player_minute": round(frag / (n_gt / 1500), 3),
                                  "coverage_breaks": frag, "tracks_per_truth_player_per_window_median": float(np.median(per_id_tracks)),
                                  "tracks_per_truth_player_per_window_mean": round(float(np.mean(per_id_tracks)), 2)}

    # ---- 7. fuse "ours" (continuity + track colour vote) and "ours_m" (maestro fills doubts)
    rows = []
    trm = det[det.track >= 0].groupby("track").pO.mean()
    pool = det[~det.track.isin(dropped.index)]
    pkey, pxy, ptk = (pool.w * F + pool.frame).to_numpy(), pool[["px", "py"]].to_numpy(), pool.track.to_numpy()
    det["near_own"] = nearest_other((det.w * F + det.frame).to_numpy(), det[["px", "py"]].to_numpy(),
                                    det.track.to_numpy(), pkey, pxy, ptk)
    for tk, h in det[det.track >= 0].sort_values("frame").groupby("track"):
        if trm[tk] >= 0.5:
            continue
        h = h.drop_duplicates("frame")
        frames_ = np.arange(h.frame.min(), h.frame.max() + 1)
        full = h.set_index("frame").reindex(frames_)
        gap_len = full.px.isna().astype(int).groupby(full.px.notna().cumsum()).transform("sum")
        fill = full.px.isna() & (gap_len <= HOLD)
        xy = full[["px", "py", "x1", "y1", "x2", "y2"]].interpolate(limit_area="inside")
        pc = full[["pL", "pR"]]
        if FIX["contrast"]:  # fix 1: in a contrast the box holds two shirts; vote with isolated frames only
            pc = pc.mask(full.near_own < PAIR_M, axis=0)
        cc = pc.rolling(2 * VOTE + 1, center=True, min_periods=1).mean()
        if FIX["contrast"]:
            cc = cc.ffill().bfill()  # a contrast longer than the vote window: nearest isolated colour
        gkf = full.gk_team.replace("", np.nan).ffill()
        keep = (full.px.notna() | fill).to_numpy()
        team = np.where(gkf.notna(), gkf.fillna(""), np.where(cc.pL >= SURE, "L", np.where(cc.pR >= SURE, "R", "")))
        rows.append(pd.DataFrame({"frame": frames_[keep], "w": int(h.w.iloc[0]), "track": tk,
                                  "px": xy.px[keep], "py": xy.py[keep], "x1": xy.x1[keep], "y1": xy.y1[keep],
                                  "x2": xy.x2[keep], "y2": xy.y2[keep], "seen": full.px.notna().to_numpy()[keep],
                                  "team": team[keep], "gk": gkf.notna().to_numpy()[keep]}))
    ours = pd.concat(rows, ignore_index=True)
    if FIX["fill"]:  # fix 2: no filled box where another detection stands within 1.5 m (it is likely him)
        f_ = np.flatnonzero(~ours.seen.to_numpy())
        dn = nearest_other((ours.w.to_numpy()[f_] * F + ours.frame.to_numpy()[f_]), ours[["px", "py"]].to_numpy()[f_],
                           ours.track.to_numpy()[f_], pkey, pxy, ptk)
        ours = ours.drop(index=f_[dn < PAIR_M]).reset_index(drop=True)
        R["fill_dropped"] = int((dn < PAIR_M).sum())
    ours["fxp"] = (ours.x1 + ours.x2) / 2
    x0 = np.array(WIN_X0)[ours.w.to_numpy()]
    ours["valid"] = (ours.fxp >= x0 + EDGE) & (ours.fxp <= x0 + WW - EDGE)
    # ours_m: unsure -> maestro players within NEAR_M weighted exp(-d^2/2 sigma^2)
    tm = ours.team.to_numpy().copy()
    un = np.flatnonzero(tm == "")
    q = P[ours.frame.to_numpy()[un]]
    dd = np.linalg.norm(q - ours[["px", "py"]].to_numpy()[un][:, None], axis=2)
    wgt = np.where(dd <= NEAR_M, np.exp(-dd ** 2 / (2 * SIGMA_M ** 2)), 0.0)
    wl = wgt[:, slot_team == "L"].sum(1)
    tot = wgt.sum(1)
    tm[un] = np.where(tot > 0, np.where(wl >= tot - wl, "L", "R"), "")
    ours["team_m"] = tm
    # score both against truth, per window
    res = {k: [] for k in ("hit_team", "hit_team_m", "near", "seen")}
    cover = {"truth": 0, "covered_raw": 0, "covered_ours": 0, "ours_FP": 0, "ours_filled": 0, "ours_filled_FP": 0}
    for w, gw in gt_in_w.items():
        ow = ours[ours.w == w]
        j = iou_assign(ow.frame.to_numpy(), ow[["x1", "y1", "x2", "y2"]].to_numpy(), mot.frame.to_numpy(), gt_box)
        gin = np.zeros(len(mot), bool)
        gin[gw] = True
        inside = (j >= 0) & gin[np.maximum(j, 0)]
        cover["truth"] += len(gw)
        cover["covered_ours"] += int(inside.sum())
        cover["covered_raw"] += int(det[(det.w == w) & det.gt_in & ~det.track.isin(dropped.index)].shape[0])
        cover["ours_FP"] += int(((j < 0) & ow.valid).sum())
        cover["ours_filled"] += int((~ow.seen & ow.valid).sum())
        cover["ours_filled_FP"] += int((~ow.seen & ow.valid & (j < 0)).sum())
        tt = mot.team.map({"left": "L", "right": "R"}).to_numpy()[j[inside]]
        res["hit_team"] += list(np.where(ow.team.to_numpy()[inside] == "", "?", ow.team.to_numpy()[inside] == tt))
        res["hit_team_m"] += list(np.where(ow.team_m.to_numpy()[inside] == "", "?", ow.team_m.to_numpy()[inside] == tt))
        res["near"] += list(mot.near.to_numpy()[j[inside]])
        res["seen"] += list(ow.seen.to_numpy()[inside])

    def team_stats(v, sel):
        v = np.asarray(v)[sel]
        dec = v != "?"
        return {"decided_share": pct(dec), "acc_when_decided": pct(v[dec] == "True"),
                "right_share_of_all": pct(v == "True"), "wrong_share_of_all": pct(v == "False")}
    near_ = np.asarray(res["near"], bool)
    R["ours"] = {"coverage": {k: v for k, v in cover.items()},
                 "recall_raw_not_dropped": round(cover["covered_raw"] / cover["truth"], 4),
                 "recall_ours": round(cover["covered_ours"] / cover["truth"], 4),
                 "team": team_stats(res["hit_team"], np.ones(len(near_), bool)),
                 "team_isolated": team_stats(res["hit_team"], ~near_),
                 "team_within_1.5m": team_stats(res["hit_team"], near_)}
    R["ours_m"] = {"team": team_stats(res["hit_team_m"], np.ones(len(near_), bool)),
                   "team_isolated": team_stats(res["hit_team_m"], ~near_),
                   "team_within_1.5m": team_stats(res["hit_team_m"], near_)}

    # ---- 8. pitch position of our feet
    m = det[det.gt_in & (det.frame >= F // 2)]  # second half of the clip: the mapping was fitted on the first
    k = m.gi.to_numpy()
    e_gsr = np.hypot(m.px - mot.px.to_numpy()[k], m.py - mot.py.to_numpy()[k])
    e_box = np.hypot(m.px - mot.fx.to_numpy()[k], m.py - mot.fy.to_numpy()[k])
    q = lambda e: {"median": round(float(np.median(e)), 2), "p90": round(float(np.percentile(e, 90)), 2)}  # noqa: E731
    R["position_m"] = {"ours_vs_gsr_pitch": q(e_gsr), "ours_vs_hand_box_foot": q(e_box), "hand_box_foot_vs_gsr_pitch": q(link_d),
                       "note": "second half of the clip; pixel->pitch fitted on the first half"}
    return R, det


def contact_sheet(match, raw, det, tracks, path):
    """One crop per dropped track (middle box), so a human can say official / player / other."""
    tiles = []
    for t in tracks[:60]:
        h = det[det.track == t]
        r = h.iloc[len(h) // 2]
        w = int(r.w)
        cap = cv2.VideoCapture(clip(match))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
        ok, img = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = [int(v) for v in (r.x1, r.y1, r.x2, r.y2)]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        crop = img[max(0, cy - 60):cy + 60, max(0, cx - 60):cx + 60]
        crop = cv2.resize(cv2.copyMakeBorder(crop, 0, 120 - crop.shape[0], 0, 120 - crop.shape[1], cv2.BORDER_CONSTANT), (240, 240))
        cv2.rectangle(crop, ((x1 - cx + 60) * 2, (y1 - cy + 60) * 2), ((x2 - cx + 60) * 2, (y2 - cy + 60) * 2), (0, 0, 255), 1)
        cv2.putText(crop, str(t), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(crop)
    if tiles:
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))


def evaluate(raw, variant):
    for k in VARIANTS[variant]:
        FIX[k] = True
    suffix = ("_raw" if raw else "") + ("" if variant == "base" else "_" + variant)
    rng = np.random.default_rng(0)
    allr = {}
    for match in MATCHES:
        r, det = evaluate_match(match, raw, rng)
        contact_sheet(match, raw, det, r["officials"].pop("_dropped_track_ids"),
                      os.path.join(OUT, "officials_{}{}.jpg".format(match, suffix)))
        allr[match] = r
        print(match, json.dumps(r, indent=1), flush=True)
    with open(os.path.join(OUT, "metrics{}.json".format(suffix)), "w") as fh:
        json.dump(allr, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["gt", "prep", "detect", "eval"])
    ap.add_argument("--raw", action="store_true", help="no recompression: crop the original panorama")
    ap.add_argument("--variant", choices=list(VARIANTS), default="base", help="eval: which fixes are on")
    a = ap.parse_args()
    {"gt": gt, "prep": prep, "detect": lambda: detect(a.raw), "eval": lambda: evaluate(a.raw, a.variant)}[a.step]()


if __name__ == "__main__":
    main()
