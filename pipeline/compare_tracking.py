"""Diagnostic: supplied Metrica tracking vs a pretrained person detector, on a short clip.

Not a submission. It draws, on the tactical video itself, the supplied player points (team
colours) and the ball, plus the boxes of an off-the-shelf YOLO person detector linked over time by
ByteTrack, so a human can see where the two disagree. Each box also gets a shirt-colour group
(k-means on the torso colour, grass removed); a group becomes a team when most of its boxes sit on
Metrica points of that team, otherwise "other" (referees, staff). Everything stays local.

    python pipeline/compare_tracking.py                     # 20 s from the first kickoff
    python pipeline/compare_tracking.py --start 11185 --frames 125

Outputs in data/diagnostics/tracking-comparison/ (data/ is git-ignored):
    comparison_<tag>.mp4   boxes white (Metrica point within 2 m) / magenta (none)
    groups_<tag>.mp4       boxes coloured by shirt-colour group
    teams_<tag>.mp4        boxes coloured by team decided per track (H / A / other)
    detections_<tag>.csv   one row per detector box
    metrica_<tag>.csv      one row per supplied player point
    summary_<tag>.json     model, settings, counts, checks

Frame indices are zero-based video frames throughout; ATD frame_id = video_frame + 1.
"""

import argparse
import csv
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
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from load_homography import load as load_homography, project, unproject  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "diagnostics", "tracking-comparison")
MODEL_DIR = os.path.join(ROOT, "data", "models")
W, H = 1920, 1080
# ponytail: fixed diagnostic radius for "a detection supports this point", not tuned on any labels.
MATCH_M = 2.0
# Feet outside the pitch by more than this (metres) are bench, staff or stands.
PITCH_MARGIN_M = 1.0
# Shirt-colour groups: kits split further by sun/shade, plus goalkeepers and officials. With 5 the
# Barcelona keeper and the referee shared a group; 7 separates them on the kickoff clip.
# ponytail: fixed k picked by inspecting one clip; choose k per clip (e.g. silhouette) for full match.
N_GROUPS = 7

HOME_BGR, AWAY_BGR, BALL_BGR = (60, 60, 230), (230, 120, 30), (0, 230, 255)
DET_BGR, ORPHAN_BGR, OTHER_BGR = (255, 255, 255), (255, 0, 255), (0, 255, 255)
GROUP_BGR = [(0, 255, 0), (255, 0, 255), (0, 165, 255), (255, 255, 0), (255, 255, 255),
             (128, 0, 255), (0, 128, 128)]
ROLE_BGR = {"H": HOME_BGR, "A": AWAY_BGR, "other": OTHER_BGR}
COUNTS = ["metrica_points", "boxes_on_pitch", "unpaired_boxes", "visible_unconfirmed_points",
          "out_of_view_points"]


def read_atd(raw_path, start, n):
    """players (n, 46, 2) and ball (n, 2), normalised, NaN where missing, for video start..start+n-1."""
    players, ball, ids = [], [], []
    with open(raw_path) as handle:
        for line in handle:
            frame_id, rest = line.split(":", 1)
            video = int(frame_id) - 1
            if video < start:
                continue
            if video >= start + n:
                break
            groups, ball_group = rest.split(":")
            players.append([[float(v) for v in g.split(",")[:2]] for g in groups.split(";")])
            ball.append([float(v) for v in ball_group.split(",")[:2]])
            ids.append(video)
    if ids != list(range(start, start + n)):
        raise ValueError("ATD rows are not exactly video frames {}..{}".format(start, start + n - 1))
    return np.array(players), np.array(ball)


def decode(video_path, start, n):
    """Exact frames start..start+n-1. Input seek to start/25 s: the file is constant 25 fps with
    start_time 0, and on frames 11185 and 41716 this was checked byte-identical to decoding from
    frame 0 with select on n (and different from frames start-1 and start+1)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", "{:.3f}".format(start / 25), "-i", video_path, "-an",
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = W * H * 3
    for _ in range(n):
        buf = proc.stdout.read(size)
        if len(buf) != size:
            raise IOError("video ended early")
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    proc.stdout.close()
    proc.wait()


def encoder(path, size=(W, H)):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "{}x{}".format(*size), "-r", "25", "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def nearest(a, b):
    """For each row of a, (distance, index) of the nearest row of b; inf/-1 if b is empty."""
    if len(a) == 0:
        return np.zeros(0), np.zeros(0, int)
    if len(b) == 0:
        return np.full(len(a), np.inf), np.full(len(a), -1)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return d.min(axis=1), d.argmin(axis=1)


def one_to_one(a, b, limit):
    """Optimal one-to-one pairing of rows of a and b within `limit`; -1 where left unpaired."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), -1), np.full(len(b), -1)
    import lap
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    _, a_to_b, b_to_a = lap.lapjv(cost, extend_cost=True, cost_limit=limit)
    return a_to_b, b_to_a


def shirt_colour(image_hsv, image_lab, box):
    """Median Lab colour of the torso (15-50% of box height, middle half of width), grass removed."""
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    ys, ye = int(y1 + 0.15 * h), int(y1 + 0.5 * h)
    xs, xe = int(max(0, x1 + 0.25 * w)), int(min(W, x2 - 0.25 * w))
    hsv, lab = image_hsv[ys:ye, xs:xe].reshape(-1, 3), image_lab[ys:ye, xs:xe].reshape(-1, 3)
    # The referee's fluorescent yellow sits at hue 33-37 like grass (38-41) but far brighter
    # (V ~250 vs ~130), so brightness is what keeps it out of the grass mask.
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 50) & (hsv[:, 2] < 200)
    keep = lab[~grass]
    return np.median(keep, axis=0) if len(keep) >= 10 else np.full(3, np.nan)


def majority(values):
    vals, counts = np.unique(values, return_counts=True)
    return vals[counts.argmax()]


def assign_groups(det):
    """Add columns group (k-means on shirt colour), group_role and team (per track)."""
    pitch = det[(det.on_pitch == 1) & det.L.notna()].index
    features = det.loc[pitch, ["L", "a", "b"]].values.astype(np.float32)
    # Standardised, or lightness (spread ~35) drowns the chroma that tells yellow apart (~5-10).
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    cv2.setRNGSeed(0)
    _, labels, _ = cv2.kmeans(features, N_GROUPS, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    det["group"] = -1
    det.loc[pitch, "group"] = labels.ravel()

    groups = {}
    for g in range(N_GROUPS):
        rows = det[det.group == g]
        matched = rows[rows.matched_slot > 0]
        share = len(matched) / max(len(rows), 1)
        teams = matched.matched_team.value_counts()
        role = teams.index[0] if share >= 0.5 and len(teams) else "other"
        groups[g] = {"role": role, "boxes": int(len(rows)), "matched_share": round(share, 3),
                     "matched_H": int(teams.get("H", 0)), "matched_A": int(teams.get("A", 0)),
                     "lab_median": np.round(rows[["L", "a", "b"]].median().values, 1).tolist()}
    det["group_role"] = det.group.map(lambda g: groups[g]["role"] if g >= 0 else "")

    # One team per track: majority over the track's boxes. Untracked boxes keep their own group.
    det["team"] = det.group_role
    tracked = det[(det.track >= 0) & (det.group >= 0)]
    for track, rows in tracked.groupby("track"):
        det.loc[rows.index, "team"] = majority(rows.group_role.values)
    return groups


def draw_common(image, M, present, mpx, mpy, teams, state, ball_xy, show_unsupported):
    for k, slot in enumerate(present):
        centre = (int(round(mpx[k])), int(round(mpy[k])))
        colour = HOME_BGR if teams[slot] == 0 else AWAY_BGR
        cv2.circle(image, centre, 6, colour, -1)
        cv2.circle(image, centre, 6, (0, 0, 0), 1)
        if show_unsupported and state[k] == "visible_unconfirmed":
            cv2.circle(image, centre, 11, ORPHAN_BGR, 2)
        cv2.putText(image, str(slot + 1), (centre[0] + 7, centre[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    if np.isfinite(ball_xy[0]):
        px, py = project(M, ball_xy[0], ball_xy[1])
        cv2.circle(image, (int(round(px)), int(round(py))), 5, BALL_BGR, -1)
        cv2.circle(image, (int(round(px)), int(round(py))), 5, (0, 0, 0), 1)


def header(image, lines):
    cv2.rectangle(image, (0, 0), (820, 12 + 21 * len(lines)), (0, 0, 0), -1)
    for row, text in enumerate(lines):
        cv2.putText(image, text, (8, 20 + row * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, default=11185, help="first video frame (0-based)")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true", help="FP16 inference (2x faster, same boxes on the kickoff bench)")
    args = parser.parse_args()

    from ultralytics import YOLO, __version__ as ul_version
    import torch

    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    length, width = GAMES[GAME]["pitch"]
    scale = np.array([length, width])
    start, n = args.start, args.frames
    end = start + n - 1
    tag = "{}_{}".format(start, end)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    teams = team_of_each_track(meta_path)
    players, ball = read_atd(raw_path, start, n)
    matrices, reproj, valid = load_homography()
    if not valid[start:end + 1].all():
        raise ValueError("invalid homography inside the clip")

    model = YOLO(os.path.join(MODEL_DIR, args.model))
    settings = dict(classes=[0], imgsz=args.imgsz, conf=args.conf, device=args.device, half=args.half,
                    verbose=False)

    # Pass 1: detect, track, measure. Nothing is drawn yet: groups need every box first.
    det_rows, met_rows, roundtrip, per_frame, frames = [], [], [], [], []
    shared_frames = 0  # frames where two points' nearest box within 2 m is the same box
    t0 = time.time()
    decoded = 0
    for i, image in enumerate(decode(video_path, start, n)):
        decoded += 1
        frame = start + i
        M = matrices[frame]

        present = np.flatnonzero(np.isfinite(players[i, :, 0]))
        met_norm = players[i, present]
        mpx, mpy = project(M, met_norm[:, 0], met_norm[:, 1])
        bx, by = unproject(M, mpx, mpy)
        roundtrip.append(np.abs(np.c_[bx, by] - met_norm).max() * max(length, width)
                         if len(present) else 0.0)

        boxes = model.track(image, persist=True, tracker="bytetrack.yaml", **settings)[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        track = (boxes.id.cpu().numpy().astype(int) if boxes.id is not None
                 else np.full(len(conf), -1))
        fx, fy = (xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]
        dnx, dny = unproject(M, fx, fy)
        det_norm = np.c_[dnx, dny]
        on_pitch = ((det_norm > -PITCH_MARGIN_M / scale)
                    & (det_norm < 1 + PITCH_MARGIN_M / scale)).all(axis=1)

        met_m, det_m = met_norm * scale, det_norm * scale
        pitch_det = np.flatnonzero(on_pitch)
        d_met, j_met = nearest(met_m, det_m[pitch_det])
        close = j_met[d_met <= MATCH_M]
        shared_frames += int(len(close) != len(np.unique(close)))
        d_det, _ = nearest(det_m, met_m)
        # Pairs are one-to-one: a box supports at most one point and vice versa.
        met_to_box, box_to_met = one_to_one(met_m, det_m[pitch_det], MATCH_M)
        partner = np.full(len(conf), -1)
        partner[pitch_det] = box_to_met
        in_view = (mpx >= 0) & (mpx < W) & (mpy >= 0) & (mpy < H)
        state = np.where(met_to_box >= 0, "confirmed",
                         np.where(in_view, "visible_unconfirmed", "out_of_view"))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        for k, slot in enumerate(present):
            met_rows.append([frame, slot + 1, "H" if teams[slot] == 0 else "A",
                             *np.round(met_norm[k], 5), round(mpx[k], 1), round(mpy[k], 1),
                             round(float(d_met[k]), 2), state[k]])
        for k in range(len(conf)):
            slot = int(present[partner[k]]) if partner[k] >= 0 else -1
            match_m = (round(float(np.linalg.norm(det_m[k] - met_m[partner[k]])), 2)
                       if slot >= 0 else "")
            det_rows.append([frame, k, int(track[k]), round(float(conf[k]), 3),
                             *np.round(xyxy[k], 1), *np.round(det_norm[k], 5), int(on_pitch[k]),
                             round(float(d_det[k]), 2), slot + 1 if slot >= 0 else -1,
                             ("H" if teams[slot] == 0 else "A") if slot >= 0 else "", match_m,
                             *np.round(shirt_colour(hsv, lab, xyxy[k]), 1)])

        per_frame.append((len(present), int(on_pitch.sum()), int((on_pitch & (partner < 0)).sum()),
                          int((state == "visible_unconfirmed").sum()),
                          int((state == "out_of_view").sum()), bool(np.isfinite(ball[i, 0]))))
        frames.append((present, mpx, mpy, state))
    detect_seconds = time.time() - t0

    det = pd.DataFrame(det_rows, columns=[
        "frame", "det", "track", "conf", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y",
        "on_pitch", "nearest_metrica_m", "matched_slot", "matched_team", "match_m",
        "L", "a", "b"])
    groups = assign_groups(det)

    # Pass 2: draw the three views from the same decoded frames.
    names = ("comparison", "groups", "teams")
    outs = {name: encoder(os.path.join(OUT_DIR, "{}_{}.mp4".format(name, tag))) for name in names}
    by_frame = {f: rows for f, rows in det[det.on_pitch == 1].groupby("frame")}
    empty = det.iloc[:0]
    role_counts = []
    for i, image in enumerate(decode(video_path, start, n)):
        frame = start + i
        M = matrices[frame]
        present, mpx, mpy, state = frames[i]
        rows = by_frame.get(frame, empty)
        role_counts.append([int((rows.team == r).sum()) for r in ("H", "A", "other")])
        stamp = "video frame {}  (ATD frame_id {})  t+{:.2f}s".format(frame, frame + 1, i / 25)
        legend = "dot = Metrica slot (red H / blue A), yellow dot = Metrica ball"
        for name in names:
            canvas = image.copy()
            for r in rows.itertuples():
                if name == "comparison":
                    colour = ORPHAN_BGR if r.matched_slot < 0 else DET_BGR
                    label = "{:.2f}".format(r.conf)
                elif name == "groups":
                    colour = GROUP_BGR[r.group] if r.group >= 0 else (128, 128, 128)
                    label = "g{} #{}".format(r.group, r.track) if r.group >= 0 else "?"
                else:
                    colour = ROLE_BGR.get(r.team, (128, 128, 128))
                    label = "{} #{}".format(r.team or "?", r.track)
                p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
                cv2.rectangle(canvas, p1, p2, colour, 2 if name != "comparison" else 1)
                cv2.putText(canvas, label, (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            colour, 1, cv2.LINE_AA)
            draw_common(canvas, M, present, mpx, mpy, teams, state, ball[i],
                        show_unsupported=(name == "comparison"))
            if name == "comparison":
                lines = [stamp, legend,
                         "box = YOLO person (white: paired 1-to-1 with a Metrica point <{:.0f} m, magenta: no)"
                         .format(MATCH_M),
                         "magenta ring = Metrica point in view but with no paired box (<{:.0f} m)"
                         .format(MATCH_M),
                         "Metrica {}  boxes {}  unpaired boxes {}  in-view unconfirmed {}  out of view {}"
                         .format(*per_frame[i][:5])]
            elif name == "groups":
                lines = [stamp, legend, "box = shirt-colour group g0..g{} (k-means), #track id"
                         .format(N_GROUPS - 1),
                         "  ".join("g{}={}".format(g, groups[g]["role"]) for g in groups)]
            else:
                lines = [stamp, legend,
                         "box = team by track majority: red H, blue A, yellow other",
                         "boxes  H {}  A {}  other {}".format(*role_counts[-1])]
            header(canvas, lines)
            outs[name].stdin.write(canvas.tobytes())
    for proc in outs.values():
        proc.stdin.close()
        proc.wait()

    det.to_csv(os.path.join(OUT_DIR, "detections_{}.csv".format(tag)), index=False)
    with open(os.path.join(OUT_DIR, "metrica_{}.csv".format(tag)), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "slot", "team", "x", "y", "px", "py", "nearest_box_m", "state"])
        w.writerows(met_rows)

    # Check: on boxes sitting on a Metrica point, does the colour-derived team agree with Metrica?
    matched = det[(det.on_pitch == 1) & (det.matched_slot > 0) & (det.team != "")]
    agree = float((matched.team == matched.matched_team).mean()) if len(matched) else None
    states = pd.Series([row[-1] for row in met_rows]).value_counts().to_dict()
    pf, rc = np.array(per_frame, dtype=float), np.array(role_counts, dtype=float)
    summary = {
        "frames": [start, end], "n_decoded": decoded, "fps": 25,
        "model": args.model, "ultralytics": ul_version, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if args.device != "cpu" else "cpu",
        "predict": {k: v for k, v in settings.items() if k != "verbose"},
        "tracker": "bytetrack.yaml (ultralytics default)",
        "match_radius_m": MATCH_M, "pitch_margin_m": PITCH_MARGIN_M, "pitch_m": [length, width],
        "homography_reprojection_error_max": float(reproj[start:end + 1].max()),
        "roundtrip_max_m": float(max(roundtrip)),
        "ball_observed_frames": int(pf[:, 5].sum()),
        "per_frame_mean": dict(zip(COUNTS, np.round(pf[:, :5].mean(0), 2).tolist())),
        "per_frame_max": dict(zip(COUNTS, pf[:, :5].max(0).astype(int).tolist())),
        "metrica_point_states": states,
        "frames_with_two_points_nearest_same_box": shared_frames,
        "groups": groups,
        "team_boxes_per_frame": {r: {"mean": round(float(rc[:, k].mean()), 2),
                                     "min": int(rc[:, k].min()), "max": int(rc[:, k].max())}
                                 for k, r in enumerate(("H", "A", "other"))},
        "team_agrees_with_metrica_on_matched_boxes": agree,
        "n_tracks": int(det[det.track >= 0].track.nunique()),
        "detect_seconds": round(detect_seconds, 1),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_DIR, "summary_{}.json".format(tag)), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
