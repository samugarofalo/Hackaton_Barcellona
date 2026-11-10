"""Diagnostic: where the supplied ball is present but more than 10 m from every tracked player.

1. scan   -- ATD only, graded frames: episodes of "ball observed, nearest tracked player > 10 m",
             classified by ball speed computed from consecutive positions (the ATD speed field is
             empty on ~92% of ball frames).
2. detect -- runs compare_tracking.py around the suspicious episodes (ball still/slow, jumps).
3. report -- per episode: is the ball in the image, and is there a detector box near it that the
             Metrica tracking does not have (a missing player), any box at all, or nothing?

    python pipeline/ball_check.py              # all three steps
    python pipeline/ball_check.py --report     # step 3 only, on what is already detected

Outputs in data/diagnostics/ball-check/: ball_observed.csv, far_episodes.csv, episodes_report.csv.
Videos per clip are compare_tracking's, in data/diagnostics/tracking-comparison/.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402
from compare_tracking import OUT_DIR as CMP_DIR, W, H  # noqa: E402

OUT = os.path.join(ROOT, "data", "diagnostics", "ball-check")
FAR_M = 10.0
JUMP_MS = 50.0  # no kicked ball moves 2 m in 1/25 s
PAD, MAX_FRAMES = 50, 500
LENGTH, WIDTH = GAMES[GAME]["pitch"]


def scan():
    raw = require(["atd_raw"])[0]
    blocks = half_video_blocks(GAME)
    rows = []
    with open(raw) as handle:
        for line in handle:
            frame_id, players, ball = line.rstrip("\n").split(":")
            v = int(frame_id) - 1
            if ball[0] == "N" or not any(a <= v <= z for _, a, z in blocks):
                continue
            bx, by = [float(t) for t in ball.split(",")[:2]]
            p = np.array([[float(t) for t in g.split(",")[:2]]
                          for g in players.split(";") if g[0] != "N"])
            dmin = (np.hypot((p[:, 0] - bx) * LENGTH, (p[:, 1] - by) * WIDTH).min()
                    if len(p) else np.nan)
            rows.append((v, bx * LENGTH, by * WIDTH, dmin))
    df = pd.DataFrame(rows, columns=["frame", "bx", "by", "dmin"])
    contiguous = df.frame.diff() == 1
    df["v"] = np.where(contiguous, np.hypot(df.bx.diff(), df.by.diff()) * 25, np.nan)
    df.to_csv(os.path.join(OUT, "ball_observed.csv"), index=False)

    far = df[df.dmin > FAR_M].frame.values
    cut = np.flatnonzero(np.diff(far) != 1)
    episodes = []
    for a, z in zip(np.r_[far[0], far[cut + 1]], np.r_[far[cut], far[-1]]):
        s = df[(df.frame >= a) & (df.frame <= z)]
        out = ((s.bx < 0) | (s.bx > LENGTH) | (s.by < 0) | (s.by > WIDTH)).mean()
        episodes.append((a, z, z - a + 1, s.v.median(), s.v.max(), s.dmin.max(), out))
    ep = pd.DataFrame(episodes, columns=["start", "end", "n", "v_med", "v_max", "dmax",
                                         "out_share"])

    def category(r):
        if r.v_max > JUMP_MS:
            return "jump"
        if r.out_share > 0.5:
            return "off_pitch"
        if r.v_med >= 10 and r.n <= 75:
            return "fast_short"
        if r.v_med < 3:
            return "still_slow"
        return "intermediate"

    ep["category"] = ep.apply(category, axis=1)
    ep.to_csv(os.path.join(OUT, "far_episodes.csv"), index=False)
    print(ep.groupby("category").n.agg(["size", "sum"]).to_string())
    return ep


def clip_of(r):
    start = int(r.start) - PAD
    return start, int(min(r.end + PAD - start + 1, MAX_FRAMES))


def detect(ep):
    for r in ep[ep.category.isin(["still_slow", "jump"])].itertuples():
        start, n = clip_of(r)
        tag = "{}_{}".format(start, start + n - 1)
        if os.path.exists(os.path.join(CMP_DIR, "detections_{}.csv".format(tag))):
            continue
        print("detect", tag, r.category, flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "compare_tracking.py"),
                        "--start", str(start), "--frames", str(n)],
                       check=True, stdout=subprocess.DEVNULL)


def report(ep):
    ball = pd.read_csv(os.path.join(OUT, "ball_observed.csv")).set_index("frame")
    matrices, _, _ = load_homography()
    rows = []
    for r in ep[ep.category.isin(["still_slow", "jump"])].itertuples():
        start, n = clip_of(r)
        tag = "{}_{}".format(start, start + n - 1)
        det = pd.read_csv(os.path.join(CMP_DIR, "detections_{}.csv".format(tag)))
        det = det[det.on_pitch == 1]
        det["X"], det["Y"] = det.pitch_x * LENGTH, det.pitch_y * WIDTH
        by_frame = dict(list(det.groupby("frame")))
        per = []
        for f in range(int(r.start), int(r.end) + 1):
            b = ball.loc[f]
            px, py = project(matrices[f], b.bx / LENGTH, b.by / WIDTH)
            in_view = bool(0 <= px < W and 0 <= py < H)
            boxes = by_frame.get(f, det.iloc[:0])
            d = np.hypot(boxes.X - b.bx, boxes.Y - b.by)
            unpaired = d[boxes.matched_slot < 0]
            per.append((in_view, d.min() if len(d) else np.inf,
                        unpaired.min() if len(unpaired) else np.inf))
        per = np.array(per, dtype=float)
        in_view = per[:, 0].mean()
        box2 = (per[:, 1] <= 2).mean()
        unpaired2 = (per[:, 2] <= 2).mean()
        # ponytail: majority shares, thresholds fixed for diagnosis only.
        if in_view < 0.5:
            verdict = "ball out of the image"
        elif unpaired2 >= 0.5:
            verdict = "missing player: unpaired box at the ball"
        elif box2 >= 0.5:
            verdict = "box at the ball, paired to a Metrica point >10 m?"
        else:
            verdict = "nobody at the ball (ball wrong, or player undetected)"
        rows.append((int(r.start), int(r.end), int(r.n), r.category, round(r.v_med, 1),
                     round(r.dmax, 1), round(in_view, 2), round(box2, 2), round(unpaired2, 2),
                     round(float(np.median(per[:, 1])), 1), verdict, tag))
    rep = pd.DataFrame(rows, columns=[
        "start", "end", "n", "category", "v_med", "dmax_m", "ball_in_view", "box_within_2m",
        "unpaired_box_within_2m", "median_nearest_box_m", "verdict", "clip"])
    rep.to_csv(os.path.join(OUT, "episodes_report.csv"), index=False)
    print(rep.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", action="store_true", help="only step 3")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.report:
        ep = pd.read_csv(os.path.join(OUT, "far_episodes.csv"))
    else:
        ep = scan()
        detect(ep)
    report(ep)


if __name__ == "__main__":
    main()
