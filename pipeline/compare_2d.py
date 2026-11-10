"""Debug video: 2D pitch, supplied Metrica tracking (left) next to ours (right). To watch, not to label.

  left   Metrica ATD: players coloured by slot team (blue H, red A), white ball
  right  ours: night-run boxes' foot points coloured by the shirt-colour classifier (grey = unsure,
         small yellow = official), green ball from ball_track.py
  each side: the team of the player nearest its own ball (what the baseline votes on)

    .venv/bin/python pipeline/compare_2d.py --start 20000 --frames 4500
Writes data/cleanup/compare_2d_<start>_<last>.mp4. Prints only aggregate counts.
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import GAME, GAMES, require  # noqa: E402
from baseline import team_of_each_track, HOME  # noqa: E402
from compare_tracking import encoder  # noqa: E402
from our_video import players, ball, SURE  # noqa: E402

L, Wd = GAMES[GAME]["pitch"]
PX = 8  # pixels per metre
PAD, TOP = 20, 50
PW, PH = int(L * PX), int(Wd * PX)
SIZE = (2 * PW + 3 * PAD, PH + TOP + PAD)
BLUE, RED, YELLOW, GREY, WHITE, GREEN = (255, 120, 0), (40, 40, 230), (0, 230, 255), (150, 150, 150), (255, 255, 255), (0, 255, 0)


def pitch(img, x0):
    def p(x, y):
        return int(x0 + x * PX), int(TOP + y * PX)
    cv2.rectangle(img, p(0, 0), p(L, Wd), WHITE, 1)
    cv2.line(img, p(L / 2, 0), p(L / 2, Wd), WHITE, 1)
    cv2.circle(img, p(L / 2, Wd / 2), int(9.15 * PX), WHITE, 1)
    for x in (0, L - 16.5):
        cv2.rectangle(img, p(x, Wd / 2 - 20.16), p(x + 16.5, Wd / 2 + 20.16), WHITE, 1)
    for x in (0, L - 5.5):
        cv2.rectangle(img, p(x, Wd / 2 - 9.16), p(x + 5.5, Wd / 2 + 9.16), WHITE, 1)
    return p


def nearest_team(points, teams, b):
    """Team of the point nearest the ball b (metres), or None."""
    if b is None or not len(points):
        return None
    return teams[int(np.argmin(np.linalg.norm(points - b, axis=1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=4500)
    a = ap.parse_args()
    start, n = a.start, a.frames
    scale = np.array([L, Wd])
    atd = np.load(os.path.join(ROOT, "data", "diagnostics", "broadcast", "atd_tactical.npz"))
    mp, mb = atd["players"][start:start + n] * scale, atd["ball"][start:start + n] * scale
    slot_team = np.where(team_of_each_track(require(["atd_meta"])[0]) == HOME, "H", "A")

    # Foot points of our boxes: the same rows as our_video.players, positions from carrier_codes order.
    pf, _, pp = players(start, n)
    import glob
    foot = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "night", "tactical", "*.npz"))):
        c = np.load(path)
        keep = (c["cls"] == 0) & np.all((c["foot"] > -0.02) & (c["foot"] < 1.02), axis=1)
        foot.append(c["foot"][keep])
    foot = np.concatenate(foot)
    fr_all = np.load(os.path.join(ROOT, "data", "cleanup", "carrier_codes.npz"))["det_frame"]
    foot = foot[(fr_all >= start) & (fr_all < start + n)] * scale
    ob = ball(start, n) * scale
    rows = {}
    for i, f in enumerate(pf):
        rows.setdefault(int(f), []).append(i)

    out = os.path.join(ROOT, "data", "cleanup", "compare_2d_{}_{}.mp4".format(start, start + n - 1))
    enc = encoder(out, SIZE)
    agree = disagree = 0
    for k in range(n):
        f = start + k
        img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        img[:] = (40, 90, 40)
        left, right = pitch(img, PAD), pitch(img, 2 * PAD + PW)

        on = np.isfinite(mp[k, :, 0])
        for s in np.flatnonzero(on):
            cv2.circle(img, left(*mp[k, s]), 6, BLUE if slot_team[s] == "H" else RED, -1)
        b_m = mb[k] if np.isfinite(mb[k, 0]) else None
        if b_m is not None:
            cv2.circle(img, left(*b_m), 4, WHITE, -1)
        t_m = nearest_team(mp[k][on], slot_team[on], b_m)

        pts, teams = [], []
        for i in rows.get(f, []):
            cls = int(np.argmax(pp[i]))
            sure = pp[i, cls] >= SURE
            if cls == 2 and sure:
                cv2.circle(img, right(*foot[i]), 3, YELLOW, -1)
                continue
            col = (BLUE, RED)[cls] if sure and cls < 2 else GREY
            cv2.circle(img, right(*foot[i]), 6, col, -1)
            if sure and cls < 2:
                pts.append(foot[i]); teams.append("HA"[cls])
        b_o = ob[k] if np.isfinite(ob[k, 0]) else None
        if b_o is not None:
            cv2.circle(img, right(*b_o), 5, GREEN, -1)
        t_o = nearest_team(np.array(pts).reshape(-1, 2), np.array(teams), b_o)
        if t_m and t_o:
            agree += t_m == t_o
            disagree += t_m != t_o

        cv2.putText(img, "Metrica   nearest to ball: {}".format(t_m or "-"), (PAD, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(img, "Ours   nearest to ball: {}".format(t_o or "-"), (2 * PAD + PW, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(img, "{}:{:02d}  frame {}".format(f // 1500, f // 25 % 60, f), (SIZE[0] // 2 - 90, SIZE[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        if t_m and t_o and t_m != t_o:
            cv2.rectangle(img, (0, 0), (SIZE[0] - 1, SIZE[1] - 1), (0, 0, 255), 4)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    tot = max(agree + disagree, 1)
    print("{}\nframes where both have a ball: {}; nearest-player team differs in {:.1%}".format(
        out, agree + disagree, disagree / tot))


if __name__ == "__main__":
    main()
