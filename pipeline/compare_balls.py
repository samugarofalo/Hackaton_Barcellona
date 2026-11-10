"""Debug video: method 4 (fuse.py "ours_m" players) with three ball versions, plus Metrica. 2x2 grid.

  1 Metrica              supplied ATD; white ball
  4 our ball only        green; nothing where the detector has no ball
  4a gaps interpolated   green = detected, cyan = interpolated (gaps <= fuse.BALL_GAP frames)
  4b Metrica fill        green = detected, white = Metrica's ball where ours is missing
Keepers (keepers.py) in panels 4/4a/4b: square with a yellow frame, in their team's colour.
A panel's title turns red when its nearest-to-ball team differs from Metrica's.

    .venv/bin/python pipeline/compare_balls.py --start 20000 --frames 4500
Writes data/cleanup/compare_balls_<start>_<last>.mp4. Prints only aggregate percentages.
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from compare_tracking import encoder  # noqa: E402
from fuse import load, SCALE  # noqa: E402
from compare_4way import panel, nearest, SIZE, COL, WHITE, GREEN, ALERT  # noqa: E402

CYAN = (255, 255, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=4500)
    a = ap.parse_args()
    start, n = a.start, a.frames
    v = load(start, n)
    mp, mteam, mb = v["metrica"]
    ours_m, ob = v["ours_m"]
    balls = [ob, v["ball_interp"], v["ball_mfill"]]
    names = ["1 Metrica", "4 our ball only", "4a gaps interpolated", "4b Metrica fill"]
    out = os.path.join(ROOT, "data", "cleanup", "compare_balls_{}_{}.mp4".format(start, start + n - 1))
    enc = encoder(out, SIZE)
    stats = {i: [0, 0] for i in (1, 2, 3)}
    for k in range(n):
        img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        img[:] = (40, 90, 40)
        panels = [panel(img, i) for i in range(4)]
        on = np.flatnonzero(np.isfinite(mp[k, :, 0]))
        p1 = panels[0][0]
        for s in on:
            cv2.circle(img, p1(*mp[k, s] * SCALE), 5, COL[mteam[k, s]], -1)
        b1 = mb[k] if np.isfinite(mb[k, 0]) else None
        if b1 is not None:
            cv2.circle(img, p1(*b1 * SCALE), 4, WHITE, -1)
        t = [nearest(mp[k, on], mteam[k, on], b1)]
        items = ours_m.get(k, [])
        detected = np.isfinite(ob[k, 0])
        for (p, _), bl, fill in zip(panels[1:], balls, (None, CYAN, WHITE)):
            for it in items:
                if np.isfinite(it[0][0]):
                    c = p(*it[0] * SCALE)
                    if it[4] == "keeper":
                        cv2.rectangle(img, (c[0] - 6, c[1] - 6), (c[0] + 6, c[1] + 6), COL[it[1]], -1)
                        cv2.rectangle(img, (c[0] - 8, c[1] - 8), (c[0] + 8, c[1] + 8), (0, 255, 255), 2)
                    else:
                        cv2.circle(img, c, 5, COL[it[1]], -1)
                    if it[4] == "metrica":
                        cv2.circle(img, c, 9, WHITE, 2)
            b = bl[k] if np.isfinite(bl[k, 0]) else None
            if b is not None:
                cv2.circle(img, p(*b * SCALE), 4, GREEN if detected else fill, -1)
            t.append(nearest([it[0] for it in items], [it[1] for it in items], b))
        for i in (1, 2, 3):
            if t[0] and t[i]:
                stats[i][0] += 1
                stats[i][1] += t[i] != t[0]
        for i, (_, (x, y)) in enumerate(panels):
            colour = ALERT if i and t[0] and t[i] and t[i] != t[0] else WHITE
            cv2.putText(img, "{}   ball: {}".format(names[i], t[i] or "-"), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        f = start + k
        cv2.putText(img, "{}:{:02d}  frame {}".format(f // 1500, f // 25 % 60, f), (SIZE[0] // 2 - 80, SIZE[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    print(out)
    for i, b in zip((1, 2, 3), balls):
        both, diff = stats[i]
        print("{:<22} ball in {:5.1%} of frames; nearest-to-ball team differs from Metrica in {:.1%} of {} frames".format(
            names[i], np.isfinite(b[:, 0]).mean(), diff / max(both, 1), both))
    print("{:<22} ball in {:5.1%} of frames".format(names[0], np.isfinite(mb[:, 0]).mean()))


if __name__ == "__main__":
    main()
