"""Debug video: 2D pitch, four versions in a 2x2 grid (fuse.py). To watch, not to label.

  1 Metrica             supplied ATD; white ball
  2 Metrica + ours      Metrica positions, team flipped where our shirt colour proves it (white ring),
                        extra player at the ball (green ring); ball green = ours, white = Metrica fill
  3 ours only           detector tracks with continuity, grey = colour unsure; our ball
  4 ours + Metrica fill as 3, but an unsure colour takes the team of the Metrica players around it,
                        weighted by distance (white ring = team borrowed from Metrica)
A panel's title turns red when its nearest-to-ball team differs from Metrica's.

    .venv/bin/python pipeline/compare_4way.py --start 20000 --frames 4500
Writes data/cleanup/compare_4way_<start>_<last>.mp4. Prints only aggregate percentages.
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

L, Wd = float(SCALE[0]), float(SCALE[1])
PX, PAD, TOP = 6, 16, 40
PW, PH = int(L * PX), int(Wd * PX)
CW, CH = PW + 2 * PAD, PH + TOP + PAD
SIZE = ((2 * CW) // 2 * 2, (2 * CH + 20) // 2 * 2)  # even: yuv420p
BLUE, RED, GREY, WHITE, GREEN, ALERT = (255, 120, 0), (40, 40, 230), (150, 150, 150), (255, 255, 255), (0, 255, 0), (60, 60, 255)
COL = {"H": BLUE, "A": RED, None: GREY}


def panel(img, i):
    x0, y0 = (i % 2) * CW + PAD, (i // 2) * CH + TOP

    def p(x, y):
        return int(x0 + x * PX), int(y0 + y * PX)
    cv2.rectangle(img, p(0, 0), p(L, Wd), WHITE, 1)
    cv2.line(img, p(L / 2, 0), p(L / 2, Wd), WHITE, 1)
    cv2.circle(img, p(L / 2, Wd / 2), int(9.15 * PX), WHITE, 1)
    for x in (0, L - 16.5):
        cv2.rectangle(img, p(x, Wd / 2 - 20.16), p(x + 16.5, Wd / 2 + 20.16), WHITE, 1)
    return p, (x0, y0 - 12)


def nearest(points, teams, ball):
    if ball is None:
        return None
    pts = [(np.linalg.norm((q - ball) * SCALE), t) for q, t in zip(points, teams) if t]
    return min(pts, key=lambda v: v[0])[1] if pts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=4500)
    a = ap.parse_args()
    start, n = a.start, a.frames
    v = load(start, n)
    mp, mteam, mb = v["metrica"]
    _, fteam, fball, fsrc, extra = v["fused"]
    ours, ob = v["ours"]
    ours_m, _ = v["ours_m"]

    out = os.path.join(ROOT, "data", "cleanup", "compare_4way_{}_{}.mp4".format(start, start + n - 1))
    enc = encoder(out, SIZE)
    names = ["1 Metrica", "2 Metrica + ours", "3 ours only", "4 ours + Metrica fill"]
    stats = {i: [0, 0] for i in (1, 2, 3)}
    for k in range(n):
        img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        img[:] = (40, 90, 40)
        (p1, h1), (p2, h2), (p3, h3), (p4, h4) = (panel(img, i) for i in range(4))
        on = np.flatnonzero(np.isfinite(mp[k, :, 0]))
        ob_k = ob[k] if np.isfinite(ob[k, 0]) else None

        for s in on:
            cv2.circle(img, p1(*mp[k, s] * SCALE), 5, COL[mteam[k, s]], -1)
        b1 = mb[k] if np.isfinite(mb[k, 0]) else None
        if b1 is not None:
            cv2.circle(img, p1(*b1 * SCALE), 4, WHITE, -1)
        t = [nearest(mp[k, on], mteam[k, on], b1)]

        for s in on:
            c = p2(*mp[k, s] * SCALE)
            cv2.circle(img, c, 5, COL[fteam[k, s]], -1)
            if fteam[k, s] != mteam[k, s]:
                cv2.circle(img, c, 9, WHITE, 2)
        ex = extra.get(k, [])
        for q, tm in ex:
            c = p2(*q * SCALE)
            cv2.circle(img, c, 5, COL[tm], -1)
            cv2.circle(img, c, 9, GREEN, 2)
        b2 = fball[k] if np.isfinite(fball[k, 0]) else None
        if b2 is not None:
            cv2.circle(img, p2(*b2 * SCALE), 4, GREEN if fsrc[k] == "ours" else WHITE, -1)
        t.append(nearest(list(mp[k, on]) + [q for q, _ in ex], list(fteam[k, on]) + [tm for _, tm in ex], b2))

        for p, items, ring in ((p3, ours.get(k, []), False), (p4, ours_m.get(k, []), True)):
            for it in items:
                if np.isfinite(it[0][0]):
                    c = p(*it[0] * SCALE)
                    cv2.circle(img, c, 5, COL[it[1]], -1)
                    if ring and it[4] == "metrica":
                        cv2.circle(img, c, 9, WHITE, 2)
            if ob_k is not None:
                cv2.circle(img, p(*ob_k * SCALE), 4, GREEN, -1)
            t.append(nearest([it[0] for it in items], [it[1] for it in items], ob_k))

        for i in (1, 2, 3):
            if t[0] and t[i]:
                stats[i][0] += 1
                stats[i][1] += t[i] != t[0]
        for i, (x, y) in enumerate((h1, h2, h3, h4)):
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
    for i, (both, diff) in stats.items():
        print("{:<22} vs Metrica: nearest-to-ball team differs in {:.1%} of {} frames".format(names[i], diff / max(both, 1), both))
    print("method 4: unsure boxes {:.1%} of ours, of which given a team from Metrica {:.1%}; still unsure {:.1%}".format(
        (v["borrowed"] + v["unsure_m"]) / v["n_ours"], v["borrowed"] / max(v["borrowed"] + v["unsure_m"], 1),
        v["unsure_m"] / v["n_ours"]))


if __name__ == "__main__":
    main()
