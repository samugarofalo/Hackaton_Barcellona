"""Our ball per frame: detections kept by ball_link.link (trajectory filter tuned on SoccerNet), no
Metrica gate. Where two kept detections share a frame, the one nearer the Metrica ball wins, else the
more confident. Values JUMP/MIN_LEN chosen on SoccerNet-Tracking test (ball_link.py output).

    .venv/bin/python pipeline/ball_track.py data/cleanup/ball_det_11185_90651.npz [...]
Writes data/cleanup/ball_track_<start>_<last>.npz (frame-indexed xy, NaN where no ball) and prints
coverage and agreement with Metrica.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE]
from ball_link import link, select, MIN_CONF  # noqa: E402

JUMP, MIN_LEN = 4, 8
SCALE = np.array([105.3, 68.0])
OFF_PITCH_M = 1.0  # spare balls lie beyond the lines; a ball out there is dead anyway


def track(path, metrica):
    z = np.load(path)
    a, last = map(int, os.path.basename(path)[9:-4].split("_"))
    f, c, xy, bb = z["frame"], z["conf"], z["xy"], z["xyxy"]
    px = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
    size = np.median((bb[:, 2] - bb[:, 0] + bb[:, 3] - bb[:, 1]) / 2)
    on = np.all((xy * SCALE > -OFF_PITCH_M) & (xy * SCALE < SCALE + OFF_PITCH_M), axis=1)
    f, c, xy, px = f[on], c[on], xy[on], px[on]
    keep, tid = link(f, px, c, size, JUMP, MIN_LEN)
    keep &= c >= MIN_CONF
    n = last - a + 1
    out = np.full((n, 2), np.nan)
    d = np.linalg.norm((xy - metrica[f]) * SCALE, axis=1)
    prefer = np.where(np.isfinite(d), d, 1e6 - c)  # nearer Metrica first, else confidence
    for i in select(f, px, c, tid, keep, size, JUMP, prefer):
        out[f[i] - a] = xy[i]
    return a, last, out, size


def main():
    metrica = np.load(os.path.join(ROOT, "data", "diagnostics", "broadcast", "atd_tactical.npz"))["ball"]
    for path in sys.argv[1:]:
        a, last, out, size = track(path, metrica)
        np.savez_compressed(os.path.join(ROOT, "data", "cleanup", "ball_track_{}_{}.npz".format(a, last)),
                            start=a, xy=out)
        m = metrica[a:last + 1]
        play = np.zeros(len(m), bool)  # graded play only: from the kickoff (games.py) on
        kick = {11185: 11185, 110808: 113741}.get(a, a)
        play[kick - a:] = True
        has_m, has_o = np.isfinite(m[:, 0]) & play, np.isfinite(out[:, 0])
        d = np.linalg.norm((out - m) * SCALE, axis=1)
        print("{}..{} (ball {:.1f} px): our ball in {:.1%} of frames; where Metrica has it {:.1%} "
              "(within 2 m {:.1%}, 2-4 m {:.1%}, > 4 m {:.1%}); where Metrica has none {:.1%}".format(
                  a, last, size, has_o.mean(), has_o[has_m].mean(), (d[has_m] < 2).mean(),
                  ((d[has_m] >= 2) & (d[has_m] < 4)).mean(), (d[has_m] >= 4).mean(), has_o[~has_m & play].mean()))


if __name__ == "__main__":
    main()
