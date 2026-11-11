"""Write data/cleanup/ball_runs.npz: how long our ball detector sees / loses the ball, aggregate only.

dfl_method4.py, decoder.py and touch_control.py use it to give the DFL simulation the same ball
gaps as our detector on the provided match. The file holds only two arrays of run lengths:

    seen (n,) int   lengths in frames of consecutive frames where our ball is present
    gaps (m,) int   lengths in frames of consecutive frames where it is missing

counted only where the game is live, approximated as "the Metrica ATD has the ball" (it loses the
ball almost only at dead ball, CONTESTO.md section 3), inside the graded halves from the kickoff on.
A run is cut where that live stretch ends, so no run spans a stoppage. No position, no frame number,
no label goes in the file, so it can be shared (privately) as an aggregate.

Input: the ball_track_<start>_<last>.npz files written by ball_track.py, and the Metrica ball from
data/diagnostics/broadcast/atd_tactical.npz (as ball_track.py reads it).

Run it locally, never through an online AI tool:
    python pipeline/make_ball_runs.py            # all data/cleanup/ball_track_*.npz
    python pipeline/make_ball_runs.py a.npz b.npz
Prints only counts and medians.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLEANUP = os.path.join(ROOT, "data", "cleanup")
METRICA = os.path.join(ROOT, "data", "diagnostics", "broadcast", "atd_tactical.npz")
OUT = os.path.join(CLEANUP, "ball_runs.npz")
KICKOFF = {11185: 11185, 110808: 113741}  # first graded frame -> kickoff frame, as in ball_track.py


def run_lengths(mask):
    """Lengths of the True runs and of the False runs of a boolean array."""
    if len(mask) == 0:
        return np.array([], int), np.array([], int)
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.r_[0, edges]
    lengths = np.diff(np.r_[starts, len(mask)])
    values = mask[starts]
    return lengths[values], lengths[~values]


def runs_in_live_play(ours_present, live):
    """Seen and missing run lengths of `ours_present`, computed inside each live stretch only."""
    seen, gaps = [], []
    live = np.asarray(live, bool)
    edges = np.flatnonzero(np.diff(live.astype(np.int8))) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(live)]):
        if live[a]:
            s, g = run_lengths(np.asarray(ours_present[a:b], bool))
            seen.append(s)
            gaps.append(g)
    cat = lambda xs: np.concatenate(xs).astype(int) if xs else np.array([], int)  # noqa: E731
    return cat(seen), cat(gaps)


def main(paths):
    paths = paths or sorted(glob.glob(os.path.join(CLEANUP, "ball_track_*.npz")))
    if not paths:
        sys.exit("No ball_track_*.npz in {}: run ball_track.py first.".format(CLEANUP))
    if not os.path.isfile(METRICA):
        sys.exit("Metrica ball not found at {} (written by the broadcast diagnostics).".format(METRICA))
    metrica = np.load(METRICA)["ball"]
    seen, gaps = [], []
    for path in paths:
        z = np.load(path)
        a, xy = int(z["start"]), z["xy"]
        live = np.isfinite(metrica[a:a + len(xy), 0])
        live[:max(0, KICKOFF.get(a, a) - a)] = False
        s, g = runs_in_live_play(np.isfinite(xy[:, 0]), live)
        seen.append(s)
        gaps.append(g)
        print("{}: {} seen runs, {} gaps".format(os.path.basename(path), len(s), len(g)))
    seen, gaps = np.concatenate(seen), np.concatenate(gaps)
    np.savez_compressed(OUT, seen=seen, gaps=gaps)
    print("wrote {}: {} seen runs (median {} frames), {} gaps (median {} frames)".format(
        os.path.relpath(OUT, ROOT), len(seen), int(np.median(seen)) if len(seen) else 0,
        len(gaps), int(np.median(gaps)) if len(gaps) else 0))


if __name__ == "__main__":
    main(sys.argv[1:])
