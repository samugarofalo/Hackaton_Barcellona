"""Does DFL mean by 'possession' what the challenge means? Test only, DFL data only, aggregates only.

The challenge publishes, for its public window (DATA.md): 300 s, 241 s of live play, 17 possessions
of mean 14.2 s. Its rules: a touch without control, a deflection, an aerial duel won without control
do not change possession. Here: how long the DFL possessions last, how many are short, and where the
published window falls among DFL windows of the same length.

    .venv/bin/python pipeline/dfl_definition.py
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), HERE]
from dfl_bench import CONVERTED, load  # noqa: E402

FPS, WIN = 25, 7500
PUBLIC_N, PUBLIC_MEAN_S = 17, 14.2  # DATA.md, "The public ground truth"


def spells(gt, half):
    """(start, stop, label) of the maximal runs, cut at half time."""
    cut = np.flatnonzero((gt[1:] != gt[:-1]) | (half[1:] != half[:-1])) + 1
    return [(int(a), int(z), gt[a]) for a, z in zip(np.r_[0, cut], np.r_[cut, len(gt)])]


def absorb_short(gt, half, seconds):
    """Challenge-like labels: a possession shorter than `seconds` that follows another team's
    possession with no stoppage in between never showed control, so the previous team keeps the
    ball ('contested or loose: possession stays with the last team that had it')."""
    out = gt.copy()
    prev = None  # label now carried by the live stretch
    for a, z, lab in spells(gt, half):
        if lab not in ("H", "A"):
            prev = None
        elif prev and lab != prev and z - a < seconds * FPS:
            out[a:z] = prev
        else:
            prev = lab
    return out


def main():
    dur, sand, wn, wm, first, live_frames = [], {1: 0, 2: 0, 3: 0}, [], [], [], 0
    merged = {t: [0, 0] for t in (1, 2, 3)}  # possessions, live frames after absorb_short
    for path in sorted(glob.glob(os.path.join(CONVERTED, "*.npz"))):
        d, gt = load(path)
        half = d["half"]
        sp = spells(gt, half)
        live = [(a, z) for a, z, lab in sp if lab in ("H", "A")]
        dur += [(z - a) / FPS for a, z in live]
        live_frames += sum(z - a for a, z in live)
        for (a0, z0, l0), (a, z, lab), (a1, z1, l1) in zip(sp, sp[1:], sp[2:]):  # A-B-A, nothing dead between
            if lab in ("H", "A") and l0 == l1 and l0 in ("H", "A") and l0 != lab and half[a0] == half[z1 - 1]:
                for t in sand:
                    sand[t] += (z - a) / FPS < t
        for t in merged:
            m = [(a, z) for a, z, lab in spells(absorb_short(gt, half, t), half) if lab in ("H", "A")]
            merged[t][0] += len(m)
            merged[t][1] += sum(z - a for a, z in m)
        starts, is_live = np.array([a for a, z in live]), np.isin(gt, ("H", "A"))
        for h in np.unique(half):
            idx = np.flatnonzero(half == h)
            k0 = idx[0] + int(np.argmax(is_live[idx]))  # kick-off of the half
            for j, w in enumerate(range(k0, idx[-1] - WIN + 2, WIN)):
                n = int(((starts >= w) & (starts < w + WIN)).sum())
                wn.append(n)
                wm.append(is_live[w:w + WIN].sum() / FPS / max(n, 1))
                if j == 0:
                    first.append((n, wm[-1]))
    dur, wn, wm, first = np.array(dur), np.array(wn), np.array(wm), np.array(first)
    print("DFL as labelled, 7 matches: {} possessions, {:.1f} per minute of live play, mean {:.1f} s, median {:.1f} s".format(
        len(dur), len(dur) / (live_frames / FPS / 60), dur.mean(), np.median(dur)))
    print("challenge, public window:   {} possessions, {:.1f} per minute of live play, mean {:.1f} s".format(
        PUBLIC_N, 60 / PUBLIC_MEAN_S, PUBLIC_MEAN_S))
    for t in (1, 2, 3, 5):
        print("  DFL possessions shorter than {} s: {:.1%}, {:.1%} of live time".format(t, (dur < t).mean(), dur[dur < t].sum() / dur.sum()))
    for t in sand:
        print("  DFL A-B-A with B shorter than {} s: {:.1%} of the possessions".format(t, sand[t] / len(dur)))
    for t, (n, fr) in merged.items():
        print("  DFL with possessions under {} s given to the previous team: {:.1f} per minute, mean {:.1f} s".format(
            t, n / (fr / FPS / 60), fr / FPS / n))
    q = [5, 25, 50, 75, 95]
    print("\n{} five-minute windows on DFL: possessions p5 {:.0f}  p25 {:.0f}  median {:.0f}  p75 {:.0f}  p95 {:.0f}".format(len(wn), *np.percentile(wn, q)))
    print("  mean possession (s): p5 {:.1f}  p25 {:.1f}  median {:.1f}  p75 {:.1f}  p95 {:.1f}".format(*np.percentile(wm, q)))
    print("  windows with <= {} possessions: {:.1%}; with mean >= {} s: {:.1%}".format(
        PUBLIC_N, (wn <= PUBLIC_N).mean(), PUBLIC_MEAN_S, (wm >= PUBLIC_MEAN_S).mean()))
    print("  first five minutes of each half ({}): possessions min {:.0f}  median {:.0f}  max {:.0f}; mean possession min {:.1f}  median {:.1f}  max {:.1f} s".format(
        len(first), first[:, 0].min(), np.median(first[:, 0]), first[:, 0].max(), first[:, 1].min(), np.median(first[:, 1]), first[:, 1].max()))


def windows():
    """The organisers' baseline (nearest player, vote over +/- w frames) on a Metrica-like ball, scored
    against DFL as labelled and made challenge-like. Published for the real match (EVALUATION.md):
    +/-50 gives 0.2433 on the full match and +/-125 gives 0.2242, i.e. 0.92 of it; on the public
    window 0.3256 and 0.4242, i.e. 1.30."""
    import pandas as pd
    import dfl_degrade as dd
    from baseline import smooth
    from dfl_bench import score
    from dfl_control import nearest
    dd.ALIVE_GAP_SHARE = 0.0
    rows = []
    for path in sorted(glob.glob(os.path.join(CONVERTED, "*.npz"))):
        d, gt = load(path)
        team = np.broadcast_to(d["team"], d["players"].shape[:2])
        ball = dd.metrica_like_ball(np.random.default_rng(0), d["ball"][:, :2], d["alive"])
        dead = ~np.isfinite(ball[:, 0])
        ball = pd.DataFrame(ball).rolling(12, center=True, min_periods=1).mean().to_numpy(copy=True)
        ball[dead] = np.nan
        near, _ = nearest(d["players"], team, ball)
        codes = np.where(dead, -1, np.where(near == "H", 0, np.where(near == "A", 1, -1)))
        for w in (25, 50, 75, 125, 200):
            lab = smooth(codes, w).copy()
            lab[dead] = "D"
            for t in (0, 2, 3):
                rows.append((t, w, score(d, absorb_short(gt, d["half"], t) if t else gt, lab)[1]))
    tab = pd.DataFrame(rows, columns=["absorbed_s", "w", "spell_f1"]).groupby(["absorbed_s", "w"]).spell_f1.mean().unstack()
    tab["125 / 50"] = tab[125] / tab[50]
    print("\nbaseline nearest +/- w on DFL (rows: possessions under that many seconds given to the previous team)")
    print(tab.round(3).to_string())


def demo():
    gt = np.array(list("HHHHAHHHDDAHHH" * 1))
    half = np.zeros(len(gt), int)
    out = absorb_short(gt, half, 2 / FPS)  # under 2 frames
    assert "".join(out) == "HHHHHHHHDDAHHH", "".join(out)  # the A after the stoppage starts a stretch: kept


if __name__ == "__main__":
    demo()
    main()
    windows()
