"""Score possession rules on the 7 DFL matches (other games: tuning here is allowed).

Ground truth per frame from the official DFL data: BallStatus 0 -> D, else BallPossession H/A.
Halves are scored as separate runs (an X frame between them). Scored with the competition's own
evaluation/possession_metrics.py, so the numbers mean the same as on the leaderboard.

Rules compared (each name says what it uses):
  nearest_wN            starter baseline: nearest player's team, majority vote +/- N frames.
                        DFL always tracks the ball, so it never says D.
  nearest_w50_oracleD   same, but D wherever the GT says dead: the ceiling of the team part
                        alone. A diagnostic, never a submission.

    python pipeline/dfl_bench.py
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from baseline import smooth, HOME, AWAY, NOBODY  # noqa: E402

CONVERTED = os.path.join(ROOT, "data", "external", "bassek2025", "converted")
SCALE = np.array([105.0, 68.0], np.float32)


def load(path):
    d = np.load(path)
    gt = np.where(d["alive"] == 0, "D", d["possession"])
    gt = np.where(gt == "", "D", gt)
    return d, gt


def nearest_codes(d):
    players = d["players"] * SCALE
    ball = d["ball"][:, :2] * SCALE
    dist = np.linalg.norm(players - ball[:, None], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    k = dist.argmin(axis=1)
    team = d["team"][k]
    codes = np.where(team == "H", HOME, AWAY).astype(np.int8)
    codes[~np.isfinite(dist.min(axis=1))] = NOBODY
    return codes


def with_halves(d, labels):
    """Insert one X between the halves so no spell or transition spans half time."""
    cut = np.flatnonzero(np.diff(d["half"]) != 0)
    return np.insert(np.asarray(labels, dtype=object), cut + 1, "X")


def score(d, gt, pred):
    res = pm.evaluate(pm.label_codes(with_halves(d, gt)), pm.label_codes(with_halves(d, pred)))
    return res["macro_f1"], res["spell_f1_mean"]


def main():
    rows = {}
    for path in sorted(glob.glob(os.path.join(CONVERTED, "*.npz"))):
        d, gt = load(path)
        codes = nearest_codes(d)
        variants = {"nearest_w{}".format(w): smooth(codes, w) for w in (0, 25, 50, 125)}
        oracle = smooth(codes, 50).copy()
        oracle[gt == "D"] = "D"
        variants["nearest_w50_oracleD"] = oracle
        name = os.path.basename(path)[:-4]
        for v, pred in variants.items():
            rows.setdefault(v, []).append(score(d, gt, pred))
        print(name, {v: tuple(round(x, 3) for x in rows[v][-1]) for v in variants}, flush=True)
    print("\nmean over {} matches (macroF1, spellF1):".format(len(next(iter(rows.values())))))
    for v, r in rows.items():
        r = np.array(r)
        print("  {:<22} macroF1 {:.3f}  spellF1 {:.3f}  (spellF1 min {:.3f} max {:.3f})".format(
            v, r[:, 0].mean(), r[:, 1].mean(), r[:, 1].min(), r[:, 1].max()))


if __name__ == "__main__":
    main()
