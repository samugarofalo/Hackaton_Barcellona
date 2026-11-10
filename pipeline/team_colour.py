"""Team from shirt colour, learned from the supplied tracking instead of guessed by k-means.

compare_tracking.py reduced each box to one median colour of the torso, which mixes Athletic's
stripes into pink and lets sun/shade outweigh the kit. Here each box gets a Lab colour histogram of
the shirt and of the shorts (grass removed), and a small linear classifier learns home vs away from
the boxes paired one-to-one with a Metrica point -- the supplied tracking, not possession labels.

Cue per box (used by identities.py):
  other  officials: the shirt-colour group compare_tracking already marked "other"
  GK     goalkeepers: a colour group whose Metrica-paired boxes mix both teams (>= 15% each); in
         this match the two keepers wear similar kits, so their team comes from the half they
         defend, not from colour
  H / A  classifier probability >= CONFIDENT for that side
  ""     unsure

    python pipeline/team_colour.py --start 11185 --frames 7500
Writes data/diagnostics/tracking-comparison/team_colour_<tag>.csv (det row, p_home, cue) and
prints the held-out check: trained on the first half of the clip, scored on the second.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import require  # noqa: E402
from compare_tracking import OUT_DIR, W, decode  # noqa: E402

BINS = 4  # per Lab axis -> 64 bins per region
REGIONS = {"shirt": (0.15, 0.50), "shorts": (0.50, 0.70)}  # fractions of box height
CONFIDENT = 0.8
GK_MIX = 0.15


def histogram(lab, hsv):
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 50) & (hsv[:, 2] < 200)
    keep = lab[~grass]
    if len(keep) < 5:
        return np.zeros(BINS ** 3, np.float32)
    q = (keep.astype(np.int32) * BINS) // 256
    h = np.bincount(q[:, 0] * BINS * BINS + q[:, 1] * BINS + q[:, 2], minlength=BINS ** 3)
    return (h / h.sum()).astype(np.float32)


def features(video, det, start, n):
    feats = np.zeros((len(det), BINS ** 3 * len(REGIONS)), np.float32)
    rows = {f: g for f, g in det.groupby("frame")}
    pos = {d: i for i, d in enumerate(det.index)}
    for i, image in enumerate(decode(video, start, n)):
        g = rows.get(start + i)
        if g is None:
            continue
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for r in g.itertuples():
            h = r.y2 - r.y1
            w = r.x2 - r.x1
            xs, xe = int(max(0, r.x1 + 0.2 * w)), int(min(W, r.x2 - 0.2 * w))
            parts = []
            for top, bottom in REGIONS.values():
                ys, ye = int(r.y1 + top * h), int(r.y1 + bottom * h)
                parts.append(histogram(lab[ys:ye, xs:xe].reshape(-1, 3),
                                       hsv[ys:ye, xs:xe].reshape(-1, 3)))
            feats[pos[r.Index]] = np.concatenate(parts)
    return feats


def train(x, y, epochs=300):
    """Logistic regression (torch, already installed); returns a predict(x) -> p_home."""
    import torch
    xt, yt = torch.tensor(x), torch.tensor(y, dtype=torch.float32)
    model = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
    loss = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss(model(xt).squeeze(1), yt).backward()
        opt.step()
    return lambda z: torch.sigmoid(model(torch.tensor(z)).squeeze(1)).detach().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, default=11185)
    parser.add_argument("--frames", type=int, default=7500)
    args = parser.parse_args()
    start, n = args.start, args.frames
    tag = "{}_{}".format(start, start + n - 1)
    det = pd.read_csv(os.path.join(OUT_DIR, "detections_{}.csv".format(tag)))
    det = det[det.on_pitch == 1]

    x = features(require(["video"])[0], det, start, n)
    groups = det.groupby("group").matched_team.value_counts().unstack(fill_value=0)
    share = groups.min(axis=1) / groups.sum(axis=1).clip(lower=1)
    gk_groups = set(share[share >= GK_MIX].index) - set(
        det[det.group_role == "other"].group.unique()) - {-1}  # -1: box without colour
    officials = det.group_role.eq("other").values
    keepers = det.group.isin(gk_groups).values

    labelled = det.matched_team.isin(["H", "A"]).values & ~officials & ~keepers
    y = (det.matched_team == "H").values.astype(np.float32)
    early = det.frame.values < start + n // 2

    # Held-out check: learn on the first half of the clip, score the second.
    predict = train(x[labelled & early], y[labelled & early])
    test = labelled & ~early
    p = predict(x[test])
    sure = (p >= CONFIDENT) | (p <= 1 - CONFIDENT)
    new_acc = ((p >= 0.5) == (y[test] == 1)).mean()
    old = det.group_role.values[test]
    old_acc = (old == det.matched_team.values[test]).mean()
    print("held-out boxes (second half, paired to Metrica, no officials/keepers): {}".format(test.sum()))
    print("  k-means group cue agrees with Metrica team:   {:.4f}".format(old_acc))
    print("  histogram classifier agrees (all):            {:.4f}".format(new_acc))
    print("  confident share {:.3f}, agreement when confident {:.4f}".format(
        sure.mean(), ((p[sure] >= 0.5) == (y[test][sure] == 1)).mean()))
    print("keeper colour groups: {} ({} boxes)".format(sorted(gk_groups), int(keepers.sum())))

    # Final model on every labelled box of the clip.
    p_home = train(x[labelled], y[labelled])(x)
    cue = np.where(p_home >= CONFIDENT, "H", np.where(p_home <= 1 - CONFIDENT, "A", ""))
    cue = np.where(keepers, "GK", cue)
    cue = np.where(officials, "other", cue)
    out = pd.DataFrame({"det": det.index, "p_home": np.round(p_home, 4), "cue": cue})
    out.to_csv(os.path.join(OUT_DIR, "team_colour_{}.csv".format(tag)), index=False)
    print(out.cue.value_counts().to_dict())


if __name__ == "__main__":
    main()
