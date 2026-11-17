"""Convert the Bassek et al. (2025) DFL dataset to arrays shaped like our ATD.

Data: 7 Bundesliga matches, CC BY 4.0, https://doi.org/10.6084/m9.figshare.28196177, downloaded
to data/external/bassek2025/. Other games, so usable to train and tune (RULES.md: data about
other games is fine). Positions are metres with the origin at the centre spot; converted to the
ATD's normalised 0-1 so the same code runs on both.

Per match, data/external/bassek2025/converted/<match>.npz:
    frame (F,)        DFL frame number N (25 Hz), both halves concatenated
    half (F,)         1 or 2
    players (F,P,2)   normalised x,y, NaN when the person is not on the pitch
    team (P,)         'H' home / 'A' away
    keeper (P,)       True for goalkeepers (PlayingPosition TW)
    referees (F,R,2)
    ball (F,3)        normalised x,y and height z in metres
    possession (F,)   'H' / 'A' / '' -- BallPossession 1 = home, 2 = away
    alive (F,)        BallStatus: 1 ball in play, 0 dead

    python pipeline/convert_dfl.py
    DFL_SRC=/kaggle/input/<dataset> DFL_CONVERTED=/kaggle/working/dfl_converted python pipeline/convert_dfl.py
"""

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# $DFL_SRC: folder with the original XML (e.g. a read-only Kaggle input); $DFL_CONVERTED: where the
# .npz go. Both default to the local layout.
SRC = os.environ.get("DFL_SRC") or os.path.join(ROOT, "data", "external", "bassek2025")
OUT = os.environ.get("DFL_CONVERTED") or os.path.join(SRC, "converted")
LENGTH, WIDTH = 105.0, 68.0

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]+" TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?'
                   r'(?:[^>]*BallPossession="(\d)")?(?:[^>]*BallStatus="(\d)")?')


def match_info(match):
    path = glob.glob(os.path.join(SRC, "DFL_02_01_matchinformation_*_{}.xml".format(match)))[0]
    root = ET.parse(path).getroot()
    team_role, players = {}, {}
    for team in root.iter("Team"):
        role = "H" if team.get("Role") == "home" else "A"
        team_role[team.get("TeamId")] = role
        for p in team.iter("Player"):
            players[p.get("PersonId")] = (role, p.get("PlayingPosition") == "TW")
    return team_role, players


def convert(match):
    team_role, players = match_info(match)
    path = glob.glob(os.path.join(SRC, "DFL_04_03_positions_raw_observed_*_{}.xml".format(match)))[0]
    series = {}  # (kind, person) -> {(half, N): values}
    current = None
    with open(path) as handle:
        for line in handle:
            m = FRAMESET.search(line)
            if m:
                section, team, person = m.groups()
                half = 1 if section == "firstHalf" else 2
                kind = "ball" if team == "BALL" else "ref" if team == "referee" else "player"
                current = (kind, person, half)
                series.setdefault((kind, person), {})
                continue
            m = FRAME.search(line)
            if m and current:
                n, x, y, z, poss, status = m.groups()
                series[(current[0], current[1])][(current[2], int(n))] = (
                    float(x), float(y), float(z or 0), poss, status)

    ball = series[next(k for k in series if k[0] == "ball")]
    keys = sorted(ball)  # (half, N): the ball defines the timeline
    index = {k: i for i, k in enumerate(keys)}
    F = len(keys)

    def norm(x, y):
        return (x + LENGTH / 2) / LENGTH, (y + WIDTH / 2) / WIDTH

    people = [k for k in series if k[0] == "player"]
    refs = [k for k in series if k[0] == "ref"]
    P = np.full((F, len(people), 2), np.nan, np.float32)
    R = np.full((F, len(refs), 2), np.nan, np.float32)
    for arr, group in ((P, people), (R, refs)):
        for j, key in enumerate(group):
            for k, (x, y, *_rest) in series[key].items():
                if k in index:
                    arr[index[k], j] = norm(x, y)
    B = np.zeros((F, 3), np.float32)
    possession = np.full(F, "", dtype="<U1")
    alive = np.zeros(F, np.int8)
    for k, (x, y, z, poss, status) in ball.items():
        i = index[k]
        B[i, :2] = norm(x, y)
        B[i, 2] = z
        possession[i] = {"1": "H", "2": "A"}.get(poss, "")
        alive[i] = int(status or 0)
    team = np.array([players.get(k[1], ("?", False))[0] for k in people])
    keeper = np.array([players.get(k[1], ("?", False))[1] for k in people])
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, match + ".npz"),
                        frame=np.array([k[1] for k in keys]), half=np.array([k[0] for k in keys]),
                        players=P, team=team, keeper=keeper, referees=R, ball=B,
                        possession=possession, alive=alive)
    on = np.isfinite(P[:, :, 0]).sum(axis=1)
    print("{}: {} frames, players on pitch median {}, alive {:.1%}, possession H {:.1%} A {:.1%}, "
          "keepers {}, unknown team {}".format(
              match, F, int(np.median(on)), alive.mean(), (possession == "H").mean(),
              (possession == "A").mean(), int(keeper.sum()), int((team == "?").sum())), flush=True)


def main():
    matches = sorted({re.search(r"(DFL-MAT-\w+)\.xml$", p).group(1)
                      for p in glob.glob(os.path.join(SRC, "DFL_04_03_*.xml"))})
    for m in matches:
        if not os.path.exists(os.path.join(OUT, m + ".npz")) or "--force" in sys.argv:
            convert(m)


if __name__ == "__main__":
    main()
