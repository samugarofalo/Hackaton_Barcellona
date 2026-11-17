"""SoccerTrack v2 (10 partite universitarie, CC BY 4.0) nel formato delle partite DFL convertite.

Fonte: il tracking del fornitore in data/external/soccertrack-v2/raw/<partita>/
<partita>_tracker_box_data.xml (+ _metadata.xml): per fotogramma (25 Hz) posizioni normalizzate
0-1 di giocatori e palla e `ballStatus` HOME / AWAY / NEUTRAL / BALLOUT.
Uscita come pipeline/convert_dfl.py: frame, half, players (F,P,2), team (P,) H/A, keeper (P,),
referees (F,0,2), ball (F,3) con altezza 0 (non misurata), possession (F,) H/A/'', alive (F,).
  BALLOUT -> alive 0; HOME / AWAY -> possesso; NEUTRAL (conteso) -> alive 1 e possesso della
  squadra che l'aveva prima nel tratto vivo (regola della gara: nei contrasti il possesso resta),
  o della prossima se il tratto comincia conteso.
Stampa solo aggregati.

    python converti_soccertrack.py   (con il .venv del progetto)
"""

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import numpy as np

B = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(B, "data", "external", "soccertrack-v2", "raw")
OUT = os.environ.get("ST_CONVERTED") or os.path.join(B, "data", "external", "soccertrack-v2", "converted")
LOC = re.compile(r"\[([-\d.eE+]+|nan), ([-\d.eE+]+|nan)\]")
HALF = {"FIRST_HALF": 1, "SECOND_HALF": 2}


def _loc(s):
    m = LOC.match(s or "")
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def fill_neutral(poss, alive):
    """'' a gioco vivo (conteso) -> ultima squadra in possesso nel tratto vivo, altrimenti la prossima."""
    out = poss.copy()
    edge = np.diff(np.r_[0, alive.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        seg = out[a:z]
        last = ""
        for i in range(len(seg)):
            if seg[i]:
                last = seg[i]
            elif last:
                seg[i] = last
        nxt = ""
        for i in range(len(seg) - 1, -1, -1):
            if seg[i]:
                nxt = seg[i]
            elif nxt:
                seg[i] = nxt
    return out


def convert(match):
    meta = ET.parse(os.path.join(SRC, match, match + "_tracker_box_metadata.xml")).getroot()
    side = {t.get("id"): ("H" if t.get("side") == "home" else "A") for t in meta.iter("team")}
    info = {p.get("id"): (side.get(p.get("teamId"), "?"), p.get("position") == "GK") for p in meta.iter("player")}
    frames, rows = [], {}
    for _, el in ET.iterparse(os.path.join(SRC, match, match + "_tracker_box_data.xml")):
        if el.tag != "frame":
            continue
        h = HALF.get(el.get("eventPeriod"))
        if h is None:
            el.clear()
            continue
        i = len(frames)
        ball = (np.nan, np.nan)
        for c in el:
            if c.tag == "ball":
                ball = _loc(c.get("loc"))
            elif c.tag == "player":
                rows.setdefault(c.get("playerId"), []).append((i,) + _loc(c.get("loc")))
        frames.append((h, int(float(el.get("frameNumber"))), el.get("ballStatus"), ball))
        el.clear()
    F = len(frames)
    pids = [p for p in rows if p in info]
    P = np.full((F, len(pids), 2), np.nan, np.float32)
    for j, p in enumerate(pids):
        r = np.array(rows[p], np.float64)
        P[r[:, 0].astype(int), j] = r[:, 1:]
    status = np.array([f[2] for f in frames])
    alive = (status != "BALLOUT").astype(np.int8)
    poss = np.where(status == "HOME", "H", np.where(status == "AWAY", "A", "")).astype("<U1")
    poss = fill_neutral(poss, alive)
    ball = np.zeros((F, 3), np.float32)
    ball[:, :2] = np.array([f[3] for f in frames], np.float32)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "ST-{}.npz".format(match)),
                        frame=np.array([f[1] for f in frames]), half=np.array([f[0] for f in frames]),
                        players=P, team=np.array([info[p][0] for p in pids]),
                        keeper=np.array([info[p][1] for p in pids]), referees=np.zeros((F, 0, 2), np.float32),
                        ball=ball, possession=poss, alive=alive)
    on = np.isfinite(P[:, :, 0]).sum(1)
    return ("ST-{}: {} fotogrammi, giocatori in campo mediana {}, vivo {:.1%}, conteso {:.1%}, "
            "possesso H {:.1%} A {:.1%}, palla presente {:.1%}, portieri {}").format(
        match, F, int(np.median(on)), alive.mean(), (status == "NEUTRAL").mean(), (poss == "H").mean(),
        (poss == "A").mean(), np.isfinite(ball[:, 0]).mean(), int(sum(info[p][1] for p in pids)))


def main():
    matches = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SRC, "*")) if os.path.isdir(p))
    with ProcessPoolExecutor(5) as ex:
        for line in ex.map(convert, matches):
            print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
