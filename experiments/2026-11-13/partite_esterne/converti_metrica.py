"""Metrica Sports Sample Data, partite 1 e 2 (tracking + eventi dello stesso fornitore della gara)
nel formato delle partite DFL convertite.

Fonte: data/external/metrica-sample/data/Sample_Game_<n>/ (CSV, 25 Hz, coordinate 0-1, palla di
Metrica con i suoi vuoti). La partita 3 e' in un altro formato (EPTS + json): non convertita.
Etichette ricavate dagli eventi (approssimate, nessuna etichetta del fornitore per fotogramma):
  alive      (default, 15/11) 0 dalla fine dell'ultima azione prima di ogni ripresa fino alla ripresa,
             se il vuoto e' di almeno MIN_DEAD_S; ripresa = SET PIECE (tranne il calcio d'inizio del
             tempo) o evento con sottotipo GOAL KICK (il rinvio dal fondo e' un PASS, non un SET PIECE).
             Come converti_pff.py: prende anche le soste senza BALL OUT / FAULT RECEIVED prima
             (fuorigioco, falli senza FAULT RECEIVED). 1 altrove dal primo calcio d'inizio di ogni
             tempo alla fine. Con $MS_VIVO=oggi la regola di prima: 0 da un BALL OUT, da un FAULT
             RECEIVED o da un tiro finito in gol fino all'inizio del SET PIECE successivo
  possession squadra dell'ultimo PASS / RECOVERY / SET PIECE / SHOT iniziato; BALL LOST e CHALLENGE
             non cambiano niente (il possesso passa col RECOVERY dell'altra squadra, regola della gara)
Stampa solo aggregati.
"""

import os

import numpy as np
import pandas as pd

B = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(B, "data", "external", "metrica-sample", "data")
OUT = os.environ.get("MS_CONVERTED") or os.path.join(B, "data", "external", "metrica-sample", "converted")
GIVE = {"PASS", "RECOVERY", "SET PIECE", "SHOT"}
RULE = os.environ.get("MS_VIVO", "riprese")
MIN_DEAD_S = 0.5          # come converti_pff.py
FPS = 25


def is_restart(r):
    sub = r.Subtype if isinstance(r.Subtype, str) else ""
    return r.Type == "SET PIECE" or "GOAL KICK" in sub


def tracking(path):
    h = pd.read_csv(path, skiprows=2)
    names = [c for c in h.columns if c.startswith("Player")]
    xy = np.stack([h[[c, h.columns[h.columns.get_loc(c) + 1]]].to_numpy(np.float32) for c in names], 1)
    ball = h[["Ball", h.columns[h.columns.get_loc("Ball") + 1]]].to_numpy(np.float32)
    return h["Period"].to_numpy(), h["Frame"].to_numpy(), xy, ball


def convert(n):
    d = os.path.join(SRC, "Sample_Game_{}".format(n))
    per, frame, ph, ball = tracking(os.path.join(d, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(n)))
    _, _, pa, _ = tracking(os.path.join(d, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(n)))
    ev = pd.read_csv(os.path.join(d, "Sample_Game_{}_RawEventsData.csv".format(n)))
    F = len(frame)
    idx = {f: i for i, f in enumerate(frame)}
    alive = np.zeros(F, np.int8)
    poss = np.full(F, "", "<U1")
    for p in (1, 2):
        e = ev[ev.Period == p].sort_values("Start Frame")
        rows = np.flatnonzero(per == p)
        ko = e[(e.Type == "SET PIECE")]["Start Frame"].min()
        a, z = idx.get(int(ko), rows[0]), rows[-1] + 1
        alive[a:z] = 1
        stop, prev_end = None, None
        for _, r in e.iterrows():
            s = idx.get(int(r["Start Frame"]))
            if s is None:
                continue
            if RULE == "oggi":
                goal = r.Type == "SHOT" and isinstance(r.Subtype, str) and "GOAL" in r.Subtype
                if r.Type in ("BALL OUT", "FAULT RECEIVED") or goal:
                    stop = idx.get(int(r["End Frame"]), s) if r.Type == "BALL OUT" or goal else s
                elif r.Type == "SET PIECE" and stop is not None:
                    alive[stop:s] = 0
                    stop = None
            else:
                if is_restart(r) and s > a and prev_end is not None and s - prev_end >= MIN_DEAD_S * FPS:
                    alive[prev_end:s] = 0
                if r.Type != "CARD":  # cards are shown while play is stopped
                    end = idx.get(int(r["End Frame"]), s)
                    prev_end = max(prev_end or end, end)
            if r.Type in GIVE:
                poss[s:z] = "H" if r.Team == "Home" else "A"
    poss[alive == 0] = ""
    players = np.concatenate([ph, pa], 1)
    team = np.array(["H"] * ph.shape[1] + ["A"] * pa.shape[1])
    os.makedirs(OUT, exist_ok=True)
    b3 = np.zeros((F, 3), np.float32)
    b3[:, :2] = ball
    np.savez_compressed(os.path.join(OUT, "MS-{}.npz".format(n)), frame=frame, half=per, players=players,
                        team=team, keeper=np.zeros(len(team), bool), referees=np.zeros((F, 0, 2), np.float32),
                        ball=b3, possession=poss, alive=alive)
    on = np.isfinite(players[:, :, 0]).sum(1)
    vivo = alive == 1
    return ("MS-{}: {} fotogrammi, giocatori mediana {}, vivo {:.1%}, possesso H {:.1%} A {:.1%}, "
            "palla presente a gioco vivo {:.1%} / fermo {:.1%}").format(
        n, F, int(np.median(on)), vivo.mean(), (poss == "H").mean(), (poss == "A").mean(),
        np.isfinite(ball[vivo, 0]).mean(), np.isfinite(ball[~vivo, 0]).mean())


if __name__ == "__main__":
    for n in (1, 2):
        print(convert(n), flush=True)
