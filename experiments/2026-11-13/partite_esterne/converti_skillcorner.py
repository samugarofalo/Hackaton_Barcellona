"""SkillCorner Open Data (20 partite A-League 2024/25, licenza MIT) nel formato delle partite DFL convertite.

Fonte: data/external/skillcorner-opendata/data/matches/<id>/: tracking da broadcast a 10 Hz
(giocatori fuori inquadratura stimati), <id>_match.json (squadre, campo), <id>_phases_of_play.csv.
Etichette come DFL:
  alive      1 dentro una fase di gioco (le fasi esistono solo a palla in gioco), 0 fuori
  possession squadra del giocatore che ha la palla (`possession.group`); dove nessuno ce l'ha
             (passaggio, palla libera) a gioco vivo, l'ultima squadra che l'aveva nel tratto vivo
             (regola della gara), o la prossima se il tratto comincia cosi'
Ricampionato a 25 Hz (interpolazione lineare delle posizioni tra fotogrammi vicini, etichette dal
fotogramma piu' vicino). Posizioni normalizzate 0-1 sulle dimensioni del campo del match. Altezza
della palla da `z` dove c'e'. Stampa solo aggregati.
"""

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from converti_soccertrack import fill_neutral

B = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(B, "data", "external", "skillcorner-opendata", "data", "matches")
OUT = os.environ.get("SC_CONVERTED") or os.path.join(B, "data", "external", "skillcorner-opendata", "converted")
HZ_IN, HZ_OUT = 10.0, 25.0


def _interp(t_in, v_in, t_out, max_gap=0.5):
    """Lineare dove i due vicini esistono e distano <= max_gap s, NaN altrove."""
    ok = np.isfinite(v_in)
    out = np.full(len(t_out), np.nan)
    if ok.sum() < 2:
        return out
    ti, vi = t_in[ok], v_in[ok]
    j = np.clip(np.searchsorted(ti, t_out), 1, len(ti) - 1)
    t0, t1 = ti[j - 1], ti[j]
    good = (t_out >= t0) & (t_out <= t1) & (t1 - t0 <= max_gap)
    w = np.where(t1 > t0, (t_out - t0) / np.maximum(t1 - t0, 1e-9), 0)
    out[good] = (vi[j - 1] + w * (vi[j] - vi[j - 1]))[good]
    return out


def convert(mid):
    d = os.path.join(SRC, mid)
    meta = json.load(open(os.path.join(d, mid + "_match.json")))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home = meta["home_team"]["id"]
    team_of = {p["id"]: ("H" if p["team_id"] == home else "A") for p in meta["players"]}
    keeper = {p["id"]: p["player_role"]["acronym"] == "GK" for p in meta["players"]}
    rows = [json.loads(l) for l in open(os.path.join(d, mid + "_tracking_extrapolated.jsonl"))]
    rows = [r for r in rows if r["period"] in (1, 2)]
    ph = pd.read_csv(os.path.join(d, mid + "_phases_of_play.csv"))
    pids = sorted({p["player_id"] for r in rows for p in r["player_data"] if p["player_id"] in team_of})
    col = {p: i for i, p in enumerate(pids)}
    out = {k: [] for k in ("frame", "half", "players", "ball", "possession", "alive")}
    for half in (1, 2):
        R = [r for r in rows if r["period"] == half]
        fin = np.array([r["frame"] for r in R], float)
        t_in = (fin - fin[0]) / HZ_IN
        n_out = int(t_in[-1] * HZ_OUT) + 1
        t_out = np.arange(n_out) / HZ_OUT
        P = np.full((len(R), len(pids), 2), np.nan)
        Bl = np.full((len(R), 3), np.nan)
        grp = np.empty(len(R), "<U1")
        for i, r in enumerate(R):
            for p in r["player_data"]:
                if p["player_id"] in col and p["x"] is not None:
                    P[i, col[p["player_id"]]] = ((p["x"] + L / 2) / L, (p["y"] + W / 2) / W)
            b = r["ball_data"]
            if b.get("x") is not None:
                Bl[i] = ((b["x"] + L / 2) / L, (b["y"] + W / 2) / W, b.get("z") or 0.0)
            g = r["possession"]["group"]
            grp[i] = "H" if g == "home team" else ("A" if g == "away team" else "")
        alive_in = np.zeros(len(R), np.int8)
        for _, p in ph[ph.period == half].iterrows():
            alive_in[(fin >= p.frame_start) & (fin <= p.frame_end)] = 1
        near = np.clip(np.round(t_out * HZ_IN).astype(int), 0, len(R) - 1)
        Po = np.stack([np.stack([_interp(t_in, P[:, j, k], t_out) for k in range(2)], -1)
                       for j in range(len(pids))], 1)
        Bo = np.stack([_interp(t_in, Bl[:, k], t_out) for k in range(3)], -1)
        Bo[:, 2] = np.nan_to_num(Bo[:, 2])
        alive = alive_in[near]
        poss = fill_neutral(np.where(alive == 1, grp[near], ""), alive)
        out["frame"].append(np.arange(n_out))
        out["half"].append(np.full(n_out, half))
        out["players"].append(Po.astype(np.float32))
        out["ball"].append(Bo.astype(np.float32))
        out["possession"].append(poss.astype("<U1"))
        out["alive"].append(alive)
    o = {k: np.concatenate(v) for k, v in out.items()}
    F = len(o["frame"])
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "SC-{}.npz".format(mid)), team=np.array([team_of[p] for p in pids]),
                        keeper=np.array([keeper.get(p, False) for p in pids]),
                        referees=np.zeros((F, 0, 2), np.float32), **o)
    on = np.isfinite(o["players"][:, :, 0]).sum(1)
    vivo = o["alive"] == 1
    return ("SC-{}: {} fotogrammi, giocatori mediana {}, vivo {:.1%}, possesso H {:.1%} A {:.1%}, "
            "vivo senza squadra {:.1%}, palla presente {:.1%}").format(
        mid, F, int(np.median(on)), vivo.mean(), (o["possession"] == "H").mean(), (o["possession"] == "A").mean(),
        np.mean(o["possession"][vivo] == ""), np.isfinite(o["ball"][:, 0]).mean())


def main():
    mids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SRC, "*")) if os.path.isdir(p))
    with ProcessPoolExecutor(6) as ex:
        for line in ex.map(convert, mids):
            print(line, flush=True)


if __name__ == "__main__":
    main()
