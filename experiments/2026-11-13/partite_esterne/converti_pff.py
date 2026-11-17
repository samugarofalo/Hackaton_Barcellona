"""PFF FC, Mondiali 2022 (64 partite) nel formato delle partite DFL convertite.

Fonte: data/external/pff-wc2022/ (Drive PFF, accesso con iscrizione alla newsletter):
Tracking_Data/<id>.jsonl.bz2 (broadcast Sportlogiq, 29,97 fps, metri con origine al centro; chi e'
fuori inquadratura e' stimato), Event_Data/<id>.json, Metadata/<id>.json.
Si usano le posizioni grezze (`homePlayers`, `awayPlayers`, `balls`), non quelle lisciate: la palla
"lisciata" e' ricostruita dagli eventi.
Etichette, dagli eventi (secondi del video, lo stesso orologio del tracking):
  alive       1 dal calcio d'inizio alla fine di ogni tempo, 0 da fine dell'evento precedente
              all'inizio di ogni calcio piazzato (setpieceType diverso da O: rimessa, punizione,
              angolo, rinvio, rigore, calcio d'inizio dopo un gol, rimessa dell'arbitro), se il vuoto
              e' di almeno MIN_DEAD_S
  possession  regola a* (default, docs/soluzioni_problemi_finali.md §3.1): il possesso resta
              all'ultima squadra finche' un evento di gioco OTB dell'altra squadra mostra controllo
              (pff_eventi_brevi.controllo: azione voluta, respinta, tocco pesante o cessione al
              compagno, contrasto del portatore o vinto tenendo la palla); rimpalli, tocchi falliti e
              duelli senza controllo non lo cambiano. Il cambio e' all'inizio del primo evento
              accettato. Con $PFF_POSSESSO=oggi la regola di prima: passa a ogni evento OTB.
              '' a gioco fermo.
Ricampionato a 25 Hz (interpolazione lineare tra fotogrammi distanti al massimo 0,5 s).
Stampa solo aggregati.
"""

import bz2
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from converti_skillcorner import _interp

B = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(B, "experiments", "2026-11-14", "convenzioni"))
from pff_eventi_brevi import controllo  # noqa: E402

SRC = os.path.join(B, "data", "external", "pff-wc2022")
OUT = os.environ.get("PFF_CONVERTED") or os.path.join(SRC, "converted")
HZ = 25.0
MIN_DEAD_S = 0.5
RULE = os.environ.get("PFF_POSSESSO", "a*")


def labels(E, tout, rule=RULE):
    """alive (0/1) and possession ('H'/'A'/'') per output time, from the events E of one half (sorted
    by startTime). rule 'oggi': possession goes to the team of every OTB event; 'a*': only to a team
    whose game event shows control (any of its possession events, pff_eventi_brevi.controllo)."""
    if rule not in ("oggi", "a*"):
        raise ValueError("PFF_POSSESSO must be 'oggi' or 'a*', not {!r}".format(rule))
    ctrl = {}
    for x in E:
        pe = x.get("possessionEvents") or {}
        if pe.get("possessionEventType"):
            k = x.get("gameEventId")
            ctrl[k] = ctrl.get(k, False) or controllo(pe, x["gameEvents"].get("playerId"))
    alive = np.ones(len(tout), np.int8)
    poss = np.full(len(tout), "", "<U1")
    prev_end, cur = None, None
    for x in E:
        g = x["gameEvents"]
        s, e = x["startTime"], x["endTime"] if x["endTime"] is not None else x["startTime"]
        sp = g.get("setpieceType")
        if sp not in (None, "O") and prev_end is not None and s - prev_end >= MIN_DEAD_S:
            alive[(tout >= prev_end) & (tout < s)] = 0
        if g.get("gameEventType") == "OTB" and g.get("homeTeam") is not None:
            team = "H" if g["homeTeam"] else "A"
            if rule == "oggi" or cur is None or team == cur or ctrl.get(x.get("gameEventId"), False):
                if team != cur:
                    poss[tout >= s] = team
                    cur = team
        if g.get("gameEventType") in ("OTB", "OUT"):
            prev_end = max(prev_end or e, e)
    poss[alive == 0] = ""
    return alive, poss


def convert(gid):
    out_path = os.path.join(OUT, "PFF-{}.npz".format(gid))
    meta = json.load(open(os.path.join(SRC, "Metadata", gid + ".json")))
    meta = meta[0] if isinstance(meta, list) else meta
    pitch = meta["stadium"]["pitches"][0]
    L, W = float(pitch["length"]), float(pitch["width"])
    ev = json.load(open(os.path.join(SRC, "Event_Data", gid + ".json")))
    ev = sorted(ev, key=lambda x: (x["startTime"] or 0))
    # tracking: un fotogramma per frameNum (i duplicati portano solo eventi diversi)
    seen, T, rows = set(), [], {}
    with bz2.open(os.path.join(SRC, "Tracking_Data", gid + ".jsonl.bz2"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["frameNum"] in seen or r.get("period") not in (1, 2):
                continue
            seen.add(r["frameNum"])
            i = len(T)
            b = (r.get("balls") or [{}])[0] or {}
            T.append((r["period"], r["videoTimeMs"] / 1000.0, b.get("x"), b.get("y"), b.get("z")))
            for side, key in (("H", "homePlayers"), ("A", "awayPlayers")):
                for p in r.get(key) or []:
                    if p.get("x") is not None:
                        rows.setdefault((side, str(p.get("jerseyNum"))), []).append((i, p["x"], p["y"]))
    T = np.array([[t[0], t[1]] + [np.nan if v is None else v for v in t[2:]] for t in T], float)
    cols = sorted(rows)
    raw = np.full((len(T), len(cols), 2), np.nan)
    for j, c in enumerate(cols):
        a = np.array(rows[c])
        raw[a[:, 0].astype(int), j] = a[:, 1:]
    o = {k: [] for k in ("frame", "half", "players", "ball", "possession", "alive")}
    for h in (1, 2):
        sel = (T[:, 0] == h)
        tin = T[sel, 1]
        # inizio e fine del tempo dai metadati; dove mancano, dal tracking stesso
        t0 = meta.get("startPeriod{}".format(h)) or tin.min()
        t1 = meta.get("endPeriod{}".format(h)) or tin.max()
        tout = np.arange(t0, t1, 1 / HZ)
        P = np.stack([np.stack([_interp(tin, raw[sel, j, k], tout) for k in range(2)], -1) for j in range(len(cols))], 1)
        P = np.stack([(P[..., 0] + L / 2) / L, (P[..., 1] + W / 2) / W], -1)
        Bl = np.stack([_interp(tin, T[sel, 2 + k], tout) for k in range(3)], -1)
        ball = np.stack([(Bl[:, 0] + L / 2) / L, (Bl[:, 1] + W / 2) / W, np.nan_to_num(Bl[:, 2])], -1)
        E = [x for x in ev if (x.get("gameEvents") or {}).get("period") == h and x["startTime"] is not None]
        alive, poss = labels(E, tout)
        o["frame"].append(np.round((tout - t0) * HZ).astype(int))
        o["half"].append(np.full(len(tout), h))
        o["players"].append(P.astype(np.float32))
        o["ball"].append(ball.astype(np.float32))
        o["possession"].append(poss)
        o["alive"].append(alive)
    o = {k: np.concatenate(v) for k, v in o.items()}
    F = len(o["frame"])
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(out_path, team=np.array([c[0] for c in cols]), keeper=np.zeros(len(cols), bool),
                        referees=np.zeros((F, 0, 2), np.float32), **o)
    on = np.isfinite(o["players"][:, :, 0]).sum(1)
    vivo = o["alive"] == 1
    return ("PFF-{} (possesso {}): {} fotogrammi, giocatori mediana {}, vivo {:.1%}, possesso H {:.1%} A {:.1%}, "
            "vivo senza squadra {:.1%}, palla a gioco vivo {:.1%}").format(
        gid, RULE, F, int(np.median(on)), vivo.mean(), (o["possession"] == "H").mean(), (o["possession"] == "A").mean(),
        np.mean(o["possession"][vivo] == ""), np.isfinite(o["ball"][vivo, 0]).mean())


def main():
    import sys
    ids = sys.argv[1:] or sorted(os.path.basename(p).split(".")[0]
                                 for p in glob.glob(os.path.join(SRC, "Tracking_Data", "*.jsonl.bz2")))
    ids = [i for i in ids if os.path.exists(os.path.join(SRC, "Event_Data", i + ".json"))
           and os.path.exists(os.path.join(SRC, "Metadata", i + ".json"))]
    ids = [i for i in ids if not os.path.exists(os.path.join(OUT, "PFF-{}.npz".format(i)))]
    with ProcessPoolExecutor(8) as ex:
        for line in ex.map(_safe, ids):
            print(line, flush=True)


def _safe(gid):
    try:
        return convert(gid)
    except Exception as e:  # una partita difettosa non ferma le altre
        return "PFF-{}: NON convertita ({}: {})".format(gid, type(e).__name__, e)


if __name__ == "__main__":
    main()
