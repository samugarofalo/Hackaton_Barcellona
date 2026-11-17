"""Parametri finali di A1 per la consegna (soluzioni_problemi_finali §3.4, 15/11).

Dalla griglia di a1_valuta.py (<run>/a1_griglia_<tag>.csv) sceglie la colonna con la media piu' alta sul
gruppo "controllo" ($POOL, default ST-,MS-,PFF-), con peso uguale per dataset, usando TUTTE le partite
del gruppo (per la consegna non c'e' partita da lasciare fuori). Scrive decoder_params.json nel formato
letto da make_submission.py (b1_window se la colonna ha il suffisso _b1m<finestra>). Solo aggregati.

    python parametri_a1.py <griglia.csv> <cartella di uscita>
"""
import json
import os
import re
import sys

import pandas as pd

POOL = [p for p in os.environ.get("POOL", "ST-,MS-,PFF-").split(",") if p]
NOME = re.compile(r"^sw([\d.]+)_min(\d+)/(\d+)_net([\d.]+)(?:_b1m(\d+))?$")


def scegli(grid, pool=POOL):
    g = grid.loc[[m for m in grid.index if any(m.startswith(p) for p in pool)]]
    medie = g.groupby(lambda m: m.split("-")[0]).mean().mean(axis=0)
    best = sorted(medie.index, key=lambda n: (-medie[n], n))[0]
    return best, medie[best]


def parametri(nome):
    sc, ha, dd, w, b1 = NOME.match(nome).groups()
    par = dict(switch_cost=float(sc), min_dur=dict(H=int(ha), A=int(ha), D=int(dd)), net_weight=float(w))
    if b1:
        par["b1_window"] = int(b1)
    return par


if __name__ == "__main__":
    if len(sys.argv) == 1:   # controllo
        assert parametri("sw64_min12/25_net4_b1m51") == dict(switch_cost=64.0, min_dur=dict(H=12, A=12, D=25),
                                                             net_weight=4.0, b1_window=51)
        assert "b1_window" not in parametri("sw4_min25/50_net0")
        g = pd.DataFrame({"a": [0.5, 0.9, 0.1], "b": [0.6, 0.6, 0.9]}, index=["ST-1", "PFF-1", "SC-1"])
        assert scegli(g)[0] == "a"   # SC fuori dal gruppo; ST e PFF pesano uguale
        print("ok")
    else:
        grid = pd.read_csv(sys.argv[1], index_col=0)
        nome, media = scegli(grid)
        par = dict(parametri(nome), scelta=nome, media_controllo=round(float(media), 4), pool=POOL,
                   griglia=os.path.basename(sys.argv[1]))
        os.makedirs(sys.argv[2], exist_ok=True)
        with open(os.path.join(sys.argv[2], "decoder_params.json"), "w") as fh:
            json.dump(par, fh, indent=1)
        print(json.dumps(par))
