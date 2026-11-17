"""A1 in parallelo su molte partite, parametri scelti su un sottoinsieme ($DFL_SCELTA, es. ST-),
risultato per dataset. Stessa simulazione (seme 0) e stessi file B1/B2/C1 di $DFL_OUT.

Griglia: costi di cambio e durate minime di decoder.grid() x pesi della rete in $DFL_NET_WEIGHTS.
Opzionale $A1_NET_FILE=<suffisso>: al posto di <partita>_net.npz legge <partita>_<suffisso>.npz
(es. P(casa) imparato da pesi_contesto.py); con $A1_SOLO_NET=1 la rete e' l'unico indizio H/A.
Per ogni partita i parametri sono scelti sulle partite del sottoinsieme diverse da lei. Solo
aggregati.

    DFL_CONVERTED=... DFL_OUT=... DFL_SIM=medium DFL_DEAD_BALL=moving DFL_SCELTA=ST- \
    DFL_NET_WEIGHTS=0,2,4,6,8,10 python a1_valuta.py   (dalla cartella del codice)
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path[:0] = [".", "pipeline", "starter", "evaluation"]
import decoder as dec  # noqa: E402
import dfl_bench  # noqa: E402
import dfl_paths  # noqa: E402


def una(path):
    m = dfl_paths.match_name(path)
    npz, gt = dfl_bench.load(path)
    d = {k: npz[k] for k in ("players", "team", "ball", "alive", "half")}
    ev = dec.simulate(d, dec.SEEDS[0])
    ext = dec._external(dfl_paths, m, len(gt))
    alt = os.environ.get("A1_NET_FILE")
    if alt:
        ext["p_home_net"] = np.load(os.path.join(dfl_paths.out_dir(), "{}_{}.npz".format(m, alt)))["p_home"].astype(float)
    solo = os.environ.get("A1_SOLO_NET") == "1"
    # $A1_B1_WINDOWS (es. "1,25,51,91", soluzioni_problemi_finali §3.3.1): il filtro mediano di B1 lo
    # sceglie A1 insieme agli altri parametri, dalla probabilita' grezza salvata da deadball.py (p_raw)
    windows = [None]
    if os.environ.get("A1_B1_WINDOWS"):
        if "p_raw" not in ext:
            raise FileNotFoundError("{}: manca p_raw in <partita>_deadball.npz, rifare B1".format(m))
        from deadball import median_filter
        windows = [int(x) for x in os.environ["A1_B1_WINDOWS"].split(",")]
    out = {}
    for wb in windows:
        ext_w = ext if wb is None else dict(ext, p_dead=median_filter(ext["p_raw"], wb, d["half"]).astype(float))
        suffix = "" if wb is None else "_b1m{}".format(wb)
        for w in dec.NET_WEIGHTS:
            if solo:
                lp = dec.frame_logp(ev["near"], ev["dist"], ball_missing=ev["ball_missing"], p_dead=ext_w.get("p_dead"),
                                    dead_mask=ev["dead"], near_conf=0.5, p_home_net=ext_w.get("p_home_net"), net_weight=w)
            else:
                lp = dec.match_logp(ev, ext_w, w)
            for sc, md in dec.grid():
                pred = dec.decode(lp, d["half"], sc, md)
                out[dec.param_name(sc, md, w) + suffix] = dfl_bench.score({"half": d["half"]}, gt, pred)[1]
    return m, out


def main():
    paths = dfl_paths.match_paths()
    with ProcessPoolExecutor(int(os.environ.get("A1_PROC", "12"))) as ex:
        grid = pd.DataFrame(dict(ex.map(una, paths))).T
    pool = dfl_paths.for_choice(list(grid.index))
    rows = []
    for m in grid.index:
        p = [n for n in pool if n != m] or pool
        means = grid.loc[p].mean(axis=0)
        best = sorted(means.index, key=lambda n: (-means[n], n))[0]
        rows.append((m.split("-")[0], m, best, grid.loc[m, best]))
    df = pd.DataFrame(rows, columns=["dataset", "partita", "parametri", "spell_f1"])
    print("pesi della rete:", dec.NET_WEIGHTS, "| scelta su:", os.environ.get("DFL_SCELTA") or "tutte",
          "| rete:", os.environ.get("A1_NET_FILE") or "C1", "| solo rete:", os.environ.get("A1_SOLO_NET") == "1")
    print(df.groupby("dataset").spell_f1.agg(["size", "mean", "min", "max"]).round(3).to_string())
    print("\nparametri scelti:", df.parametri.value_counts().head(5).to_dict())
    tag = os.environ.get("A1_TAG")
    if tag:
        grid.to_csv(os.path.join(dfl_paths.out_dir(), "..", "a1_griglia_{}.csv".format(tag)))


if __name__ == "__main__":
    main()
