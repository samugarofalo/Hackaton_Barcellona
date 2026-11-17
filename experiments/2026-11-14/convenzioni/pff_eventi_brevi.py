"""PFF: i possessi brevi delle nostre etichette sono controllo vero o tocchi senza controllo?

Le etichette PFF le ricava converti_pff.py: il possesso passa alla squadra di OGNI evento di gioco
OTB ("a possession with a player on the ball"). La specifica PFF v2.5 dice pero' che un evento OTB
puo' essere fatto solo di RE (rebound: "a touch where a player has no intention to control or
distribute": parate, blocchi, tocchi involontari), CL (clearance: palla allontanata senza cercare
un compagno), CH (contrasto/duello) o TC (tocco fallito): per CHALLENGE.md non sono cambi di possesso.
Qui si ricostruiscono i possessi come fa il convertitore, in tempo continuo, e per ogni possesso si
guarda di che eventi e' fatto. In piu': il campo `sequence` di PFF ("an uninterrupted possession by
one team") come etichetta alternativa del fornitore.
Stampa solo aggregati sulle 64 partite.

    .venv/bin/python experiments/2026-11-14/convenzioni/pff_eventi_brevi.py
"""
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

B = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(B, "data", "external", "pff-wc2022", "Event_Data")
MIN_DEAD_S = 0.5                          # come converti_pff.py
CTRL = {"PA", "CR", "SH", "BC", "IT"}     # azione voluta o primo controllo riuscito
BIN = [0, 1, 2, 3, 5, 10, 1e9]
BN = ["<1", "1-2", "2-3", "3-5", "5-10", ">10"]


def controllo(pe, pid):
    """Regola a* (concordata con Codex il 14/11): l'evento di possesso `pe` del giocatore `pid` mostra
    controllo della palla? Azione voluta o respinta; tocco pesante (TC/H: "the player controls the
    ball, but the touch is a bit too heavy") o cessione voluta al compagno (TC/T); contrasto in cui il
    giocatore e' il portatore ("the player in control of the ball") o il vincitore che tiene la palla
    (esito R: "winner of the challenge retains possession"). Rimpalli (RE), tocchi falliti e duelli
    senza controllo no."""
    t = pe.get("possessionEventType")
    if t in CTRL or t == "CL":
        return True
    if t == "TC":
        return pe.get("touchType") in ("H", "T")
    if t == "CH" and pid is not None:
        return (pe.get("carrierPlayerId") == pid or pe.get("dribblerPlayerId") == pid
                or (pe.get("challengeOutcomeType") == "R" and pe.get("challengeWinnerPlayerId") == pid))
    return False


def game_events(ev, h):
    """Eventi di gioco del tempo h, in ordine: [inizio, fine, tipo, casa?, calcio piazzato, tipi,
    sequence, tocchi, controllo secondo la regola a*]."""
    rows = [x for x in ev if (x.get("gameEvents") or {}).get("period") == h and x["startTime"] is not None]
    rows.sort(key=lambda x: x["startTime"])
    out, seen = [], {}
    for x in rows:
        g = x["gameEvents"]
        k = x["gameEventId"]
        if k not in seen:
            seen[k] = len(out)
            e = x["endTime"] if x["endTime"] is not None else x["startTime"]
            out.append([x["startTime"], e, g.get("gameEventType"), g.get("homeTeam"), g.get("setpieceType"),
                        set(), x.get("sequence"), g.get("touches") or 0, False])
        pe = x.get("possessionEvents") or {}
        t = pe.get("possessionEventType")
        if t:
            out[seen[k]][5].add(t)
            out[seen[k]][8] = out[seen[k]][8] or controllo(pe, g.get("playerId"))
    return out


def spells(G, flips):
    """Possessi (inizio, fine, squadra, eventi dentro) con la regola del convertitore; flips(g) dice se
    l'evento OTB g fa passare il possesso alla sua squadra. Ritorna anche i secondi di gioco vivo."""
    cuts = []                                   # (tempo, 'D' | 'L' | squadra, evento)
    prev_end = None
    for g in G:
        s, e, typ, home, sp = g[:5]
        if sp not in (None, "O") and prev_end is not None and s - prev_end >= MIN_DEAD_S:
            cuts.append((prev_end, "D", None))
            cuts.append((s, "L", None))
        if typ == "OTB" and home is not None:
            cuts.append((s, "H" if home else "A", g))
        if typ in ("OTB", "OUT"):
            prev_end = max(prev_end or e, e)
    cuts.sort(key=lambda c: (c[0], {"D": 0, "L": 1}.get(c[1], 2)))
    t_end = max(g[1] for g in G)
    out, cur, dead, vivo, t_prev = [], None, False, 0.0, None
    for t, k, g in cuts + [(t_end, "END", None)]:
        if t_prev is not None and not dead and cur is not None:
            vivo += max(t - t_prev, 0)
        if k == "D":
            if cur is not None:
                cur[1] = t
                out.append(cur)
            cur, dead = None, True
        elif k == "L":
            dead = False
        elif k == "END":
            if cur is not None:
                cur[1] = t
                out.append(cur)
        else:
            if dead:                             # evento a gioco fermo: il convertitore lo azzera
                pass
            elif cur is None:
                cur = [t, t, k, [g], "D"]
            elif k != cur[2]:
                if flips(g):
                    cur[1] = t
                    out.append(cur)
                    cur = [t, t, k, [g], cur[2]]
                # altrimenti il possesso resta: tocco senza controllo
            else:
                cur[3].append(g)
        t_prev = t
    return out, vivo


def classe(sp):
    """Di che cosa e' fatto un possesso."""
    ev = sp[3]
    ctrl = [g for g in ev if g[5] & CTRL]
    if not ctrl:
        return "solo RE/CL/CH/TC"
    if len(ev) == 1:
        return "1 giocatore, azione voluta"
    return "2+ eventi di gioco"


def main():
    righe, tot = [], collections.defaultdict(float)
    primo = collections.Counter()
    seq_misti = seq_tot = 0
    for p in sorted(glob.glob(os.path.join(SRC, "*.json"))):
        ev = json.load(open(p))
        for h in (1, 2):
            G = game_events(ev, h)
            if not G:
                continue
            S, vivo = spells(G, lambda g: True)
            tot["vivo_min"] += vivo / 60
            tot["possessi"] += len(S)
            for i, sp in enumerate(S):
                d = sp[1] - sp[0]
                nxt = S[i + 1][4] if i + 1 < len(S) and abs(S[i + 1][0] - sp[1]) < 1e-6 else "D"
                panino = sp[4] not in ("D", sp[2]) and nxt == sp[2] and i + 1 < len(S) and S[i + 1][2] != sp[2]
                righe.append((d, classe(sp), panino, sp[4] == "D", nxt == "D"))
                if d < 3:
                    primo["+".join(sorted(sp[3][0][5])) or "nessuno"] += 1
            # regole alternative di conversione
            for nome, f in (("regola a* (controllo, vedi controllo())", lambda g: g[8]),
                            ("solo azioni volute", lambda g: bool(g[5] & CTRL)),
                            ("azioni volute, non IT da solo", lambda g: bool(g[5] & (CTRL - {"IT"})))):
                S2, _ = spells(G, f)
                tot[nome] += len(S2)
                tot[nome + " <3s"] += sum((s[1] - s[0]) < 3 for s in S2)
            # il campo sequence del fornitore
            by = collections.defaultdict(set)
            for g in G:
                if g[2] == "OTB" and g[3] is not None and g[6] is not None:
                    by[g[6]].add(g[3])
            seq_tot += len(by)
            seq_misti += sum(len(v) > 1 for v in by.values())
            tot["sequence"] += len(by)
    df = pd.DataFrame(righe, columns=["durata", "classe", "panino", "dopo_fermo", "prima_fermo"])
    df["fascia"] = pd.cut(df.durata, BIN, labels=BN, right=False)
    vm = tot["vivo_min"]
    print("Partite 64, minuti di gioco vivo {:.0f}, possessi {} = {:.2f} al minuto (regola del convertitore)".format(
        vm, len(df), len(df) / vm))
    print("\nPossessi per durata e contenuto (quota di riga)")
    print(pd.crosstab(df.fascia, df.classe, normalize="index").round(3).to_string())
    print("\nNumero di possessi per durata e contenuto")
    print(pd.crosstab(df.fascia, df.classe).to_string())
    c = df[df.durata < 3]
    print("\nPossessi sotto 3 s: {} = {:.1%} di tutti".format(len(c), len(c) / len(df)))
    print("  a panino (A-B-A senza sosta) {:.1%} | subito dopo una sosta {:.1%} | subito prima di una sosta {:.1%}".format(
        c.panino.mean(), c.dopo_fermo.mean(), c.prima_fermo.mean()))
    print("  contenuto, tra quelli a panino:      ", c[c.panino].classe.value_counts(normalize=True).round(3).to_dict())
    print("  contenuto, tra quelli prima di sosta:", c[c.prima_fermo & ~c.dopo_fermo].classe.value_counts(normalize=True).round(3).to_dict())
    print("  contenuto, tra quelli dopo una sosta:", c[c.dopo_fermo].classe.value_counts(normalize=True).round(3).to_dict())
    n = sum(primo.values())
    print("\nPossessi sotto 3 s: tipi dell'evento che li fa nascere (quota)")
    print("  " + ", ".join("{} {:.1%}".format(k, v / n) for k, v in primo.most_common(12)))
    print("\nPossessi al minuto di gioco vivo con regole di conversione diverse")
    print("  convertitore di oggi (ogni evento OTB)        {:.2f}   sotto 3 s {:.1%}".format(len(df) / vm, (df.durata < 3).mean()))
    for nome in ("regola a* (controllo, vedi controllo())", "solo azioni volute", "azioni volute, non IT da solo"):
        print("  {:<45s} {:.2f}   sotto 3 s {:.1%}".format(nome, tot[nome] / vm, tot[nome + " <3s"] / tot[nome]))
    print("  campo `sequence` di PFF (sequenze distinte)   {:.2f}   sequenze con eventi OTB di entrambe le squadre {:.1%}".format(
        tot["sequence"] / vm, seq_misti / max(seq_tot, 1)))


if __name__ == "__main__":
    main()
