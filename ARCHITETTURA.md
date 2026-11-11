# Architettura e piano (dal 12/11)

Questo file descrive la catena che seguiamo da ora fino alla consegna del 17/11: cosa usa, quali
pezzi ha, cosa abbiamo scartato e come si esegue. Sostituisce la "catena proposta" del §0 di
`CONTESTO.md`, che resta il diario delle prove fatte.

## In una frase

Si usa solo il tracking fornito dalla gara. Tre pezzi producono ciascuno un indizio (gioco fermo,
chi ha toccato la palla, chi ha il possesso secondo i giocatori) e un decoder li unisce in
un'unica sequenza di possessi.

## Perché cambiamo strada

La catena proposta l'11/11 scriveva "palla ferma" dove il tracking fornito non vede la palla.
`DATA.md` dice però che la palla si perde anche durante il gioco: mischie, area affollata, palla
in aria. Tra la nostra documentazione e quella della gara vince quella della gara, quindi abbiamo
controllato.

Il controllo usa la finestra pubblica di 5 minuti, solo per la correttezza e con soli numeri
totali (7.500 fotogrammi, 4 interruzioni):

| Domanda | Risposta |
|---|---|
| Quanta palla ferma vera riconosce "manca la palla"? | il 34% |
| Quando manca la palla, il gioco è davvero fermo? | nel 56% dei fotogrammi |
| Vuoto più lungo a gioco vivo | circa 9 secondi |
| Interruzioni in cui la palla resta visibile | 1 su 4 |
| Bordi | la palla sparisce circa 1,2 s dopo l'inizio dell'interruzione |

Sono 5 minuti: è un indizio, non una misura, e non lo usiamo per tarare nulla. Basta però a dire
che "manca la palla" non può essere la regola per la palla ferma. Diventa uno degli indizi.

Seconda conseguenza: la nostra palla (rilevatore, proiezione, due camere) non entra più nella
catena. La prova equa di `CONTESTO.md` §14 la dà pari a quella fornita, e con la palla fornita la
catena è più semplice, deterministica e facile da rigenerare.

## La catena

```
Tracking fornito (giocatori, squadre, palla)
        │
   make_submission.load_atd  → stesso formato delle partite DFL
        │
   ┌────┼────────────────┬───────────────────┐
   B1                    B2                  C1
 modello              regola              rete neurale
 "gioco fermo?"    "tocco confermato"   "chi ha la palla,
                                         dai movimenti"
   └────┬────────────────┴───────────────────┘
        │
   A1 decoder → sequenza H / A / D
        │
   CSV: X fuori dai tempi, D prima del calcio d'inizio del 2° tempo
```

| Pezzo | File | Tipo | Cosa risponde | Perché serve |
|---|---|---|---|---|
| B1 | `pipeline/deadball.py` | Modello LightGBM | Il gioco è fermo? | "Manca la palla" non basta a dirlo |
| B2 | `pipeline/touch_control.py` | Regola | Chi ha toccato e poi controllato la palla? | È la definizione di possesso della gara |
| C1 | `pipeline/player_net.py` | Rete neurale (PyTorch) | Chi ha il possesso, guardando i giocatori? | Dà un indizio quando la palla sparisce a gioco vivo |
| A1 | `pipeline/decoder.py` | Decoder a segmenti | Qual è la sequenza più sensata? | Toglie lo sfarfallio, che lo spellF1 punisce |

I modelli si addestrano sulle 7 partite DFL, lasciando fuori una partita a rotazione. Sulla nostra
partita si applicano e basta: nessuna etichetta, nessuna taratura.

## Una sola simulazione, tre livelli

Prima ogni modulo simulava "il nostro tracking su DFL" a modo suo, e tutti davano per buono che la
palla manchi solo a gioco fermo. Ora c'è una sola simulazione, `pipeline/dfl_sim.py`, usata da B1,
B2, C1 e A1.

Quanto è difettoso il tracking fornito non si può misurare sulla nostra partita senza etichette, e
la finestra pubblica non si usa per tarare. Per questo **non si stima un valore: se ne provano
tre**, con numeri tondi che vanno dal caso facile a quello difficile.

| | easy | medium | hard |
|---|---|---|---|
| Interruzioni in cui la palla sparisce | 90% | 75% | 60% |
| Ritardo massimo dell'inizio del vuoto | 0 s | 1 s | 2 s |
| Gioco vivo senza palla | 0% | 4% | 8% |
| Durata di quei vuoti | – | 0,2–5 s | 0,2–10 s |
| Rumore sui giocatori | 0,02 m | 0,10 m | 0,20 m |
| Giocatori mancanti | 0% | 5% | 10% |
| Vita media di una traccia prima di cambiare slot | – | 30 s | 10 s |

`easy` è vicino a quello che i moduli assumevano prima. Il livello si sceglie con la variabile
`DFL_SIM` (default `medium`).

**Regola di decisione.** Un pezzo o una variante entra nel CSV finale solo se migliora su DFL in
tutti e tre i livelli. Vale anche per C1: A1 sceglie il peso della rete tra 0, 0,5 e 1 insieme
agli altri parametri, e se sceglie 0 la rete resta fuori. I modelli applicati alla nostra partita
sono quelli del livello `medium`. La finestra pubblica non decide mai niente.

## Cosa teniamo e cosa scartiamo

| Teniamo | Scartiamo dalla catena |
|---|---|
| Tracking fornito come unico ingresso | Il nostro rilevatore di palla e `ball_runs.npz` |
| B1, B2, C1, A1 | Metodo 4 (nostri giocatori fusi con quelli forniti) e portieri |
| Addestramento e scelte solo su DFL | Proiezione, 3D da una camera, triangolazione col broadcast |
| Finestra pubblica solo per la correttezza | Traiettorie dritte, tocco di traverso, contatto nell'immagine |
| D fisso prima del calcio d'inizio del 2° tempo | Modelli di spotting già pronti (T-DEED, AdaSpot) |
| | La regola "manca la palla = gioco fermo" |
| | Smart Tagging come ingresso: su DFL non esiste e non si può verificare. Si usa solo nei controlli finali |

Il codice scartato resta nella repo e va raccontato nel documento tecnico: è il motivo per cui la
strada semplice è una scelta e non una scorciatoia.

## Cosa è cambiato nel codice

| File | Modifica |
|---|---|
| `pipeline/dfl_sim.py` (nuovo) | La simulazione unica, con i tre livelli |
| `pipeline/deadball.py` | Usa la simulazione unica |
| `pipeline/touch_control.py` | Usa la simulazione unica e la sola palla fornita; salva `rule_params.json` |
| `pipeline/player_net.py` | Usa la simulazione unica: una sola palla, tracce che cambiano slot, giocatori mancanti. Predice su tutti i fotogrammi |
| `pipeline/decoder.py` | Usa la simulazione unica; con i file di B1, B2, C1 valuta solo sul seme su cui sono stati calcolati; sceglie il peso della rete; salva `decoder_params.json` |
| `pipeline/make_submission.py` (nuovo) | Legge il tracking fornito, applica la catena, scrive il CSV |
| `pipeline/dfl_paths.py` | Cartella unica dei modelli (`MTAH_MODELS`) |
| `notebooks/00_esegui_tutti.ipynb` | Esegue B1 → B2 → C1 → A1 per ogni livello e mostra la tabella per livello |
| Notebook singoli e `notebooks/README.md` | Testi e chiamate allineati; niente più `ball_runs` |

Tre difetti corretti nel passaggio:

1. **Semi.** A1 riusava per il seme 1 i file che B1, B2 e C1 avevano calcolato sul seme 0: in quei
   file la palla era stata vista dove il seme 1 ha un vuoto, e il punteggio saliva per questo.
2. **Tracce.** C1 si allenava con giocatori sempre presenti e sempre nello stesso posto dell'array.
   Sulla nostra partita ci sono 46 slot per 22 giocatori, che nascono, muoiono e si scambiano.
3. **Peso della rete.** Era fisso a 1 e si sommava a indizi che usano la stessa palla. Ora è scelto
   sui dati, e può essere 0.

## Come si esegue

1. **Su Kaggle**, `notebooks/00_esegui_tutti.ipynb` con Accelerator = GPU T4: addestra e misura B1,
   B2, C1 e A1 sui tre livelli. Con `QUICK = True` è una prova di pochi minuti.
2. **Scaricare** la cartella `models/medium` e metterne il contenuto in `models/`:
   `deadball_lgbm.txt`, `deadball_meta.json`, `rule_params.json`, `decoder_params.json`,
   `player_net.pt`, `player_net.json`.
3. **In locale**: `python pipeline/make_submission.py --out evaluation/results/<squadra>.csv`
   (`--no-net` per lasciare fuori C1). Stampa solo totali.
4. **Controlli**: `python evaluation/validate_submission.py` sul file esatto; finestra pubblica una
   volta sola, per la correttezza; plausibilità dei totali rispetto a DFL (circa 4,6 possessi al
   minuto di partita, durata tipica 3,7 s, palla ferma intorno al 45%).
5. **Rigenerazione**: rifare i passi 3 e 4 da zero e verificare che il file sia identico.

## Regole che non cambiano

- Niente etichette a mano sulla partita fornita, per nessuno scopo.
- Video, dati ed estratti della partita non vanno in strumenti AI online: `make_submission.py` e i
  controlli girano in locale e stampano solo totali.
- I 5 minuti pubblici servono solo a controllare che la catena sia corretta.
- Differenze sotto circa 0,04 di spellF1 sono rumore.

## Tempi

| Giorno | Lavoro |
|---|---|
| 12/11 | Run completa su Kaggle sui tre livelli |
| 13/11 | Lettura dei risultati, decisione su C1 e sulle varianti |
| 14/11 | CSV, controllo di correttezza e di plausibilità |
| 15–16/11 | Documento tecnico, rigenerazione da zero |
| 17/11 | Consegna |

## Rischi e punti aperti

- **C1 è il pezzo più incerto**: è il più lento e quello che dipende di più da quanto la
  simulazione somiglia alla partita vera. La catena funziona anche senza (`--no-net`).
- **B1 sulla nostra partita vale meno che su DFL.** Su DFL "sente" la palla ferma dalla velocità
  della palla; qui la palla spesso non c'è. I livelli `medium` e `hard` servono a non illudersi.
- **I tre livelli sono numeri tondi, non misure.** Reggere su tutti e tre riduce il rischio, non lo
  toglie.
- **I numeri vecchi non sono confrontabili con i nuovi**: 0,627 per A1 e le tabelle di `CONTESTO.md`
  §15 vengono da simulazioni più facili.
- **Il "2 vuoti a gioco vivo in tutta la partita"** di `CONTESTO.md` §6 non è compatibile con il
  controllo sopra: va rivisto.
