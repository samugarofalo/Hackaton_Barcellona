# Notebook DFL

I notebook girano sulle 7 partite DFL (Bassek et al. 2025, CC BY 4.0). Sono altre partite, quindi
addestrare e scegliere parametri su di esse è permesso. Nessun notebook in questa cartella legge la
partita BAR-ATH.

La logica sta nei moduli di `pipeline/`. Il notebook li importa, spiega i passaggi e mostra i
risultati. Il CSV finale si rigenera dai moduli, non dal notebook.

| Notebook | Modulo | Cosa produce |
|---|---|---|
| `A1_decoder_segmenti.ipynb` | `pipeline/decoder.py` | `<match>_decoded.npz` |
| `B1_dfl_palla_morta.ipynb` | `pipeline/deadball.py` | `<match>_deadball.npz` + modello in `models/` |
| `B2_dfl_tocco_controllo.ipynb` | `pipeline/touch_control.py` | `<match>_rule.npz` |

**Ordine consigliato:** B1 → B2 → A1. B2 e A1 leggono da soli gli output dei notebook precedenti se
li trovano in `/kaggle/working/dfl_outputs` (nella stessa sessione Kaggle, o caricati come dataset).
Ognuno gira anche da solo, con stime di ripiego al posto degli output mancanti.

Le istruzioni comuni per Kaggle (codice, partite DFL, `ball_runs.npz`) sono nella sezione A1 qui
sotto e valgono per tutti e tre.

Il formato dei file che i notebook si passano è descritto in `pipeline/dfl_paths.py`.

---

## A1: decoder a segmenti

### Cosa fa

Sostituisce il voto di maggioranza del baseline (±50 frame) con un decoder che sceglie l'intera
sequenza H / A / D di ogni tempo in un colpo solo. Ogni cambio di segmento costa una penalità, e ogni
segmento deve durare almeno un numero minimo di frame. Così uno sfarfallio breve non spezza un
possesso, e un cambio vero viene messo dove l'evidenza cambia, non mezza finestra dopo.

Il notebook confronta su DFL, nelle condizioni realistiche di `dfl_method4.py`:

1. baseline ±50;
2. regola del tocco (metodo 4);
3. decoder;
4. decoder con rifinitura dei confini;
5. decoder con contesto "tocco";
6. decoder con contesto "pressione" (cambio più economico se la squadra entrante è vicina alla palla).

I parametri del decoder (`switch_cost`, `min_dur`) si scelgono su 6 partite e si misurano sulla
settima, a rotazione. Il numero da guardare è lo spellF1 medio.

### Cosa caricare su Kaggle

Tutti i dataset devono essere **privati**.

1. **Il codice della repo.** Due modi, a scelta:
   - **Dataset:** zip della repo (senza `data/`), caricato come dataset Kaggle con nome
     `hackaton-barcellona-code`. Il notebook lo trova in `/kaggle/input/hackaton-barcellona-code`.
   - **Clone:** aggiungere un token GitHub nei Kaggle Secrets (nome `GITHUB_TOKEN`) e in una cella:
     ```python
     from kaggle_secrets import UserSecretsClient
     tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
     !git clone -b feature/notebooks-dfl https://{tok}@github.com/samugarofalo/Hackaton_Barcellona.git /kaggle/working/Hackaton_Barcellona
     ```
   In alternativa, impostare `MTAH_REPO` alla cartella della repo.
2. **Le partite DFL convertite:** i file `DFL-MAT-*.npz` prodotti da `pipeline/convert_dfl.py`. Il
   notebook li cerca da solo sotto `/kaggle/input`. Se ci sono solo gli XML originali, va prima
   eseguito `convert_dfl.py` (oppure impostare `DFL_CONVERTED` alla cartella con gli `.npz`).
3. **Facoltativo ma consigliato: `data/cleanup/ball_runs.npz`.** Contiene le lunghezze aggregate dei
   tratti in cui il nostro rilevatore vede o perde la palla, e serve a simulare le condizioni reali.
   Senza questo file il notebook usa lunghezze indicative e stampa `placeholder`: i numeri restano
   confrontabili tra metodi, ma non con quelli di `dfl_method4.py`. È un aggregato ricavato da
   BAR-ATH, quindi va solo in un dataset privato. In alternativa: variabile `DFL_BALL_RUNS`.
4. **Facoltativo: gli output di B1 e B2** (`<match>_deadball.npz`, `<match>_rule.npz`). Se sono in
   `/kaggle/working/dfl_outputs`, il decoder li usa al posto delle stime di ripiego.

### Come eseguirlo

1. Aprire `notebooks/A1_decoder_segmenti.ipynb` su Kaggle (File → Import notebook, o dalla repo).
2. Aggiungere i dataset dei punti 1–3 con "Add Input".
3. Eseguire tutte le celle in ordine (Run All). Servono pochi minuti, senza GPU.
4. Controllare l'output della prima cella dopo il setup: deve dire quante partite ha trovato (7) e
   se `ball_runs` è quello vero o `placeholder`.
5. Leggere la tabella della sezione 4: una riga per metodo, con media, minimo e massimo sulle 7
   partite. Differenze sotto circa 0,04 di spellF1 sono rumore.
6. Scaricare da `/kaggle/working/dfl_outputs` i file `<match>_decoded.npz`, se servono ad altri
   notebook.

In locale funziona allo stesso modo: `python pipeline/decoder.py` stampa le stesse tabelle, con i
dati in `data/external/bassek2025/converted/`.

### Cosa riportare al team

- spellF1 medio di ogni metodo e i parametri scelti nei fold (la cella che stampa
  `parametri scelti nei fold`);
- se il contesto "pressione" batte il costo costante;
- se `ball_runs` era quello vero o il `placeholder`.

Solo numeri aggregati su DFL. Niente dati o output della partita BAR-ATH in strumenti AI online.

### Dipendenze

numpy, pandas e matplotlib (già presenti nell'immagine Kaggle).

### Test

`python pipeline/test_decoder.py` esegue 9 test su dati sintetici, tra cui il confronto con una
ricerca esaustiva che verifica che il decoder trovi davvero l'ottimo.

---

## B1: rilevatore di palla morta

### Cosa fa

Il baseline scrive `D` dove la palla non è tracciata. Sul nostro tracking la palla manca nel 41% dei
frame valutati, quasi sempre a gioco fermo ma non sempre. B1 impara a riconoscere il gioco fermo
dal comportamento dei giocatori, come Lang et al. 2022: 240 feature per frame (velocità collettive,
forma delle squadre, schieramenti tipici delle riprese, palla fuori campo o ferma, durata
dell'assenza della palla), anche mediate su finestre da 0,4 a 80 s nel passato e nel futuro.

Il modello è LightGBM (con ripiego su scikit-learn), addestrato su DFL degradato e valutato
lasciando fuori una partita alla volta. Filtro mediano e soglia si scelgono sulle altre 6 partite.
Il notebook confronta quattro modi di decidere la `D`: palla mancante, rilevatore, la loro
combinazione, e la `D` vera come tetto.

### Come eseguirlo

1. Stessi input di A1: codice e partite DFL. `ball_runs.npz` non serve.
2. Run All. Servono circa 15–20 minuti. Per una prova veloce, da riga di comando:
   `python pipeline/deadball.py --quick`.
3. Uscite: `<match>_deadball.npz` in `/kaggle/working/dfl_outputs`, e il modello finale in
   `/kaggle/working/models` (`deadball_lgbm.txt` + `deadball_meta.json`). Le cartelle si cambiano con
   `DFL_OUT` e `DEADBALL_MODELS`.
4. **Scaricare la cartella `models`:** è il modello da applicare poi a BAR-ATH, in locale, con
   `deadball.predict_p_dead(...)`.

### Cosa riportare al team

- spellF1 a valle per le quattro varianti di `D` e l'F1 sugli stop di almeno 2 s;
- le 20 feature più importanti;
- finestra del filtro mediano e soglia scelte.

Test: `python pipeline/test_deadball.py`. Dipendenze: numpy, pandas, scipy, lightgbm (o
scikit-learn), matplotlib; su Kaggle ci sono già.

---

## B2: regola tocco + conferma del controllo

### Cosa fa

La challenge vuole che il possesso cambi solo con il **controllo** dell'altra squadra, non a ogni
tocco. B2 prende i tocchi del metodo 4 e accetta un cambio solo se, entro T1 secondi, arriva una
conferma: un secondo tocco della stessa squadra, una conduzione (un suo giocatore entro R metri
dalla palla per N frame) o la palla che rallenta vicino a un suo giocatore. Il cambio confermato è
datato al primo tocco, perché un secondo di ritardo costa circa 0,35 di spellF1. Deviazioni e
rimpalli non confermati lasciano il possesso all'ultima squadra.

La griglia (72 combinazioni) usa le soglie della letteratura: zona di controllo 1–1,5 m
(Vidal-Codina 2022, Borghesi 2023), soglia del tocco come in Link & Hoernig 2017. Tutti i parametri,
anche quelli del metodo 4 di confronto, si scelgono su 6 partite e si misurano sulla settima.

### Come eseguirlo

1. Stessi input di A1, compreso `ball_runs.npz` (senza, usa durate sintetiche e lo segnala: i numeri
   non sono confrontabili con lo 0,543 del metodo 4).
2. Se B1 ha già girato, la palla morta viene da B1 (`--dead auto` da riga di comando).
3. Run All. Con la griglia completa servono alcuni minuti per seme; `QUICK = True` usa una griglia
   ridotta.
4. Uscite: `<match>_rule.npz` (`labels`, `p_home`) in `/kaggle/working/dfl_outputs`, usati da A1.

### Cosa riportare al team

- spellF1 e macroF1 di baseline, metodo 4 e tocco + controllo;
- possessi predetti contro veri per partita (troppi = sfarfallio, troppo pochi = cambi persi);
- cambi persi, cambi falsi e ritardo mediano dei confini.

Test: `python pipeline/test_touch_control.py`. Dipendenze: numpy, pandas, matplotlib.
