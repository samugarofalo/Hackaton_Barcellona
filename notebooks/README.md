# Notebook DFL

I notebook girano sulle 7 partite DFL (Bassek et al. 2025, CC BY 4.0). Sono altre partite, quindi
addestrare e scegliere parametri su di esse è permesso. Nessun notebook in questa cartella legge la
partita BAR-ATH.

La logica sta nei moduli di `pipeline/`. Il notebook li importa, spiega i passaggi e mostra i
risultati. Il CSV finale si rigenera dai moduli, non dal notebook.

| Notebook | Modulo | Cosa produce |
|---|---|---|
| `A1_decoder_segmenti.ipynb` | `pipeline/decoder.py` | `<match>_decoded.npz` |
| `B1_dfl_palla_morta.ipynb` (in arrivo) | `pipeline/deadball.py` | `<match>_deadball.npz` |
| `B2_dfl_tocco_controllo.ipynb` (in arrivo) | `pipeline/touch_control.py` | `<match>_rule.npz` |

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
