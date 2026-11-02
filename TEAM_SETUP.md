# Setup del team

Questo documento spiega come preparare il progetto sul tuo PC. Leggilo tutto prima di iniziare.

## Regole fondamentali

1. **I dati della partita non entrano mai nella repo.** Non committarli, non copiarli in chat, non mandarli a nessuno fuori dai canali ufficiali.
2. **Nessun assistente AI online riceve dati, estratti o output sui dati.** Vale per ChatGPT, Claude, Gemini, Copilot e simili. Puoi chiedere aiuto per il codice, ma non incollare righe di tracking, omografie, tag, tabelle, log con valori o grafici della partita.
3. **Le etichette si generano solo con algoritmi.** Nessuna etichetta fatta a mano sulla partita, nemmeno per scegliere parametri.
4. **Il file finale deve essere riproducibile.** Chi clona la repo e ha i dati deve ottenere lo stesso CSV, byte per byte.

## Prerequisiti

- Git installato.
- Python 3.9 o superiore (il progetto richiede 3.8+, ma per i notebook serve 3.9+).
- Accesso alla repo privata del team: chiedi a chi gestisce il progetto di aggiungerti da **Settings → Collaborators**.
- Iscrizione al challenge con il nome del team identico a quello degli altri, e con l'indirizzo Gmail che userai per Google Drive. Senza questo non ricevi l'accesso ai dati.

## 1. Clona la repo

Non usare **Fork**: la copia di una repo privata non può restare privata. Clona direttamente:

```powershell
git clone https://github.com/samugarofalo/Hackaton_Barcellona.git
cd Hackaton_Barcellona
git status
```

`git status` deve dire `On branch main`. Il clone ti collega solo alla repo del team (`origin`). Non serve nessun altro remoto.

## 2. Imposta la tua identità git

Serve una volta sola per tutti i tuoi progetti:

```powershell
git config --global user.name "TUO NOME"
git config --global user.email "TUA EMAIL"
```

## 3. Crea l'ambiente Python

Dalla cartella della repo:

```powershell
python -m venv .venv
```

Per attivarlo, PowerShell deve permettere gli script. Lo imposti una volta sola per il tuo utente:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se non vuoi cambiare la policy, puoi usare direttamente l'interprete:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

La cartella `.venv` non deve comparire in `git status`. Se compare, controlla che il `.gitignore` la escluda.

## 4. Verifica che tutto funzioni (senza dati)

```powershell
python evaluation/run_evaluation.py
```

Deve mostrare la tabella con i quattro baseline, con `baseline_smooth125` in testa. Questo controllo usa solo file pubblici della repo, quindi funziona anche senza dati.

### Se pandas non si carica (Smart App Control)

Su alcuni PC Windows, pandas fallisce con `DLL load failed` o `Un criterio di controllo dell'applicazione ha bloccato il file`. Succede quando Smart App Control è attivo, perché blocca le DLL non firmate di pandas. Reinstallare pandas non risolve. numpy invece funziona.

In quel caso usa lo script `pipeline/local_eval.py`, che legge i CSV senza pandas e usa lo scorer ufficiale così com'è:

```powershell
.venv\Scripts\python.exe pipeline\local_eval.py                  # tutti i CSV in evaluation/results
.venv\Scripts\python.exe pipeline\local_eval.py mio_file.csv     # solo questi file
```

Il risultato deve coincidere con la tabella pubblicata nel README degli organizzatori. Se coincide, la pipeline è corretta. Il controllo finale del file resta `evaluation/validate_submission.py` e `evaluation/run_evaluation.py`, da eseguire su una macchina dove pandas funziona (WSL o il PC di un collega). Lo script serve solo per lavorare ogni giorno, non per la consegna.

## 5. Scarica i dati della partita

1. Scarica il bundle dal link che ricevi dopo la registrazione.
2. Mettilo **fuori dalla repo**, in una cartella dedicata, per esempio `C:\mtah2027-data`. La struttura deve essere:

```
C:\mtah2027-data\
├─ tactical\
│  ├─ BAR-ATH_tactical.mp4
│  ├─ BAR-ATH_tactical_atd.xml
│  ├─ BAR-ATH_tactical_atd.txt
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml
└─ broadcast\
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

3. Indica al progetto dove sono i dati, con una variabile d'ambiente permanente:

```powershell
[Environment]::SetEnvironmentVariable("MTAH_DATA_DIR", "C:\mtah2027-data", "User")
```

Poi chiudi e riapri il terminale e VS Code. Il codice non va modificato: legge la variabile da solo.

4. Controlla che il percorso sia visibile:

```powershell
echo $env:MTAH_DATA_DIR
```

**Non aprire `C:\mtah2027-data` con assistenti AI e non mettere quella cartella in OneDrive o Drive.**

## 6. Dove scrivere il codice

- **Il tuo codice va in una cartella nuova**, per esempio `pipeline/`.
- **Non modificare** `evaluation/`, `starter/`, `games.py` e i documenti. Così gli aggiornamenti degli organizzatori non creano conflitti e lo scorer resta quello ufficiale.
- I file generati (CSV di submission) vanno in `evaluation/results/` e restano fuori da git, tranne i baseline.

## 7. Lavorare insieme

- **Un branch per attività**, per esempio `feature/palla-morta`. Non lavorare direttamente su `main`.

```powershell
git switch -c feature/nome-attivita
```

- **Prima di ogni commit** controlla cosa stai per committare:

```powershell
git status
git ls-files data
```

`git ls-files data` deve mostrare solo `data/README.md`, se la cartella `data/` esiste ancora nella repo. Se compaiono video, file ATD o CSV della partita, **fermati e non fare push**.

- **Pull request verso `main`**: ogni modifica passa da una PR, così il team la rivede prima di unirla.
- **Aggiorna il tuo branch** con `git pull` prima di iniziare a lavorare.

## 8. Prima di consegnare un file

```powershell
python evaluation/validate_submission.py NOME-TEAM.csv
```

Il file deve passare la validazione. Non consegnare un file che non hai validato: un file sbagliato viene scartato senza avviso.

## Cose da non fare

- Non committare `data/`, video, tracking o omografie.
- Non fare fork pubblico della repo.
- Non dare dati o output sui dati a assistenti AI online.
- Non scrivere etichette a mano sulla partita, nemmeno per prova.
- Non usare i 5 minuti pubblici per scegliere parametri: servono solo per controllare che la pipeline sia corretta.
- Non modificare lo scorer ufficiale in `evaluation/`.

## Problemi comuni

- **`Impossibile caricare il file Activate.ps1`:** esegui `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, oppure usa `.venv\Scripts\python.exe` direttamente.
- **`match data not found`:** `MTAH_DATA_DIR` non è impostata, o hai riaperto il terminale prima di impostarla. Controlla con `echo $env:MTAH_DATA_DIR`.
