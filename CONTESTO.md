# Contesto di lavoro: idee, scoperte, decisioni

Note interne del progetto per la gara *More than a Hack 2027* (Metrica Sports). Aggiornato al
10/11/2026. Consegna: **17/11/2026**.

Il file non contiene dati della partita: solo numeri aggregati e riferimenti agli script.
Gli output stanno in `data/` (ignorato da git).

---

## 1. Il compito e le regole per assegnare le etichette

Le fonti sono `CHALLENGE.md`, `FAQ.md`, `DATA.md`, `EVALUATION.md` e `RULES.md`. Se il nostro
riassunto e il repository non coincidono, ha ragione il repository.

### Il file

- Una riga per **ogni** fotogramma del video tattico: 199.319 righe, `frame,label`, con `frame`
  che parte da 0.
- Il fotogramma del video è `frame_id − 1` dell'ATD.
- Prima di caricare il file va eseguito `evaluation/validate_submission.py` sul file esatto da
  consegnare.
- Il file si chiama `<nome_squadra>.csv` e il documento tecnico `<nome_squadra>.pdf`.
- Si consegna con un Google Form, insieme al repository pubblico.

### Le quattro etichette

| etichetta | significato |
|---|---|
| `H` | possesso della squadra di **casa** (Barcellona), palla viva |
| `A` | possesso della squadra **ospite** (Athletic), palla viva |
| `D` | **palla ferma**, fuori gioco |
| `X` | nessuna risposta |

### Quali fotogrammi contano

- Si valutano solo i due tempi, nei fotogrammi del video:
  - **11.185–90.651**;
  - **110.808–190.197**.
- Tutto il resto non viene valutato: prima della partita, intervallo, dopo il fischio finale. Lì
  scriviamo `X`.
- **Trappola:** il secondo tempo è valutato da 110.808, ma il calcio d'inizio è al fotogramma
  **113.742**. I 2.934 fotogrammi in mezzo sono valutati e sono **tutti `D`**.

### Regole del possesso, come le chiede la challenge

1. Mentre la palla è viva, **una squadra ha sempre il possesso**.
2. Il possesso cambia solo quando un giocatore avversario ottiene il **possesso controllato** della
   palla.
3. Palla **contesa o vagante**: il possesso **resta all'ultima squadra** che lo aveva.
4. **Non** cambiano il possesso:
   - un semplice tocco senza controllo;
   - una deviazione di un passaggio che non lo intercetta;
   - un duello aereo vinto senza che quella squadra poi controlli la palla.
5. Esempi di "controllo" suggeriti dalla challenge, non come definizione: un giocatore tiene la
   palla per qualche secondo; una squadra fa più di due passaggi di fila.
6. Dopo ogni interruzione c'è un possessore chiaro: la difficoltà è capire quando il possesso
   cambia dopo.
7. **Palla ferma = `D`**, sempre. Il proprietario della palla ferma non è richiesto: chi batte la
   rimessa o chi l'ha toccata per ultimo non conta.
8. **Un possesso è una sequenza massima di una squadra con la palla viva.** `H H H A A` sono due
   possessi. `H H D D H H` sono due possessi separati da un'interruzione, non uno.
9. Le etichette possibili nella verità sono solo `H`, `A` e `D`: non esistono "conteso",
   "transizione" o "ignoto".
10. La verità **è derivata in modo meccanico dal tracking corretto** di Metrica, senza levigature
    né aggiustamenti a mano:
    - `D` dove il tracking corretto segna la palla ferma;
    - altrimenti la squadra che il tracking indica come proprietaria.

### Come conviene usare `X`

- Lo spellF1, la metrica della classifica, **non vede** `X`: lo sostituisce con l'ultima etichetta
  data in quel tempo.
- Il macroF1, primo criterio di spareggio, conta `X` come errore.
- Quindi `X` in mezzo a un possesso è gratis sulla classifica e più sicuro di un'ipotesi che lo
  spezza in due.
- `X` a cavallo di un passaggio di possesso costa caro, perché si trascina dietro l'etichetta
  vecchia. Se il dubbio è su una transizione, meglio tirare a indovinare.

### Valutazione

- La classifica è sullo **spellF1** della partita intera, la cui verità resta privata.
- Il **macroF1** serve da spareggio.
- I 5 minuti pubblici (fotogrammi 11.185–18.684, 17 possessi) servono **solo a controllare che la
  catena sia corretta**, mai a scegliere tra due versioni.
- La banda di rumore dello spellF1 è circa **0,04**.

---

## 2. Cosa è permesso: i limiti che ci siamo dati

- **Niente etichette a mano sulla partita fornita**, in nessuna quantità e per nessuno scopo:
  né per la consegna, né per addestrare, né per validare.
- I dati annotati di **altre partite** si possono usare. Noi usiamo:
  - DFL Bassek 2025, licenza CC BY 4.0;
  - SoccerNet-v3 H250, licenza CC BY 4.0;
  - SoccerNet-Tracking 2023, pubblico su Hugging Face e senza NDA.
- **Guardare il video è permesso** per capire e fare debug. Francesco controlla i video generati in
  locale.
- Quello che vede **non diventa** etichette, correzioni, oppure regole o costanti legate a
  fotogrammi specifici.
- Ogni regola nata da un caso visto deve essere generale e va verificata su altri dati.
- **Pseudo-etichette fatte da un algoritmo** sulla partita: sono ammesse. Esempio: il colore delle
  maglie, con le squadre di Metrica come maestro.
- **LLM ospitati:** i Terms vietano di dare a un LLM ospitato video, dati o *estratti* della
  partita. Claude Code è un LLM ospitato.
  - Il 10/11 ci siamo accorti di aver stampato in sessione alcune coordinate per fotogramma durante
    il debug.
  - Da allora si stampano solo aggregati.
  - Francesco decide se segnalarlo agli organizzatori.
- **Riproducibilità:** il repository deve rigenerare **lo stesso file**. Va verificato che il
  rilevatore in FP16 sulla GPU dia lo stesso risultato in due esecuzioni.
- La squadra deve avere da 3 a 5 membri per l'Innovation Challenge e la Final Pitch.

### Casi grigi registrati

- In `carrier_fix.py` la finestra di voto è passata da ±12 a ±3 dopo un caso visto. Oggi non è
  nella catena: il metodo 4 usa i tratti delle tracce.
- Il filtro "palla fuori dal campo oltre 1 m" e il "divieto di salti" sono nati da casi visti:
  - sono regole fisiche generali;
  - il divieto di salti è verificato su SoccerNet;
  - il margine di 1 m è un numero tondo, non tarato.

---

## 3. Dati e strumenti

- **ATD Metrica:**
  - 46 slot, che non sono identità persistenti;
  - coordinate normalizzate;
  - campo di 105,3 × 68 m;
  - la palla manca nel 41% dei fotogrammi valutati, quasi sempre a gioco fermo.
- **Rilevamento notturno** (`pipeline/night_run.py`, in `data/night/`):
  - yolo26x a 1920 in FP16, ByteTrack;
  - istogrammi Lab di maglia e pantaloncini;
  - fatto su tutti i 158.857 fotogrammi tattici e su 93.473 fotogrammi broadcast di gioco vivo.
- **DFL** (7 partite Bundesliga): tracking pulito con portieri segnati, `BallPossession` e
  `BallStatus`. È il banco per le **regole di possesso**.
- **SoccerNet** (H250 + Tracking): palla annotata. È il banco per il **rilevatore di palla**.
- **SoccerTrack v2:**
  - **non ha la palla annotata nei pixel**: è interpolata tra gli eventi;
  - ha 21.428 eventi sulla palla con il giocatore che li compie;
  - è utile per verificare i tocchi, non per addestrare.

---

## 4. Scoperte principali, in ordine

1. **La baseline** (giocatore più vicino più voto su ±50 fotogrammi) fa **0,243** di spellF1 sulla
   nostra partita. Su DFL, con una palla tipo Metrica, fa **0,34**.
2. **Gli errori di tracking contano poco se presi uno alla volta e molto se combinati.** Su DFL:
   - il portatore di palla con la squadra sbagliata nel 5% del gioco costa −0,02;
   - errori combinati realistici costano −0,07 o −0,08.

   Una regola reattiva è **più sensibile** a questi errori.
3. **Sulla nostra partita gli errori di squadra alla palla sono rari:** circa l'1% dei fotogrammi
   con la palla (`carrier_fix.py`). Valgono quindi circa 0,005 per la baseline.
4. **La finestra di 4 s non serve contro il rumore:** compensa una regola sbagliata. Su DFL pulito,
   con la palla ferma presa dalla verità:

   | Regola | spellF1 |
   |---|---|
   | Più vicino, ±12 fotogrammi | 0,28 |
   | Più vicino, ±50 | 0,48 |
   | **Controllo palla** (avversario entro 1 m per 3 fotogrammi) | **0,57** |
   | **Tocco** (brusca variazione della velocità della palla con un giocatore vicino), idea di Francesco | **0,70** |

5. **La palla di Metrica non è rumorosa ma levigata.**
   - Equivale a una palla DFL mediata su 10–15 fotogrammi.
   - Ha la metà dei cambi bruschi.
   - Sta in media 1 m più lontana dai giocatori.

   La regola dei tocchi richiede quindi una **palla nitida**.
6. **Il nostro rilevatore di palla** (yolo26s addestrato su SoccerNet a 640 pixel: la palla di
   SoccerNet ridotta ha la stessa grandezza della nostra a 1920, 6–9 pixel):
   - trova la palla entro 2 m da Metrica nel 63–69% dei fotogrammi, contro il 16% del modello
     generico COCO;
   - la traiettoria è nitida quanto la palla DFL, e scuote di soli 3 cm;
   - su SoccerNet-Tracking test, con il filtro di continuità (salto ≤ 4 grandezze di palla per
     fotogramma, tratti ≥ 8 rilevamenti) e il divieto di salti tra tratti, ha precisione **0,954**
     e copertura 0,677;
   - sulla partita, durante il gioco, c'è nell'86–90% dei fotogrammi in cui Metrica ha la palla;
   - i vuoti sono quasi tutti brevi: mediana 2 fotogrammi, 90% sotto 13.
7. **Le palle fuori dal campo** (oltre 1 m dalle linee) vanno scartate: sono palloni di riserva, e
   lì il gioco è comunque fermo.
8. **I portieri.** Su DFL un giocatore più indietro di ≥ 2 m del compagno più arretrato è il
   portiere nel 99,5% dei casi.
   - Le maglie dei due portieri sono simili, quindi il colore dice solo "portiere" e la metà campo
     dice di quale squadra.
   - Metrica sbaglia sui portieri e a volte inventa un giocatore in più.

---

## 5. Metodi di unione (confronti 2D in `data/cleanup/`)

| # | Giocatori | Palla | Giocatore alla palla diverso da Metrica (3 min) |
|---|---|---|---|
| 1 | Metrica | Metrica | – |
| 2 | Metrica, squadra corretta col colore sui tratti continui, giocatore aggiunto alla palla | nostra, Metrica nei vuoti | 7,9% |
| 3 | solo nostri (tracce con continuità, colore mediato) | nostra | 15,1% |
| **4** | **nostri con continuità; dove il colore è incerto, squadra dai giocatori Metrica vicini (peso gaussiano σ = 1 m, entro 3 m); portieri da `keepers.py`** | nostra | 10,9% |

- Il metodo 4 è quello che Francesco preferisce guardando i video.
- **Palla finale scelta:** nei vuoti brevi la palla si interpola, nei vuoti lunghi si usa quella di
  Metrica. La soglia si sceglie sui dati DFL e SoccerNet (sezione 6).
- I tocchi non si cercano dove la palla è quella di Metrica.

### Portieri (`pipeline/keepers.py`)

1. **Probabilità dalla posizione:** curva logistica della profondità, imparata su DFL.
2. **Colore a 4 classi** (H, A, portiere, arbitro). Il maestro per "portiere" sono i riquadri
   agganciati a giocatori Metrica con probabilità di posizione ≥ 0,99.
3. **Somma delle due evidenze**, mediata sulla traccia del rilevatore, più due vincoli:
   - ogni riquadro deve essere il più arretrato della sua metà campo;
   - al massimo un portiere per metà campo.
4. **Squadra:** quella che difende la metà campo.

Risultato: quando il nostro portiere è agganciato a uno slot Metrica, quello slot è della stessa
squadra nel 94,9% dei casi.

### Altre condizioni possibili per i portieri (da provare, tutte imparabili su DFL)

- **Area di rigore:** la quota di tempo dentro la propria area.
- **Velocità e accelerazioni:** il portiere corre meno e su tratti più corti.
- **Movimento legato alla palla:** il portiere si sposta lateralmente seguendo la palla, restando
  tra la palla e il centro della porta.
- **Unicità e continuità per tutto il tempo:** un solo portiere per squadra per tempo, scambi solo
  con le sostituzioni (rare). Dopo un'occlusione si riaggancia per posizione, visto che si muove
  poco.
- **Cambio di campo all'intervallo:** ogni portiere passa nell'altra metà.
- **Eventi:** rinvii dal fondo e prese, cioè palla ferma nell'area vicino al portiere e poi un
  lancio lungo. Lo Smart Tagging, che è fornito e quindi ammesso, segna fasi utili.
- **Figura:** postura e guanti sono poco visibili a 7–9 pixel, ma il riquadro del portiere ha spesso
  proporzioni diverse.

---

## 6. Valutazione del metodo 4 su DFL (`pipeline/dfl_method4.py`)

### Come è fatta la simulazione

Ingressi costruiti come i nostri, sulle 7 partite DFL, con 2 semi:

- **Palla ferma:** dove manca la palla tipo Metrica.
- **Palla di Metrica:** mediata su 12 fotogrammi.
- **Nostra palla:**
  - la palla vera con un rumore di 0,03 m, quello misurato;
  - presente e assente a tratti, con le durate misurate sulla nostra partita.
- **Giocatori:** rumore di 0,2 m, più l'1% di squadra sbagliata alla palla.
- **Regola:** solo il tocco. Il possesso va alla squadra del giocatore vicino al punto del tocco.

### spellF1 medio

| Palla | Tocco > 2 m/s | Tocco > 3 m/s | Tocco > 5 m/s |
|---|---|---|---|
| Baseline ±50 con palla Metrica | 0,392 | | |
| Solo nostra palla | 0,464 | 0,467 | 0,432 |
| Vuoti ≤ 0,5 s interpolati, Metrica nel resto | 0,467 | 0,465 | 0,417 |
| Vuoti ≤ 1 s interpolati, Metrica nel resto | 0,473 | 0,471 | 0,422 |
| **Vuoti ≤ 2 s interpolati, Metrica nel resto** | **0,479** | 0,476 | 0,427 |
| Tutti i vuoti con Metrica | 0,433 | 0,430 | 0,378 |

### Cosa se ne ricava

- Il metodo 4 con la palla finale fa **+0,087 rispetto alla baseline**, più del doppio della banda
  di rumore.
- **Interpolare batte Metrica** anche su vuoti di 2 s. L'errore della retta, misurato su DFL:

  | Vuoto | Errore mediano | 90° percentile |
  |---|---|---|
  | 2 fotogrammi | 0,02 m | 0,08 m |
  | 13 fotogrammi | 0,14 m | 0,71 m |
  | 50 fotogrammi | 1,5 m | 4,7 m |

- La retta conserva la nitidezza ai bordi del vuoto, mentre la palla di Metrica la perde.
- **Soglia scelta:** 50 fotogrammi (2 s). Il guadagno continua a crescere fino al limite provato,
  quindi si può provare anche oltre.
- Con un rumore di 0,10 m invece di 0,03 m il tocco a 2 m/s scendeva a 0,41: la regola richiede
  una palla poco rumorosa.

### Da 0,74 in giù: cosa costa ogni ostacolo (corretto il 10/11)

Prima versione: −0,105 per la palla ferma. Quasi tutto veniva da un'**ipotesi sbagliata della
simulazione**: il 2% del gioco vivo con la palla di Metrica assente.

- Sulla nostra partita i vuoti di Metrica a gioco vivo sono **solo 2 in tutta la partita**.
- Senza quei buchi la palla ferma costa −0,027.
- Se Metrica perde la palla in **tutte** le interruzioni oltre 1 s, il costo è 0.
- Le interruzioni DFL di durata ≤ 1 s sono lo 0% (2 su 711).

Anche lo 0,479 della tabella sopra va letto con la stessa ipotesi pessimista.

Catena rifatta senza buchi a gioco vivo (tocco > 2 m/s, 7 partite, 1 seme):

| Passo | spellF1 | Costo del passo |
|---|---|---|
| Baseline ±50, palla Metrica (riferimento) | 0,421 | |
| 1. Palla vera, palla ferma presa dalla verità (tetto) | 0,743 | – |
| 2. Palla ferma = palla di Metrica assente | 0,716 | −0,027 |
| 3. Rumore della palla 0,03 m | 0,700 | −0,016 |
| 4. **Vuoti della nostra palla** (≤ 2 s interpolati, Metrica nel resto) | 0,593 | **−0,107** |
| 5. **Rumore dei giocatori 0,2 m**, indipendente per fotogramma | 0,552 | **−0,041** |
| 6. 1% di squadra sbagliata alla palla | 0,543 | −0,009 |

**Metodo 4 contro baseline: 0,543 contro 0,421 (+0,12).**

Cosa resta incerto sulla palla ferma:

- La quota di interruzioni in cui Metrica tiene la palla. Nella simulazione è il 10%, una stima
  non misurabile.
- Il ritardo tra il fischio e la scomparsa della palla. Nella simulazione è perfetto.

Segnali per riconoscere la palla ferma, da imparare su DFL (`BallStatus`):

- la palla esce dalle linee;
- tutti i giocatori rallentano insieme;
- la palla resta ferma in un punto tipico di ripresa: dischetto, angolo, fondo, centro;
- lo Smart Tagging fornito segna calci piazzati e rimesse.

**Priorità che ne seguono:**

1. **Vuoti della nostra palla** (−0,107).
   - I vuoti veri non sono casuali come nella simulazione: capitano quando un giocatore copre la
     palla, cioè proprio durante i tocchi.
   - Rimedi possibili:
     - una seconda ricerca guidata dalla traiettoria, a soglia più bassa;
     - a una risoluzione maggiore intorno alla posizione prevista;
     - il broadcast;
     - il tocco dedotto dal cambio di velocità tra prima e dopo il vuoto.
2. **Tremolio dei giocatori** (fino a −0,041): misurarlo sulle nostre tracce e, se c'è, levigarlo.
3. **Palla ferma**: i bordi e le interruzioni in cui Metrica tiene la palla.

### Non ancora considerato

Il tocco cambia il possesso a **ogni** tocco, mentre la challenge vuole il **controllo**. Il passo
successivo è il tocco più la conferma del controllo.

---

## 7. Decisioni prese

- **Metrica come base o noi come base:** si va con il metodo 4, nostri giocatori più Metrica dove
  siamo incerti. Il metodo 2 resta il riferimento.
- **Palla:** il nostro rilevatore con filtro di continuità, divieto di salti e palle fuori campo
  escluse. Interpolazione nei vuoti brevi, Metrica in quelli lunghi.
- **Rosa fissa con regole su misura** (`identities.py`), per esempio il portiere vincolato all'area:
  **abbandonata**. Francesco l'ha giudicata pericolosa: si inseguivano i singoli casi.
- **Omografia:** non si corregge, perché lo scarto sul terreno è di 0,19 m. Correggerla a mano è
  vietato.
- **Broadcast:** allineato (offset costante per tempo, replay scartati) e provato per la palla in 3D
  con due camere (sezione 11): la triangolazione funziona, ma **non migliora il possesso**. Non entra
  nella catena.
- **Giocatori: basta ritocchi** (11/11). Su SoccerTrack v2 gli errori sui giocatori sono piccoli e su
  DFL valgono in tutto circa 0,05 di spellF1; i ritocchi provati guadagnano millesimi. Entra solo la
  correzione 2 (riempimento prudente). Il tempo va alla catena finale.
- **Rilevatore delle persone:** yolo26x a 1920 in FP16. Modelli più grandi o una risoluzione
  maggiore non migliorano.

## 8. Prossimi passi (aggiornati l'11/11, consegna il 17/11)

1. **Catena finale in un solo script**, che produce il CSV:
   - palla nostra (filtro di continuità + ricerca guidata), interpolata nei vuoti brevi, Metrica in
     quelli lunghi;
   - giocatori col metodo 4, portieri, correzione 2 sul riempimento;
   - tocco di traverso alla linea di vista più conferma del controllo, tarati su DFL;
   - D dove manca la palla Metrica; tratto prima del calcio d'inizio del secondo tempo (113742) = D.
     Da correggere in `ball_track.py`: usa 113741.
2. **Controlli:** `validate_submission.py`, finestra pubblica solo per la correttezza, rigenerazione
   da zero dello stesso file (riproducibilità).
3. **Documento tecnico** a partire da questo file.
4. Solo se avanza tempo: la regola del tocco, dove si perde di più (misurabile su DFL in poche ore).

---

## 10. Palla in aria, vuoti, tremolio (10/11 sera)

### Tremolio dei giocatori

- Scarto dal moto medio su 5 fotogrammi: **0,02 m** di mediana, 0,18 m al 99° percentile.
- Nelle simulazioni avevamo messo 0,2 m, cioè dieci volte troppo.
- Con il valore vero la regola del tocco sale di circa +0,045.

### Vuoti della palla

Misurati su SoccerNet-Tracking test.

| | Precisione | Copertura |
|---|---|---|
| Prima | 0,954 | 0,677 |
| + rilevamenti deboli (conf ≥ 0,05) entro 2 grandezze dalla retta tra i bordi del vuoto (`ball_link.guided_fill`) | 0,947 | 0,718 |
| + ritaglio ingrandito 1,5× intorno alla posizione prevista | 0,946 | 0,740 |

- Su DFL valgono circa +0,02.
- **Il tocco dedotto attraverso il vuoto** (`dfl_control.touches_across_gaps`) **peggiora** sempre, anche con i vuoti messi dove un giocatore copre la palla. È scartato.

### Geometria della camera tattica

- **Scomposizione dell'omografia:** circa 30 m di altezza. La distanza è poco affidabile, per l'ambiguità di segno e la focale dedotta.
- **Dai giocatori:** proiettando la testa invece dei piedi, l'angolo di ripresa è di **circa 19°** di mediana.
- **Dalle distanze palla–giocatore:**
  - la nostra palla: 90° percentile 7,1 m, 19% oltre 5 m;
  - questi valori si riproducono su DFL con una camera a **25–35°** verso il centro, cioè 1,5–2 m di spostamento sul campo per ogni metro di altezza della palla.

### La palla in aria

- L'omografia vale solo a terra: una palla in aria viene proiettata più lontano lungo la linea di vista.
- Su DFL la palla sta sopra 1 m il 23% del gioco.
- Anche la palla di Metrica mostra lo stesso effetto (17% oltre 5 m).

Effetto sulle regole, camera a 25–35°:

| Regola | spellF1 |
|---|---|
| Tocco, palla vera a terra | 0,705 |
| Tocco, palla proiettata | 0,47–0,51 |
| **Tocco di traverso** (`dfl_control.sight_touches`) | **0,515–0,551** |
| Baseline ±50 | 0,44 (non risente dell'effetto) |

- Il danno viene soprattutto dalla **velocità verticale**: salita, discesa e rimbalzi fanno avanzare e frenare la proiezione, e questo inventa tocchi.
- **Tocco di traverso:**
  - conta solo il cambio di velocità perpendicolare alla linea di vista, che l'altezza non può creare;
  - il toccatore si cerca lungo il tratto di linea di vista dove può stare la palla vera.

### Ricostruzione 3D con una sola camera (`pipeline/ball3d.py`)

Si cerca la parabola con la gravità che passa per i raggi della camera.

- **Su un pallonetto sintetico** con 3 cm di rumore funziona: 0,12 m di errore contro 5,3 m della proiezione.
- **Su DFL è precisa ma timida:**
  - quando dice "in volo" ha ragione nel 99,6% dei casi;
  - ma prende solo il 7% dei fotogrammi in aria;
  - lo spellF1 non migliora.
- **Causa:** con una camera a 30 m l'altezza si vede solo dalla curvatura che la gravità dà alla proiezione. Per voli bassi o brevi, con 3 cm di rumore, quella curvatura è troppo piccola.
- **Abbiamo escluso che sia colpa dei dati:** la palla DFL in aria è fisica, con 0,03 m di scarto mediano da una parabola con g = 9,81.
- Prove fatte e scartate: finestra fissa, tratti tagliati alle discontinuità, finestre di più lunghezze.
- **Possibile via:** triangolare con la **camera broadcast**, che è già allineata nel tempo e vede la stessa palla da un altro punto.

### Errori nostri corretti oggi

- **I buchi della palla di Metrica a gioco vivo** (2% simulato; sulla partita sono solo 2) gonfiavano il costo della palla ferma: da −0,1 a −0,03.
- **La posizione della camera simulata** era sbagliata di 68 m: gli "angoli" erano tutti più bassi del previsto.
- **Il rumore dei giocatori** era dieci volte troppo alto.

---

## 11. Due camere e giocatori su un dataset esterno (10–11/11)

### Palla in 3D con tattica + broadcast (`pipeline/broadcast_ball.py`, `pipeline/triangulate.py`)

- **Centri delle camere** stimati dai giocatori: la retta che passa per la testa proiettata a terra e
  per la testa vera (1,8 m sopra il piede) passa per la camera. Tattica circa (53, 116, 29) m,
  broadcast circa (53, 110, 22) m; stabili su metà indipendenti dei dati.
- **Palla broadcast:** stesso rilevatore a 1280 (la palla è circa 14 px), stesso filtro di
  traiettoria, su 93.388 fotogrammi di gioco vivo.
- **Copertura:** palla vista da entrambe le camere nel 42% dei fotogrammi valutati.
- **Controlli senza verità:**
  - i due raggi si incontrano: distanza mediana 0,12 m;
  - offset di tempo migliore 0 in entrambi i tempi;
  - palle a terra: i due punti sul campo distano 0,23 m di mediana;
  - voli: accelerazione verticale mediana −8,1 m/s² (gravità −9,8);
  - l'altezza ha una deriva lenta di circa ±0,4 m.
- **Valore per il possesso, simulato su DFL** con il rumore misurato:
  tocco di traverso senza 3D 0,506; con il 3D stimato 0,504; con il 3D esatto 0,546; baseline 0,436.
  Con la precisione che abbiamo **non serve**: scartato.

### Giocatori su SoccerTrack v2 (`pipeline/ext_players_eval.py`)

Due partite, 4 minuti ciascuna, contro i riquadri disegnati a mano (`mot/`). Panorama tagliato in
finestre 1920×1080, giocatori già alla nostra grandezza (46–51 px), ricompresso a 5 Mbit/s come il
nostro video. Il "maestro" (al posto di Metrica) è il tracking del fornitore BePro, più pulito di
Metrica: il metodo 4 lì è un po' ottimista.

| | Partita 1 | Partita 2 |
|---|---|---|
| Giocatori trovati | 95,5% | 96,0% |
| …con un altro giocatore entro 1,5 m | 77% | 80% |
| Squadra giusta quando il colore decide | 99,9% | 99,9% |
| Metodo 4, squadra giusta | 99,4% | 98,8% |
| Metodo 4, squadra giusta entro 1,5 m da un altro | 93,0% | 88,1% |
| Portieri: precisione / squadra giusta | 99,7% / 100% | 99,4% / 100% |
| Piedi rispetto al riquadro a mano (mediana) | 0,12 m | 0,10 m |
| Continuità delle tracce (IDF1) | 0,36 | 0,38 |

- **La compressione non costa quasi niente:** sul video originale si guadagnano 0,2–0,5 punti
  percentuali, nel rumore.
- **Errore trovato e corretto:** leggere il video originale con ffmpeg in uscita grezza duplicava
  24 fotogrammi (6000 invece di 5976) e spostava tutto rispetto alle annotazioni. Ora si legge con
  `-fps_mode passthrough`. La misura ricompressa non ne era toccata.

### Tre correzioni provate, ciascuna da sola (metodo 4, squadra giusta entro 1,5 m)

| | Partita 1 | Partita 2 |
|---|---|---|
| Di partenza | 93,0% | 88,1% |
| 1. Colore dai soli fotogrammi isolati della traccia | 90,7% | 89,1% |
| **2. Niente riempimento se un altro rilevamento è entro 1,5 m** | **93,7%** | **88,6%** |
| 3. Cucitura delle tracce (≤ 2 s, raggiungibile a 8 m/s, colore compatibile) | 92,6% | 88,0% |

- **2: tenuta.** Meglio in entrambe; riquadri falsi da 1757 a 1334 e da 2585 a 2344; giocatori
  trovati −0,4 punti.
- **1: scartata.** Il colore deciso è giusto meno spesso (da 98,1% a 94,1% e da 99,1% a 96,4%): nei
  contrasti la traccia passa spesso a un altro giocatore, e il colore di prima porta la squadra
  sbagliata. Il colore misurato dentro il contrasto è più affidabile del previsto.
- **3: scartata così com'è.** Tracce lunghe il doppio (IDF1 da 0,36–0,38 a 0,41), ma cuciture
  sbagliate: riquadri sulla traccia di un altro da 7,6% a 12,1% e da 4,2% a 7,8%, riquadri falsi in
  aumento, portieri trovati −2 punti nella partita 2.
- **Conclusione:** i guadagni sono di millesimi di spellF1. Decisione: fermarsi sui giocatori e
  chiudere la catena (sezioni 7 e 8).

### Repository di squadra

Il codice (solo `pipeline/` e questo file, niente `data/`) è nel repository privato di squadra,
ramo `pipeline-fgaza`, con una richiesta di unione verso `main`.

## 12. Tocco visto nell'immagine contro tocco ricostruito in 2D (11/11, `pipeline/ext_touch_eval.py`)

SoccerTrack v2, primi 15 minuti del 1° tempo di due partite, catena completa sul video (persone +
nostro rilevatore di palla, metodo 4, correzione 2). Verità = azioni di palla etichettate a mano
(bas/), possesso derivato dagli eventi (proxy più severo della regola della gara: spell mediano
1,3 s); palla morta presa dalla verità per tutte le varianti, quindi cambia solo l'attribuzione.

Errore trovato e corretto: il tracciatore si agganciava a oggetti fermi sul lato lontano (entro la
tolleranza di 10 px dal contorno), palla a 56 m da quella vera in 117093. Filtro statico: cella di
8 px occupata in più di 1500 fotogrammi distinti = non è la palla (camera fissa). Dopo: 1,5 m
(117093) e 2,9 m (128057) di mediana agli eventi; in 128057 resta una coda (p90 65 m) di un
oggetto che si muove vicino alla linea lontana, non filtrato.

spellF1 medio sulle due partite (tetto con le azioni vere come tocchi: 0,962):

| variante | spellF1 |
|---|---|
| più vicino, finestra 50 | 0,107 |
| controllo 2D | 0,091 |
| tocco 2D, T=3 | 0,128 |
| linea di vista, T=3 | 0,109 |
| contatto nell'immagine, T=3 | 0,142 |
| contatto nell'immagine + controllo nell'immagine, T=8 | 0,147 |

Tocchi riconosciuti (±5 fotogrammi): 2D T=3 richiamo 29% / 16%, precisione ≥0,59 / 0,46;
immagine T=3 richiamo 21% / 14%, precisione ≥0,72 / 0,58; squadra giusta ~80% per tutti.

Conclusioni: il contatto nell'immagine è la fonte di tocchi migliore (+0,02 spellF1, più precisa),
ma tutte le varianti stanno tra 0,10 e 0,15 contro un tetto di 0,96. Il collo di bottiglia non è
come si riconosce il tocco ma la palla: vista nel 46% / 29% dei fotogrammi vivi. Due partite sole,
proxy severo: il guadagno di 0,02 è al limite del rumore. Per la consegna non si integra; resta il
punto 4 del §8 se avanza tempo.
