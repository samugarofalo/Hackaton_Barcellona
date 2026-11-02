"""Valutazione locale senza pandas.

Su alcuni PC Windows (Smart App Control attivo) le DLL non firmate di pandas vengono bloccate.
Le funzioni che calcolano le metriche in evaluation/possession_metrics.py usano solo numpy:
pandas serve soltanto a leggere i CSV. Qui li leggiamo con il modulo csv e usiamo il file degli
organizzatori cosi com'e', senza modificarlo.

Se pandas funziona, viene usato quello vero e il risultato e identico.

    python pipeline/local_eval.py                       # tutti i CSV in evaluation/results
    python pipeline/local_eval.py mio.csv altro.csv     # solo questi file

Punteggio contro la verita' pubblica (5 minuti): serve solo a controllare che la pipeline sia
corretta, mai a scegliere tra due versioni. Il controllo finale va fatto con
evaluation/validate_submission.py e run_evaluation.py su una macchina dove pandas funziona.
"""

import csv
import glob
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
N_FRAMES = 199319
VALID_LABELS = {"H", "A", "D", "X"}


def _use_pandas_stub_if_needed():
    """Se pandas non si carica, mette al suo posto un modulo vuoto: le metriche non lo usano."""
    try:
        import pandas  # noqa: F401
        return False
    except ImportError:
        for name in [n for n in sys.modules if n == "pandas" or n.startswith("pandas.")]:
            del sys.modules[name]
        stub = types.ModuleType("pandas")
        stub.DataFrame = type("DataFrame", (), {})
        sys.modules["pandas"] = stub
        return True


STUBBED = _use_pandas_stub_if_needed()
import possession_metrics as pm  # noqa: E402


def read_labels(path):
    """Legge un CSV frame,label. Solleva ValueError con un motivo chiaro se non e' valido."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = [cell.strip().lower() for cell in next(reader)]
        if "frame" not in header or "label" not in header:
            raise ValueError("intestazione attesa 'frame,label', trovata {}".format(header))
        frame_col, label_col = header.index("frame"), header.index("label")
        frames, labels = [], []
        for row in reader:
            frames.append(int(row[frame_col]))
            labels.append(row[label_col].strip().upper())

    if len(labels) != N_FRAMES:
        raise ValueError("righe {} invece di {}".format(len(labels), N_FRAMES))
    if not np.array_equal(np.array(frames), np.arange(N_FRAMES)):
        raise ValueError("la colonna frame non e' 0..{} in ordine".format(N_FRAMES - 1))
    bad = sorted(set(labels) - VALID_LABELS)
    if bad:
        raise ValueError("etichette non valide: {}".format(bad[:5]))
    return pm.label_codes(np.array(labels, dtype=object))


def main(paths):
    gt = read_labels(GT_PATH)
    if not paths:
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))

    rows = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            result = pm.evaluate(gt, read_labels(path))
        except ValueError as error:
            rows.append((name, None, str(error)))
            continue
        rows.append((name, result, None))

    scored = sorted((r for r in rows if r[1]), key=lambda r: -r[1]["spell_f1_mean"])
    print("pandas {}".format("sostituito da un modulo vuoto" if STUBBED else "funzionante"))
    print("GT pubblica: {} frame valutati, {} possessi veri\n".format(
        scored[0][1]["n_graded"] if scored else "?", scored[0][1]["n_gt_spells"] if scored else "?"))
    print("{:<4} {:<26} {:>8} {:>8} {:>6} {:>7}".format(
        "rank", "submission", "macroF1", "spellF1", "poss", "dur"))
    for rank, (name, result, _) in enumerate(scored, 1):
        print("{:<4} {:<26} {:>8.4f} {:>8.4f} {:>6} {:>7.1f}".format(
            rank, name, result["macro_f1"], result["spell_f1_mean"],
            result["n_pred_spells"], result["mean_spell_duration_s"]))
    for name, result, error in rows:
        if result is None:
            print("SCARTATO {}: {}".format(name, error))


if __name__ == "__main__":
    main(sys.argv[1:])
