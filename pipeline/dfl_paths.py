"""Where the DFL (Bassek et al. 2025) files are, locally or on Kaggle, and the shared output contract.

The DFL matches are other games (CC BY 4.0), so training and tuning on them is allowed.

Input: the converted matches written by convert_dfl.py, one <match>.npz each.
    1. $DFL_CONVERTED if set
    2. data/external/bassek2025/converted (dfl_bench.CONVERTED, the local default)
    3. on Kaggle, the first folder under /kaggle/input that holds DFL-MAT-*.npz files
       (the raw XML only? run convert_dfl.py with SRC pointed at it first)

Output: what the notebooks hand to each other, one file per match and per producer.
    $DFL_OUT if set, else /kaggle/working/dfl_outputs on Kaggle, else
    data/external/bassek2025/outputs locally.

    <match>_deadball.npz   written by B1   p_dead (F,) float32 in [0, 1], P(ball dead) per frame
    <match>_rule.npz       written by B2   labels (F,) '<U1' in {H, A, D}
                                           p_home (F,) float32, P(home | ball alive) per frame
    <match>_decoded.npz    written by A1   labels (F,) '<U1' in {H, A, D}

F = number of frames of that match's converted npz, same order. Every file also stores
`variant` (a short string: which condition / seed / parameters produced it).
"""

import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOCAL_CONVERTED = os.path.join(ROOT, "data", "external", "bassek2025", "converted")
LOCAL_OUT = os.path.join(ROOT, "data", "external", "bassek2025", "outputs")
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_OUT = "/kaggle/working/dfl_outputs"


def on_kaggle():
    return os.path.isdir(KAGGLE_INPUT)


def converted_dir():
    """Folder holding the converted DFL matches. Raises if none is found."""
    env = os.environ.get("DFL_CONVERTED")
    if env:
        return env
    if glob.glob(os.path.join(LOCAL_CONVERTED, "*.npz")):
        return LOCAL_CONVERTED
    if on_kaggle():
        hits = sorted(glob.glob(os.path.join(KAGGLE_INPUT, "**", "DFL-MAT-*.npz"), recursive=True))
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError(
        "No converted DFL matches found. Run pipeline/convert_dfl.py, or set DFL_CONVERTED to the "
        "folder with the <match>.npz files.")


def match_paths():
    """Sorted paths of the converted matches."""
    return sorted(glob.glob(os.path.join(converted_dir(), "*.npz")))


def out_dir():
    path = os.environ.get("DFL_OUT") or (KAGGLE_OUT if on_kaggle() else LOCAL_OUT)
    os.makedirs(path, exist_ok=True)
    return path


def out_path(match, kind):
    """kind: 'deadball', 'rule' or 'decoded' (see the module docstring)."""
    return os.path.join(out_dir(), "{}_{}.npz".format(match, kind))


def match_name(path):
    return os.path.splitext(os.path.basename(path))[0]
