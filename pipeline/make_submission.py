"""The whole chain on the provided match: tracking (ATD) -> B1 -> C1 -> A1 -> <team>.csv.

Inputs: the provided tactical ATD only (players without identity, their team, the ball) and the
models and parameters trained on the DFL matches (dfl_paths.models_dir()):
    deadball_lgbm.txt + deadball_meta.json   B1, P(ball dead) per frame
    rule_params.json (optional)              B2, touch + confirmation of control (not in the final chain)
    player_net.pt (optional)                 C1, P(home) from the players, also without the ball
    decoder_params.json                      A1, the segment decoder that joins the three
Nothing is fitted here and no label of the provided match is read. Only the graded halves are
computed; every other frame is X. The second half is graded from before its kick-off: those
frames are D (kick-off frame as published in starter/load_smart_tagging.py).

Run it locally, never through an online AI tool: only aggregates are printed.
    python pipeline/make_submission.py --out evaluation/results/<team>.csv [--models DIR] [--no-net]
Then: python evaluation/validate_submission.py evaluation/results/<team>.csv
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import deadball  # noqa: E402
import decoder as dec  # noqa: E402
import dfl_paths  # noqa: E402
import touch_control as tc  # noqa: E402
from baseline import HOME, team_of_each_track, write_submission  # noqa: E402
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

log = logging.getLogger("make_submission")
FPS = 25
MISSING = "N"  # first character of "NaN" in the raw ATD


def load_atd(game=GAME):
    """players (F,S,2) and ball (F,2), normalised 0-1 with NaN where missing, team (S,) 'H'/'A'.
    F = frames of the tactical video: row i is video frame i (frame_id - 1)."""
    paths = asset_paths(game)
    slots = team_of_each_track(paths["atd_meta"])
    n = GAMES[game]["n_video_frames"]
    players = np.full((n, len(slots), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    with open(paths["atd_raw"]) as handle:
        for line in handle:
            frame_id, groups, b = line.rstrip("\n").split(":")
            v = int(frame_id) - 1
            for s, g in enumerate(groups.split(";")):
                if g[0] != MISSING:
                    players[v, s] = [float(x) for x in g.split(",")[:2]]
            if b[0] != MISSING:
                ball[v] = [float(x) for x in b.split(",")[:2]]
    return players, np.where(slots == HOME, "H", "A"), ball


def graded_frames(game=GAME):
    """(video frames (G,), half (G,), forced D (G,)) of the graded halves, in order."""
    idx, half = [], []
    for h, first, last in half_video_blocks(game):
        idx.append(np.arange(first, last + 1))
        half.append(np.full(last - first + 1, h, np.int8))
    idx, half = np.concatenate(idx), np.concatenate(half)
    return idx, half, (half == 2) & (idx < KICKOFF_VIDEO_FRAMES["2nd Half"])


def run_chain(players, team, ball, half, forced_dead, models, use_net=True, kickoff=None):
    """H / A / D per frame from the tracking of the graded frames. Returns (labels, sources)."""
    team_f = np.broadcast_to(team, players.shape[:2])
    with open(os.path.join(models, "decoder_params.json")) as fh:
        par = json.load(fh)
    p_raw, meta = deadball.predict_p_raw(ball, players, team, half, folder=models)
    p_dead = np.where(forced_dead, 1.0, deadball.median_filter(p_raw, meta["window"], half))
    # A1 may have chosen its own median filter for B1 (a1_valuta.py with $A1_B1_WINDOWS); B2 keeps B1's
    p_dead_a1 = p_dead if "b1_window" not in par else np.where(
        forced_dead, 1.0, deadball.median_filter(p_raw, par["b1_window"], half))
    missing = ~np.isfinite(ball[:, 0])

    ext = {"p_dead": p_dead_a1}
    rule_file = os.path.join(models, "rule_params.json")
    if os.path.isfile(rule_file):   # B2 only if the models folder has it (dropped from the final chain, 17/11)
        with open(rule_file) as fh:
            rule = json.load(fh)
        f = tc.Features.from_tracking(players, team_f, ball, p_dead > 0.5, np.where(missing, 2, 0))
        touch = tc.detect_touches(ball, f.dmin, f.src, rule["T"])
        _, ext["p_home"], _ = tc.run_touch_control(
            f, touch, rule["T1"], rule["R"], rule["N"], rule["fallback_n"]
        )
    net = os.path.join(models, "player_net.pt")
    if use_net and par["net_weight"] > 0 and os.path.isfile(net):
        import player_net as pn  # needs PyTorch: only when the network is in the chain

        ext["p_home_net"] = pn.predict_p_home(
            pn.load_model(net), players, team, ball, None, half, stride=2, device="cpu", kickoff=kickoff,
            tta=not par.get("media_seq"), media_seq=int(par.get("media_seq", 0)),
        )  # media_seq (decoder_params.json): media sullo stesso fotogramma al posto delle simmetrie (16/11)
    logp = dec.match_logp(dec.evidence(players, team_f, ball), ext, par["net_weight"])
    labels = dec.decode(logp, half, par["switch_cost"], par["min_dur"])
    return labels, "B1 + " + ("B2 + " if "p_home" in ext else "") + "A1" + (
        " + C1 (weight {:g})".format(par["net_weight"]) if "p_home_net" in ext else ""
    )


def describe(labels, half, missing):
    """Aggregates only: shares, possessions per minute, typical length, agreement with the ball."""
    runs = [(a, z, k) for a, z, k in dec.run_lengths(labels, half) if k in ("H", "A")]
    lens = np.array([z - a for a, z, _ in runs])
    minutes = len(labels) / FPS / 60
    dead = labels == "D"
    return (
        "H {:.1%}, A {:.1%}, D {:.1%} | {} possessions, {:.1f} per minute, median {:.1f} s | "
        "ball missing {:.1%}; D where the ball is missing {:.1%}, D where it is there {:.1%}"
    ).format(
        (labels == "H").mean(),
        (labels == "A").mean(),
        dead.mean(),
        len(runs),
        len(runs) / minutes,
        float(np.median(lens)) / FPS if len(lens) else 0.0,
        missing.mean(),
        dead[missing].mean() if missing.any() else 0.0,
        dead[~missing].mean(),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out", required=True, help="CSV to write (evaluation/results/<team>.csv)"
    )
    ap.add_argument(
        "--models",
        default=None,
        help="folder with the models (default dfl_paths.models_dir())",
    )
    ap.add_argument(
        "--no-net", action="store_true", help="leave C1 out even if its model is there"
    )
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    players, team, ball = load_atd()
    idx, half, forced = graded_frames()
    labels, sources = run_chain(
        players[idx],
        team,
        ball[idx],
        half,
        forced,
        a.models or dfl_paths.models_dir(),
        not a.no_net,
        np.searchsorted(idx, sorted(KICKOFF_VIDEO_FRAMES.values())),
    )
    full = np.full(len(ball), "X", dtype="<U1")
    full[idx] = labels
    write_submission(a.out, full)
    log.info("chain: %s", sources)
    log.info(describe(labels, half, ~np.isfinite(ball[idx, 0])))
    log.info("wrote %s (%d rows)", a.out, len(full))


if __name__ == "__main__":
    main()
