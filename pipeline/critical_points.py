"""List the moments of a clip a human should check, from compare_tracking + identities output.

Every row is an episode (consecutive frames, gaps <= 5 merged) with the time to seek in the
clip's videos. Nothing here is a label: each row is a question for whoever watches the video.

Types
  possession_differs   ball seen; nearest player's team differs, Metrica vs fixed roster
  missing_at_ball      nearest roster player to the ball (<= 3 m) is one Metrica does not have
  team_disagrees       roster identity sits on a Metrica point of the other team
  lost_in_view         roster identity not seen while its position is in the image (>= 2 s)
  identity_jump        roster identity re-acquired further than a player can run (swap?)
  metrica_frozen       Metrica point not moving at all (>= 2 s)
  outside_roster       team-coloured box with no identity (team already has 11) (>= 0.4 s)

    python pipeline/critical_points.py --start 11185 --frames 7500
Writes data/diagnostics/tracking-comparison/critical_<tag>.csv and prints a summary.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import GAME, GAMES, require  # noqa: E402
from compare_tracking import OUT_DIR, read_atd  # noqa: E402

MIN_FRAMES = {"possession_differs": 5, "missing_at_ball": 5, "team_disagrees": 10,
              "lost_in_view": 50, "identity_jump": 1, "metrica_frozen": 50, "outside_roster": 10}
MERGE_GAP = 5


def episodes(frames, kind, start, extra=None):
    """Consecutive runs of `frames` (sorted ints) as episode rows."""
    frames = np.unique(np.asarray(frames, dtype=int))
    if not len(frames):
        return []
    cut = np.flatnonzero(np.diff(frames) > MERGE_GAP)
    out = []
    for a, z in zip(np.r_[frames[0], frames[cut + 1]], np.r_[frames[cut], frames[-1]]):
        if z - a + 1 >= MIN_FRAMES[kind]:
            t = (a - start) / 25
            out.append({"type": kind, "first": int(a), "last": int(z),
                        "seconds": round((z - a + 1) / 25, 2),
                        "video_time": "{:d}:{:05.2f}".format(int(t // 60), t % 60),
                        "detail": extra(a, z) if extra else ""})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, default=11185)
    parser.add_argument("--frames", type=int, default=500)
    args = parser.parse_args()
    length, width = GAMES[GAME]["pitch"]
    start, n = args.start, args.frames
    tag = "{}_{}".format(start, start + n - 1)
    det = pd.read_csv(os.path.join(OUT_DIR, "detections_{}.csv".format(tag)))
    met = pd.read_csv(os.path.join(OUT_DIR, "metrica_{}.csv".format(tag)))
    ids = pd.read_csv(os.path.join(OUT_DIR, "identities_{}.csv".format(tag)))
    _, ball = read_atd(require(["atd_raw"])[0], start, n)
    met["X"], met["Y"] = met.x * length, met.y * width
    ids["team"] = ids.role.str[-1].where(ids.role != "other", "other")  # GK_H -> H
    players = ids[ids.role != "other"].merge(
        det[["matched_slot", "matched_team"]], left_on="det", right_index=True, how="left")
    rows = []

    # 1-2. Nearest player to the ball, Metrica vs roster (roster: seen or lost in view).
    differs, missing, info = [], [], {}
    met_by, id_by = dict(list(met.groupby("frame"))), dict(list(players.groupby("frame")))
    for i in range(n):
        f = start + i
        if not np.isfinite(ball[i, 0]) or f not in met_by or f not in id_by:
            continue
        b = ball[i] * [length, width]
        m = met_by[f]
        r = id_by[f][id_by[f].state != "out_of_view"]
        if not len(r):
            continue
        dm = np.hypot(m.X - b[0], m.Y - b[1])
        dr = np.hypot(r.x_m - b[0], r.y_m - b[1])
        km, kr = dm.idxmin(), dr.idxmin()
        info[f] = (int(m.loc[km, "slot"]), m.loc[km, "team"], round(float(dm[km]), 1),
                   int(r.loc[kr, "identity"]), r.loc[kr, "role"], round(float(dr[kr]), 1),
                   r.loc[kr, "state"])
        if m.loc[km, "team"] != r.loc[kr, "team"]:
            differs.append(f)
        if dr[kr] <= 3 and r.loc[kr, "state"] == "observed" and r.loc[kr, "matched_slot"] < 0:
            missing.append(f)

    def ball_detail(a, z):
        mid = info.get((a + z) // 2) or info[a]
        return ("Metrica: slot {} {} a {} m | roster: {}{} a {} m ({})"
                .format(mid[0], mid[1], mid[2], mid[4], mid[3], mid[5], mid[6]))
    rows += episodes(differs, "possession_differs", start, ball_detail)
    rows += episodes(missing, "missing_at_ball", start, ball_detail)

    # 3. Identity on a Metrica point of the other team.
    bad = players[players.matched_team.notna() & (players.matched_team != players.team)]
    for k, g in bad.groupby("identity"):
        role = g.role.iloc[0]
        rows += episodes(g.frame, "team_disagrees", start, lambda a, z, k=k, role=role, g=g: (
            "{}{} su slot Metrica {}".format(role, k, sorted(set(
                g[(g.frame >= a) & (g.frame <= z)].matched_slot.astype(int))))))

    # 4. Lost while in view.
    for k, g in players[players.state == "lost"].groupby("identity"):
        role = g.role.iloc[0]
        rows += episodes(g.frame, "lost_in_view", start,
                         lambda a, z, k=k, role=role: "{}{} non visto".format(role, k))

    # 5. Re-acquired too far: more than 3 m + 0.4 m per frame since last seen (a sprint).
    seen = players[players.state == "observed"].sort_values(["identity", "frame"])
    for k, g in seen.groupby("identity"):
        gap = g.frame.diff().values
        jump = np.hypot(g.x_m.diff(), g.y_m.diff()).values
        role = g.role.iloc[0]
        for f, gp, j in zip(g.frame.values, gap, jump):
            if gp > 1 and j > 3 + 0.4 * gp:
                rows += episodes([f], "identity_jump", start, lambda a, z, k=k, role=role,
                                 j=j, gp=gp: "{}{} riagganciato a {:.1f} m dopo {} fotogrammi"
                                 .format(role, k, j, int(gp)))

    # 6. Frozen Metrica points.
    for s, g in met.sort_values("frame").groupby("slot"):
        g = g[g.frame.diff() == 1]
        still = g[(g.x.diff().abs() < 1e-5) & (g.y.diff().abs() < 1e-5)]
        team = g.team.iloc[0] if len(g) else ""
        rows += episodes(still.frame, "metrica_frozen", start, lambda a, z, s=s, team=team, g=g: (
            "slot {} {}, stato: {}".format(s, team, g[(g.frame >= a) & (g.frame <= z)]
                                           .state.mode().iloc[0])))

    # 7. Team-coloured boxes outside the roster.
    used = set(ids.det[ids.det >= 0])
    pitch = det[(det.on_pitch == 1) & det.group_role.isin(["H", "A"]) & ~det.index.isin(used)]
    for role, g in pitch.groupby("group_role"):
        rows += episodes(g.frame, "outside_roster", start,
                         lambda a, z, role=role: "riquadri {} senza identita'".format(role))

    out = pd.DataFrame(rows, columns=["type", "first", "last", "seconds", "video_time", "detail"])
    # Ball seen in the episode is a rough live-play hint (missing ball is mostly dead ball).
    seen_ball = np.isfinite(ball[:, 0])
    out["ball_seen_share"] = [round(float(seen_ball[a - start:z - start + 1].mean()), 2)
                              for a, z in zip(out["first"], out["last"])]
    out = out.sort_values(["type", "seconds"], ascending=[True, False])
    out.to_csv(os.path.join(OUT_DIR, "critical_{}.csv".format(tag)), index=False)
    print(out.groupby("type").agg(episodes=("seconds", "size"), total_s=("seconds", "sum"),
                                  longest_s=("seconds", "max")).to_string())


if __name__ == "__main__":
    main()
