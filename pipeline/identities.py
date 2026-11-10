"""Fixed-roster player identities from the detector boxes of compare_tracking.py.

Rule: once the roster exists, nobody appears, vanishes or doubles. Roles and caps: 10 outfield
players and 1 goalkeeper per team, 3 officials; an identity's role never changes. Every frame each
identity takes at most one box (one-to-one, optimal), within the distance a player can cover since
it was last seen. An identity with no box is kept: "lost" if its position is in the image,
"out_of_view" if not.

Box cues come from team_colour.py when its output exists (else compare_tracking's colour groups):
  H / A   team from shirt colour. Binding: such a box can never go to the other team.
  GK      goalkeeper kit. In this match both keepers wear similar colours, so the keeper's team
          is the one defending that half (direction from the first frame, swapped at half time).
  other   officials; never a player.
  ""      unsure: any player identity may take it, no new identity is made from it.

A box nobody claims re-acquires the nearest unseen identity of its role (no distance cap once a
player has been gone long enough to be anywhere). A new identity is made only if its role has no
unseen identity and is below its cap: an earlier version spawned a duplicate when a lost player
reappeared far away, and the duplicate then held the place the Athletic keeper needed.

The roster is built at the first frame from the boxes plus the Metrica points out of view there
(a point in its own penalty area becomes that team's keeper). With --kickoff the laws are a check:
at a kickoff every player is in his own half. Substitutions and red cards are not handled yet.

    python pipeline/compare_tracking.py --start S --frames N
    python pipeline/team_colour.py --start S --frames N          # optional, better colour cue
    python pipeline/identities.py --start S --frames N --kickoff

Outputs in data/diagnostics/tracking-comparison/:
    identities_<tag>.csv   frame, identity, role, x_m, y_m, state, det
    identities_<tag>.mp4   camera view: box and id of each identity, lost ones as rings
    pitch2d_<tag>.mp4      2D, Metrica as supplied (left) vs fixed roster without officials (right)
    identities_<tag>.json  checks
"""

import argparse
import json
import os
import sys

import cv2
import lap
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), HERE]
from games import GAME, GAMES, half_of_video_frame, require  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402
from compare_tracking import OUT_DIR, W, H, decode, encoder, read_atd, header  # noqa: E402

CAP = {"H": 10, "A": 10, "GK_H": 1, "GK_A": 1, "other": 3}
# Metres. Detector feet jitter ~0.5 m and a sprinter covers ~0.4 m per frame at 25 fps.
GATE_M, VMAX_M, GATE_MAX_M, REACQUIRE_M = 1.5, 0.4, 30.0, 3.0
# Keeper kit on an outfield identity of the same team (or the reverse): allowed, costs metres.
KEEPER_SWAP_M = 2.0
HALFWAY_MARGIN_M = 1.0
# Keeper kit counts as a keeper only inside the own penalty area plus this margin; elsewhere the
# keeper colour group also catches outfield players in bright light (a false "GA" at midfield).
KEEPER_AREA_MARGIN_M = 5.0
# A player whose box vanishes next to another one is hidden behind him: follow that one for a while.
OCCLUSION_M, OCCLUSION_MAX = 2.5, 75
# In the 2D view an identity unseen for longer than this is not drawn: its position is unknown.
STALE_FRAMES = 50
PENALTY_AREA = (16.5, 40.3)  # depth, width in metres
BGR = {"H": (60, 60, 230), "A": (230, 120, 30), "other": (0, 255, 255)}
COLOUR = {"H": "#d62728", "A": "#1f77b4", "other": "#ffd700"}
BIG = 1e6


def team(role):
    """'H', 'A' or 'other' for any role (GK_H -> H)."""
    return role[-1] if role != "other" else "other"


def label(role, identity):
    return "{}{}".format({"other": "o"}.get(role, role.replace("GK_", "G")), identity)


def cost_of(role, cue, area=""):
    """Extra cost (m) of giving a box with this cue to an identity with this role. `area` is the
    team whose penalty area (plus KEEPER_AREA_MARGIN_M) holds the box.

    A keeper only takes boxes in his own area: there a keeper or unsure box is free, a box the
    classifier calls his team costs a little, the other team's a little more (the keeper kit can
    fool the classifier). Outside it he takes nothing: an earlier version let keepers chase
    outfield players to midfield."""
    if role.startswith("GK_"):
        if area != team(role) or cue == "other":
            return BIG
        if cue in ("", role):
            return 0.0
        return KEEPER_SWAP_M if team(cue) == team(role) else 2 * KEEPER_SWAP_M
    if cue == "" or cue == role:
        return 0.0
    if "other" not in (role, cue) and team(role) == team(cue):
        return KEEPER_SWAP_M
    return BIG


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, default=11185)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--kickoff", action="store_true", help="first frame is a kickoff")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    length, width = GAMES[GAME]["pitch"]
    start, n = args.start, args.frames
    end = start + n - 1
    tag = "{}_{}".format(start, end)
    det = pd.read_csv(os.path.join(OUT_DIR, "detections_{}.csv".format(tag)))
    met = pd.read_csv(os.path.join(OUT_DIR, "metrica_{}.csv".format(tag)))
    det = det[det.on_pitch == 1].copy()
    det["X"], det["Y"] = det.pitch_x * length, det.pitch_y * width
    colour_path = os.path.join(OUT_DIR, "team_colour_{}.csv".format(tag))
    if os.path.exists(colour_path):
        cues = pd.read_csv(colour_path, keep_default_na=False).set_index("det").cue
        det["cue"] = cues.reindex(det.index).fillna("").astype(str)
        cue_source = "team_colour"
    else:
        det["cue"] = det.group_role.fillna("").astype(str)
        cue_source = "kmeans_groups"
    met["X"], met["Y"] = met.x * length, met.y * width
    by_frame = dict(list(det.groupby("frame")))
    empty = det.iloc[:0]
    matrices, _, _ = load_homography()

    # Which side each team defends, from the first frame's outfield colours.
    first = by_frame.get(start, empty)
    home_left = bool(first[first.cue == "H"].X.mean() < first[first.cue == "A"].X.mean())
    first_half = half_of_video_frame(GAME, start)

    def defender_of(x, frame):
        left_team = "H" if home_left else "A"
        if half_of_video_frame(GAME, frame) not in (None, first_half):
            left_team = "A" if left_team == "H" else "H"
        return left_team if x < length / 2 else ("A" if left_team == "H" else "H")

    def area_of(x, y, frame, margin=0.0):
        """Team whose penalty area (grown by margin) holds the point, else ''."""
        depth, wide = PENALTY_AREA
        if abs(y - width / 2) > wide / 2 + margin or depth + margin < x < length - depth - margin:
            return ""
        return defender_of(x, frame)

    def role_of(cue, x, y, frame):
        if cue != "GK":
            return cue
        keeper_team = area_of(x, y, frame, KEEPER_AREA_MARGIN_M)
        return "GK_" + keeper_team if keeper_team else ""

    def own_half(role, x):
        if not args.kickoff or role == "other" or abs(x - length / 2) <= HALFWAY_MARGIN_M:
            return True
        return defender_of(x, start) == team(role)

    def in_own_box(t, x, y):
        depth, wide = PENALTY_AREA
        return abs(y - width / 2) <= wide / 2 and defender_of(x, start) == t and (
            x <= depth or x >= length - depth)

    # Roster at the first frame.
    ids, init = [], {"relabelled_boxes": 0, "rejected_metrica_slots": [],
                     "home_defends_left": home_left, "cue_source": cue_source}
    count = lambda role: sum(i["role"] == role for i in ids)  # noqa: E731
    first_det = {}
    for b in first.sort_values("conf", ascending=False).itertuples():
        role = role_of(b.cue, b.X, b.Y, start)
        if role in ("H", "A") and not own_half(role, b.X):
            role = "A" if role == "H" else "H"
            init["relabelled_boxes"] += 1
        if role in CAP and count(role) < CAP[role]:
            first_det[len(ids)] = b.Index
            ids.append({"role": role, "pos": np.array([b.X, b.Y]), "vel": np.zeros(2),
                        "seen": start, "source": "box"})
    for p in met[(met.frame == start) & (met.state == "out_of_view")].itertuples():
        keeper = "GK_" + p.team
        role = keeper if in_own_box(p.team, p.X, p.Y) and count(keeper) == 0 else p.team
        if not own_half(role, p.X):
            init["rejected_metrica_slots"].append(int(p.slot))
        elif count(role) < CAP[role]:
            ids.append({"role": role, "pos": np.array([p.X, p.Y]), "vel": np.zeros(2),
                        "seen": None, "source": "metrica_slot_{}".format(p.slot)})

    rows, spawned, unclaimed, reacquired = [], [], 0, 0
    for frame in range(start, end + 1):
        boxes = by_frame.get(frame, empty)
        bpos = boxes[["X", "Y"]].values
        broles = [role_of(c, x, y, frame) for c, x, y in zip(boxes.cue, boxes.X, boxes.Y)]
        areas = [area_of(x, y, frame, KEEPER_AREA_MARGIN_M) for x, y in zip(boxes.X, boxes.Y)]
        det_of = [first_det.get(k) if frame == start else None for k in range(len(ids))]

        if frame > start and len(boxes) and ids:
            lost = np.array([frame - i["seen"] if i["seen"] is not None else 50 for i in ids])
            pred = np.array([i["pos"] + i["vel"] * min(l, 5) for i, l in zip(ids, lost)])
            gate = np.minimum(GATE_M + VMAX_M * (lost - 1), GATE_MAX_M)
            dist = np.linalg.norm(pred[:, None] - bpos[None], axis=2)
            extra = np.array([[cost_of(i["role"], r, a) for r, a in zip(broles, areas)]
                              for i in ids])
            cost = dist + extra
            cost[dist > gate[:, None]] = BIG
            _, id_to_box, _ = lap.lapjv(cost, extend_cost=True, cost_limit=BIG / 2)
            for k, j in enumerate(id_to_box):
                if j >= 0 and cost[k, j] < BIG:
                    i = ids[k]
                    step = (bpos[j] - i["pos"]) / lost[k]
                    i["vel"] = 0.7 * i["vel"] + 0.3 * step if i["seen"] is not None else np.zeros(2)
                    i["pos"], i["seen"] = bpos[j].copy(), frame
                    det_of[k] = boxes.index[j]

        # Unclaimed boxes: ignore a second box on someone seen now; else re-acquire the nearest
        # unseen identity the box may belong to; else spawn only if that role has nobody unseen.
        taken = {d for d in det_of if d is not None}
        for j, d in enumerate(boxes.index):
            if d in taken:
                continue
            role = broles[j]
            near = min((np.linalg.norm(i["pos"] - bpos[j]) for k, i in enumerate(ids)
                        if det_of[k] is not None), default=np.inf)
            if near <= GATE_M:
                unclaimed += 1
                continue
            free = [k for k, i in enumerate(ids)
                    if det_of[k] is None and cost_of(i["role"], role, areas[j]) < BIG]
            reach = []
            for k in free:
                gone = frame - (ids[k]["seen"] if ids[k]["seen"] is not None else start - 50)
                dist = np.linalg.norm(ids[k]["pos"] - bpos[j])
                if dist <= REACQUIRE_M + VMAX_M * gone:
                    reach.append((dist + cost_of(ids[k]["role"], role, areas[j]), k))
            if reach:
                _, k = min(reach)
                ids[k]["pos"], ids[k]["vel"], ids[k]["seen"] = bpos[j].copy(), np.zeros(2), frame
                ids[k]["occluder"] = None
                det_of[k] = d
                reacquired += 1
            elif (role in CAP and count(role) < CAP[role]
                  and not any(ids[k]["role"] == role for k in free)):
                ids.append({"role": role, "pos": bpos[j].copy(), "vel": np.zeros(2),
                            "seen": frame, "source": "spawn@{}".format(frame)})
                det_of.append(d)
                spawned.append((frame, role))
            else:
                unclaimed += 1

        # Occlusion: an identity that just lost its box next to someone seen now follows him.
        for k, i in enumerate(ids):
            if det_of[k] is not None:
                i["occluder"] = None
                continue
            if i["seen"] is None or frame - i["seen"] > OCCLUSION_MAX:
                i["occluder"] = None
                continue
            if frame - i["seen"] == 1 and not i["role"].startswith("GK_"):  # keepers stay put
                near = [(np.linalg.norm(o["pos"] - i["pos"]), q) for q, o in enumerate(ids)
                        if q != k and det_of[q] is not None]
                near = [x for x in near if x[0] <= OCCLUSION_M]
                i["occluder"] = min(near)[1] if near else None
            q = i.get("occluder")
            if q is not None and det_of[q] is not None:
                i["pos"], i["vel"] = ids[q]["pos"].copy(), ids[q]["vel"].copy()

        M = matrices[frame]
        for k, i in enumerate(ids):
            if det_of[k] is not None:
                state = "observed"
            elif i.get("occluder") is not None:
                state = "occluded"
            else:
                px, py = project(M, i["pos"][0] / length, i["pos"][1] / width)
                state = "lost" if (0 <= px < W and 0 <= py < H) else "out_of_view"
            unseen = frame - i["seen"] if i["seen"] is not None else frame - start + 50
            rows.append((frame, k, i["role"], round(float(i["pos"][0]), 2),
                         round(float(i["pos"][1]), 2), state,
                         int(det_of[k]) if det_of[k] is not None else -1, unseen))

    out = pd.DataFrame(rows, columns=["frame", "identity", "role", "x_m", "y_m", "state", "det",
                                      "unseen_frames"])
    out.to_csv(os.path.join(OUT_DIR, "identities_{}.csv".format(tag)), index=False)

    # Checks. Team consistency against Metrica uses only boxes paired 1-to-1 with a Metrica point.
    out_team = out.role.map(team)
    obs = out.assign(team=out_team)[out.det >= 0].merge(
        det[["matched_team"]], left_on="det", right_index=True)
    obs = obs[obs.team.isin(["H", "A"]) & obs.matched_team.notna()]
    counts = out.assign(team=out_team)[out_team.isin(["H", "A"])].groupby(
        ["frame", "team"]).size().unstack(fill_value=0)
    summary = {
        "frames": [start, end], "kickoff_rules": args.kickoff, "init": init,
        "roster": {k: [i["source"] for i in ids if i["role"] == k] for k in CAP},
        "spawned_after_first_frame": spawned,
        "identities_per_team_max": counts.max().to_dict(),
        "states": out.groupby(["role", "state"]).size().unstack(fill_value=0).to_dict("index"),
        "unclaimed_boxes": unclaimed,
        "reacquired_by_second_pass": reacquired,
        "max_identities_on_one_box": int(out[out.det >= 0].groupby(["frame", "det"]).size().max()),
        "team_agrees_with_metrica_on_paired_boxes": round(float((obs.team == obs.matched_team).mean()), 4),
        "settings": {"cap": CAP, "gate_m": GATE_M, "vmax_m_per_frame": VMAX_M,
                     "gate_max_m": GATE_MAX_M, "reacquire_m": REACQUIRE_M,
                     "keeper_swap_m": KEEPER_SWAP_M},
    }
    with open(os.path.join(OUT_DIR, "identities_{}.json".format(tag)), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))

    if not args.no_video:
        render(tag, start, n, out, det, met, matrices, length, width)


def render(tag, start, n, out, det, met, matrices, length, width):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils import draw_pitch

    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    _, ball = read_atd(raw_path, start, n)
    ident = dict(list(out.groupby("frame")))
    metf = dict(list(met.groupby("frame")))

    fig, axes = plt.subplots(1, 2, figsize=(19.2, 6.4), dpi=100)
    for ax, title in zip(axes, ("Metrica as supplied", "Fixed roster (detector + rules)")):
        draw_pitch(ax, length, width)
        ax.invert_yaxis()  # far touchline (y = 0) on top, as in the camera
        ax.set_title(title)
    marks = [(axes[0].scatter([], [], s=80, edgecolors="black", zorder=3),
              axes[0].scatter([], [], s=60, color="white", edgecolors="black", zorder=4)),
             (axes[1].scatter([], [], s=80, edgecolors="black", zorder=3),
              axes[1].scatter([], [], s=60, color="white", edgecolors="black", zorder=4))]
    stamp = fig.suptitle("")
    fig.canvas.draw()
    size = fig.canvas.get_width_height()
    pitch_out = encoder(os.path.join(OUT_DIR, "pitch2d_{}.mp4".format(tag)), size)
    cam_out = encoder(os.path.join(OUT_DIR, "identities_{}.mp4".format(tag)))

    for i, image in enumerate(decode(video_path, start, n)):
        frame = start + i
        m, d = metf[frame], ident[frame]
        b = ball[i] * [length, width]
        marks[0][0].set_offsets(m[["X", "Y"]].values)
        marks[0][0].set_facecolors([COLOUR[t] for t in m.team])
        marks[0][0].set_linewidths([2.5 if s != "confirmed" else 0.5 for s in m.state])
        # Officials are tracked (so they never become players) but not drawn; nor is anyone unseen
        # for over STALE_FRAMES unless he is following the player who hides him.
        p = d[(d.role != "other") & ((d.state == "observed") | (d.state == "occluded")
                                     | (d.unseen_frames <= STALE_FRAMES))]
        marks[1][0].set_offsets(p[["x_m", "y_m"]].values)
        marks[1][0].set_facecolors([COLOUR[team(r)] if s == "observed" else "none"
                                    for r, s in zip(p.role, p.state)])
        marks[1][0].set_edgecolors([COLOUR[team(r)] if s != "observed" else "black"
                                    for r, s in zip(p.role, p.state)])
        marks[1][0].set_linewidths([2.5 if s != "observed" else 0.5 for s in p.state])
        for _, ball_mark in marks:
            ball_mark.set_offsets(b[None] if np.isfinite(b[0]) else np.empty((0, 2)))
        teams = d.role.map(team)
        nh, na = int((teams == "H").sum()), int((teams == "A").sum())
        stamp.set_text("video frame {}  t+{:.2f}s   left: thick edge = Metrica point not confirmed"
                       "   right: hollow = lost / out of view   roster H {} A {}"
                       .format(frame, i / 25, nh, na))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pitch_out.stdin.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR).tobytes())

        M = matrices[frame]
        for r in d.itertuples():
            c = BGR[team(r.role)]
            if r.det >= 0:
                box = det.loc[r.det]
                p1, p2 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
                cv2.rectangle(image, p1, p2, c, 2)
                cv2.putText(image, label(r.role, r.identity), (p1[0], p1[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            else:
                px, py = project(M, r.x_m / length, r.y_m / width)
                if 0 <= px < W and 0 <= py < H:
                    cv2.circle(image, (int(px), int(py)), 12, c, 2)
                    cv2.putText(image, label(r.role, r.identity) + ("?occ" if r.state == "occluded" else "?"),
                                (int(px) + 12, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1,
                                cv2.LINE_AA)
        if np.isfinite(ball[i, 0]):
            px, py = project(M, ball[i, 0], ball[i, 1])
            cv2.circle(image, (int(px), int(py)), 5, (0, 230, 255), -1)
        states = d.state.value_counts().to_dict()
        header(image, ["video frame {}  t+{:.2f}s   fixed roster: H {}  A {}  officials {}".format(
                           frame, i / 25, nh, na, int((d.role == "other").sum())),
                       "box = identity seen (red H, blue A, yellow official); G = goalkeeper",
                       "ring + '?' = not seen (held/predicted); '?' after id + 'occ' = hidden behind another",
                       "observed {}  lost {}  out of view {}".format(
                           states.get("observed", 0), states.get("lost", 0),
                           states.get("out_of_view", 0))])
        cam_out.stdin.write(image.tobytes())

    for proc in (pitch_out, cam_out):
        proc.stdin.close()
        proc.wait()
    plt.close(fig)


if __name__ == "__main__":
    main()
