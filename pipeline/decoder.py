"""A1: exact segment decoder for H / A / D, tuned and scored on the 7 DFL matches.

Why a segment decoder
---------------------
spellF1 scores possessions, not frames: every maximal run of H or A is one object, matched to the
true ones by temporal IoU. One spurious flip inside a true possession splits it in two pieces, and
often neither piece reaches IoU 0.5 any more: one false positive and one miss for a few frames of
error. A per-frame classifier, or a majority vote over a window, has no notion of that. The vote
only makes flips rarer; it still flips wherever the local majority wobbles, and it moves every true
boundary by up to half the window.

decode() instead chooses the whole label sequence at once:

    maximise  sum_f log p_f(y_f)  -  sum_{changes f} [ switch_cost(y_{f-1}, y_f) + context_cost[f, y_{f-1}, y_f] ]
    subject to every run of label k lasting at least min_dur[k] frames.

The first term is the per-frame evidence; the second is a price per SEGMENT, i.e. per possession or
stoppage started, exactly the unit spellF1 counts. A flip is accepted only when the evidence for it,
summed over its whole duration, pays for the two changes it creates; a real change of possession
supported by enough frames costs one change and is placed where the evidence turns, not half a
window later. The minimum duration forbids the very short runs that can never be matched (a run of
a few frames has IoU ~0 with any real possession) and acts as a hard prior on possession length.
This is a hidden semi-Markov model with a duration floor (Viterbi on segments), solved exactly.

State space and cost
--------------------
States are (label k, length of the current run truncated at L_k = min_dur[k]): L_H + L_A + L_D
states. In that chain the states (k, 2..L_k-1) are a deterministic shift -- a run that entered
label k at frame g is in state (k, f-g+1) at frame f with score Start_k(g) + sum logp[g+1..f, k].
So the chain collapses onto two values per label:

    Start_k(f) = logp[f, k] + max_{j != k} ( Full_j(f-1) - switch_cost[j, k] - context_cost[f, j, k] )
    Full_k(f)  = max( Full_k(f-1) + logp[f, k],                          # stay, run already >= L_k
                      Start_k(f-L_k+1) + sum logp[f-L_k+2..f, k] )       # run just reached L_k

with cumulative sums for the inner sum. Only Full states may change label (the run is long
enough). This is the same optimum as the O(F*K*Lmin) DP over (label, truncated length), at
O(F*K^2) = 9 operations per frame; brute force over all 3^n sequences confirms it in
test_decoder.py. ~150k frames decode in about a second in pure Python.

Halves: the sequence is decoded independently in each block of constant `half`, so a change at the
half-time cut costs nothing and no run spans it. A block shorter than min_dur[k] uses min_dur[k] =
block length (so a decoding always exists). The last run of a block must also reach min_dur.

Determinism: on exact ties H < A < D in label order, and "stay" beats "change" (fewer segments).

    python pipeline/decoder.py            # the DFL evaluation below; prints aggregate tables only
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LABELS = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2
K = 3
NEG = float("-inf")
EPS = 1e-4

# -- evaluation settings (DFL, conditions of dfl_sim.py) ---------------------------------------
SEEDS = (0, 1)
TOUCH_T = 3.0            # touch threshold, m/s (one of dfl_method4's values)
SWITCH_COSTS = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
MIN_DURS = ((12, 25), (25, 25), (50, 25), (25, 50))   # (H and A, D) in frames; D >= 1 s (Lang et al. 2022)
REFINE_WINDOW = 12
TOUCH_BONUS = 0.5        # context variant: a change onto the toucher's team at a touch costs 50% less
PRESS_NEAR_M, PRESS_FAR_M = 1.5, 5.0   # pressure variant: new team's nearest player to the ball ...
PRESS_CHEAP, PRESS_DEAR = 0.5, 0.5     # ... within 1.5 m: change 50% cheaper; none within 5 m: 50% dearer
NET_WEIGHT = 1.0         # weight of C1's p_home (<match>_net.npz) in log-odds, when the file exists
NET_WEIGHTS = tuple(float(x) for x in (os.environ.get("DFL_NET_WEIGHTS") or "0,0.5,1").split(","))   # ... chosen with the other parameters when the file exists (0 = network off)


# -- the decoder -------------------------------------------------------------------------------

def _switch_matrix(switch_cost):
    sw = np.asarray(switch_cost, dtype=np.float64)
    if sw.ndim == 0:
        sw = np.full((K, K), float(sw))
    if sw.shape != (K, K):
        raise ValueError("switch_cost must be a scalar or a 3x3 matrix (from, to)")
    sw = sw.copy()
    np.fill_diagonal(sw, 0.0)
    return sw


def _min_dur(min_dur):
    if isinstance(min_dur, dict):
        out = [min_dur.get(k, 1) for k in LABELS]
    elif np.ndim(min_dur) == 0:
        out = [min_dur] * K
    else:
        out = list(min_dur)
    if len(out) != K:
        raise ValueError("min_dur: an int, {'H':..,'A':..,'D':..} or 3 values")
    return [max(1, int(v)) for v in out]


def half_blocks(half, n):
    """[(start, stop)] blocks of constant `half` (the whole range when half is None)."""
    if half is None:
        return [(0, n)]
    half = np.asarray(half)
    if len(half) != n:
        raise ValueError("half has {} frames, logp {}".format(len(half), n))
    cut = np.flatnonzero(half[1:] != half[:-1]) + 1
    bounds = np.r_[0, cut, n]
    return [(int(a), int(z)) for a, z in zip(bounds[:-1], bounds[1:]) if z > a]


def _decode_block(lp, sw, L, ctx):
    """Exact decoding of one block. lp (n,3) float64, sw 3x3 list, L list of 3, ctx (n,3,3) or None."""
    n = len(lp)
    L = [min(l, n) for l in L]
    cs = np.vstack([np.zeros((1, K)), np.cumsum(lp, axis=0)]).tolist()  # cs[f] = sum lp[:f]
    lpl = lp.tolist()
    ctxl = ctx.tolist() if ctx is not None else None
    start = [[NEG] * n for _ in range(K)]          # Start_k(f)
    start_from = [bytearray(n) for _ in range(K)]  # previous label + 1 (0 = block start)
    from_start = [bytearray(n) for _ in range(K)]  # Full_k(f) reached from Start_k(f-L_k+1)
    full = [NEG] * K
    sw0, sw1, sw2 = sw
    L0, L1, L2 = L
    for f in range(n):
        row = lpl[f]
        # Start_k(f): enter label k at f, from a completed run of another label at f-1.
        if f == 0:
            s = row[:]
            bj = [0, 0, 0]
        else:
            f0, f1, f2 = full
            if ctxl is None:
                c = ((0.0,) * 3,) * 3
            else:
                c = ctxl[f]
            # into H: from A or D (first max wins: A before D)
            a, b = f1 - sw1[0] - c[1][0], f2 - sw2[0] - c[2][0]
            s0, j0 = (a, 2) if a >= b else (b, 3)
            a, b = f0 - sw0[1] - c[0][1], f2 - sw2[1] - c[2][1]
            s1, j1 = (a, 1) if a >= b else (b, 3)
            a, b = f0 - sw0[2] - c[0][2], f1 - sw1[2] - c[1][2]
            s2, j2 = (a, 1) if a >= b else (b, 2)
            s = [s0 + row[0], s1 + row[1], s2 + row[2]]
            bj = [j0, j1, j2]
        for k in range(K):
            start[k][f] = s[k]
            start_from[k][f] = bj[k] if s[k] > NEG else 0
        # Full_k(f): run of k at least L_k long ending at f.
        cf = cs[f + 1]
        new = [NEG] * K
        for k, Lk in ((0, L0), (1, L1), (2, L2)):
            stay = full[k] + row[k]
            g = f - Lk + 1
            comp = start[k][g] + cf[k] - cs[g + 1][k] if g >= 0 else NEG
            if comp > stay:
                new[k] = comp
                from_start[k][f] = 1
            else:
                new[k] = stay
        full = new

    # Backtrack from the best completed run at the last frame (ties: H, A, D).
    k = int(np.argmax(full))
    if full[k] == NEG:
        raise RuntimeError("no feasible decoding (all evidence -inf?)")
    codes = np.empty(n, np.int8)
    f = n - 1
    while f >= 0:
        if from_start[k][f]:
            g = f - L[k] + 1
            codes[g:f + 1] = k
            j = start_from[k][g] - 1
            f, k = g - 1, j
        else:
            codes[f] = k
            f -= 1
    return codes, max(full)


def decode(logp, half=None, switch_cost=16.0, min_dur=None, context_cost=None, return_score=False):
    """Exact segment decoding of per-frame log-probabilities into H / A / D (see module docstring).

    logp          (F, 3) log p_f(label), columns H, A, D. -inf allowed (label forbidden there).
    half          (F,) or None: blocks of constant value are decoded independently (free change,
                  no run across the cut).
    switch_cost   price of one change of label, in the units of logp (nats): a scalar, or a 3x3
                  matrix [from, to] (diagonal ignored) to price e.g. H<->A differently from ->D.
    min_dur       minimum run length in frames: int, {'H': n, 'A': n, 'D': m} or 3 values.
                  Default {'H': 25, 'A': 25, 'D': 25}.
    context_cost  (F, 3, 3) or None: extra cost of changing from i to j with j starting at frame f
                  (negative = cheaper there, e.g. at a touch by j's team).
    Returns labels (F,) '<U1' (and the optimal objective if return_score).
    """
    logp = np.asarray(logp, dtype=np.float64)
    if logp.ndim != 2 or logp.shape[1] != K:
        raise ValueError("logp must be (F, 3)")
    if np.isnan(logp).any():
        raise ValueError("logp contains NaN")
    F = len(logp)
    sw = _switch_matrix(switch_cost).tolist()
    L = _min_dur(25 if min_dur is None else min_dur)
    if context_cost is not None:
        context_cost = np.asarray(context_cost, dtype=np.float64)
        if context_cost.shape != (F, K, K):
            raise ValueError("context_cost must be (F, 3, 3)")
    codes = np.empty(F, np.int8)
    total = 0.0
    for a, z in half_blocks(half, F):
        codes[a:z], sc = _decode_block(logp[a:z], sw, L, None if context_cost is None else context_cost[a:z])
        total += sc
    labels = LABELS[codes]
    return (labels, total) if return_score else labels


def objective(logp, labels, half=None, switch_cost=16.0, context_cost=None):
    """The quantity decode() maximises, for any labelling (no duration check). For tests."""
    logp = np.asarray(logp, dtype=np.float64)
    codes = _codes(labels)
    sw = _switch_matrix(switch_cost)
    total = float(logp[np.arange(len(codes)), codes].sum())
    for a, z in half_blocks(half, len(codes)):
        for f in range(a + 1, z):
            i, j = codes[f - 1], codes[f]
            if i != j:
                total -= sw[i, j] + (0.0 if context_cost is None else context_cost[f, i, j])
    return total


def run_lengths(labels, half=None):
    """[(start, stop, label)] maximal runs inside each half block."""
    codes = _codes(labels)
    out = []
    for a, z in half_blocks(half, len(codes)):
        w = codes[a:z]
        cut = np.flatnonzero(w[1:] != w[:-1]) + 1
        b = np.r_[0, cut, len(w)]
        out += [(a + int(s), a + int(e), LABELS[w[s]]) for s, e in zip(b[:-1], b[1:])]
    return out


def _codes(labels):
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        return labels.astype(np.int64)
    codes = np.full(len(labels), -1, np.int64)
    for k, name in enumerate(LABELS):
        codes[labels == name] = k
    if (codes < 0).any():
        raise ValueError("labels outside H/A/D")
    return codes


# -- boundary refinement -----------------------------------------------------------------------

def refine_boundaries(labels, evidence, window, half=None, team_only=True):
    """Move each change of label to the frame of strongest change evidence within +/- window.

    evidence  (F,) or (F, 3): how much frame f looks like the START of a new run (of the column's
              label, if (F, 3); e.g. a touch by that team). A boundary moves only where some frame
              in its search range has evidence > 0; ties go to the frame nearest the original
              boundary, then the earlier one.
    The search range never reaches the neighbouring boundaries (every run keeps >= 1 frame; note
    that min_dur may then be undercut by at most `window`) nor crosses a half. team_only: refine
    only H<->A changes (D boundaries come from the dead-ball evidence, not from touches).
    """
    labels = np.array(labels, dtype="<U1")
    ev = np.asarray(evidence, dtype=np.float64)
    out = labels.copy()
    for a, z in half_blocks(half, len(labels)):
        w = labels[a:z]
        bounds = [a + int(i) for i in np.flatnonzero(w[1:] != w[:-1]) + 1]
        prev = a  # start of the run left of the current boundary (after earlier moves)
        for n, b in enumerate(bounds):
            nxt = bounds[n + 1] if n + 1 < len(bounds) else z
            left, right = labels[b - 1], labels[b]
            if team_only and (left == "D" or right == "D"):
                prev = b
                continue
            lo, hi = max(prev + 1, b - window), min(nxt - 1, b + window)
            col = ev[lo:hi + 1] if ev.ndim == 1 else ev[lo:hi + 1, int(np.flatnonzero(LABELS == right)[0])]
            if len(col) and col.max() > 0:
                cand = lo + np.flatnonzero(col == col.max())
                new = int(cand[np.lexsort((cand, np.abs(cand - b)))[0]])
                if new > b:
                    out[b:new] = left
                elif new < b:
                    out[new:b] = right
                b = new
            prev = b
    return out


# -- per-frame evidence ------------------------------------------------------------------------

def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def frame_logp(near, dmin, ball_missing=None, touch=None, rule_labels=None, p_dead=None,
               dead_mask=None, p_home_ext=None, near_conf=0.75, near_scale_m=3.0, rule_conf=0.8,
               touch_conf=0.95, ext_weight=1.0, dead_if_missing=0.9, dead_if_seen=0.02,
               p_home_net=None, net_weight=1.0, ball_weight=1.0):
    """(F, 3) log p(H), log p(A), log p(D) from whatever evidence is available. Pure numpy.

    near, dmin    team ('H'/'A'/'') and distance (m) of the player nearest the ball
                  (dfl_control.nearest). Confidence near_conf at 0 m, fading to 0.5 with
                  exp(-d / near_scale_m).
    ball_missing  (F,) bool: no H/A evidence at all there (logit 0, uniform between H and A).
                  It is NOT dead-ball evidence. Default: where near == ''.
    touch         (F,) bool (dfl_control.touches): the nearest team touched it, touch_conf.
    rule_labels   (F,) H/A/D from a rule (e.g. the touch rule): rule_conf towards its team.
    p_home_ext    (F,) P(home | alive) from B2's <match>_rule.npz, added in log-odds x ext_weight.
    p_home_net    (F,) P(home | alive) from C1's <match>_net.npz (network on the players), added in
                  log-odds x net_weight. Unlike the other H/A channels it is kept where the ball is
                  missing: telling the team without the ball is what the network is for.
    p_dead        (F,) P(dead) from B1's <match>_deadball.npz. If None: from dead_mask (the ball
                  missing in the Metrica-like ball, as the baseline does): dead_if_missing where
                  missing, dead_if_seen elsewhere; if dead_mask is None too, from ball_missing.
    The independent channels add in log-odds; every probability is clipped to [EPS, 1-EPS].
    """
    near = np.asarray(near)
    F = len(near)
    dmin = np.asarray(dmin, dtype=np.float64)
    sign = np.where(near == "H", 1.0, np.where(near == "A", -1.0, 0.0))
    if ball_missing is None:
        ball_missing = sign == 0
    ball_missing = np.asarray(ball_missing, bool) | ~np.isfinite(dmin)
    conf = 0.5 + (near_conf - 0.5) * np.exp(-np.where(np.isfinite(dmin), dmin, 1e9) / near_scale_m)
    z = sign * _logit(conf)
    if touch is not None:
        z = z + np.asarray(touch, bool) * sign * _logit(touch_conf)
    if rule_labels is not None:
        r = np.asarray(rule_labels)
        z = z + np.where(r == "H", 1.0, np.where(r == "A", -1.0, 0.0)) * _logit(rule_conf)
    z = ball_weight * z   # peso comune degli indizi dalla palla (piu' vicino, tocco, regola)
    if p_home_ext is not None:
        z = z + ext_weight * _logit(np.nan_to_num(np.asarray(p_home_ext, np.float64), nan=0.5))
    z = np.where(ball_missing, 0.0, z)
    if p_home_net is not None:
        z = z + net_weight * _logit(np.nan_to_num(np.asarray(p_home_net, np.float64), nan=0.5))
    p_home = np.clip(1.0 / (1.0 + np.exp(-z)), EPS, 1 - EPS)
    if p_dead is None:
        m = ball_missing if dead_mask is None else np.asarray(dead_mask, bool)
        p_dead = np.where(m, dead_if_missing, dead_if_seen)
    p_dead = np.clip(np.nan_to_num(np.asarray(p_dead, np.float64), nan=0.5), EPS, 1 - EPS)
    if len(p_dead) != F:
        raise ValueError("p_dead has {} frames, near {}".format(len(p_dead), F))
    return np.log(np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1))


def touch_context(touch, near, F, bonus):
    """context_cost (F,3,3): an H<->A change onto the toucher's team at a touch frame costs -bonus."""
    ctx = np.zeros((F, K, K))
    touch = np.asarray(touch, bool)
    ctx[touch & (near == "H"), A, H] = -bonus
    ctx[touch & (near == "A"), H, A] = -bonus
    return ctx


def pressure_context(d_home, d_away, switch_cost, near_m=PRESS_NEAR_M, far_m=PRESS_FAR_M,
                     cheap=PRESS_CHEAP, dear=PRESS_DEAR):
    """context_cost (F,3,3) from pressure: dynamic transitions in the spirit of PathCRF.

    d_home, d_away: distance (m) of each team's nearest player to the ball (inf/NaN if unknown).
    A change i -> j between H and A costs (1 - cheap) * switch_cost when a player of j (the team
    taking over, the opponent of i) is within near_m of the ball, (1 + dear) * switch_cost when
    none is within far_m, switch_cost otherwise. Changes to / from D are unaffected. Where the ball
    is unknown (both distances non-finite) nothing changes.
    """
    dh = np.nan_to_num(np.asarray(d_home, np.float64), nan=np.inf)
    da = np.nan_to_num(np.asarray(d_away, np.float64), nan=np.inf)
    known = np.isfinite(dh) | np.isfinite(da)
    ctx = np.zeros((len(dh), K, K))
    for i, j, dj in ((A, H, dh), (H, A, da)):
        ctx[known & (dj < near_m), i, j] = -cheap * switch_cost
        ctx[known & (dj > far_m), i, j] = dear * switch_cost
    return ctx


# -- DFL evaluation (conditions of dfl_sim.py) -------------------------------------------------

def _bench():
    for p in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    import dfl_paths, dfl_bench, dfl_sim, dfl_control  # noqa: E401
    return dfl_paths, dfl_bench, dfl_sim, dfl_control


def _external(dfl_paths, match, F):
    """p_dead from B1, p_home from B2 and p_home_net from C1 if their files exist with the right
    length (C1's <match>_net.npz stores it as p_home; here it is keyed p_home_net). p_raw: B1's
    unfiltered P(dead), when the file has it (to let A1 choose the median filter)."""
    out = {}
    for kind, key, name in (("deadball", "p_dead", "p_dead"), ("deadball", "p_raw", "p_raw"),
                            ("rule", "p_home", "p_home"), ("net", "p_home", "p_home_net")):
        path = dfl_paths.out_path(match, kind)
        if os.path.isfile(path):
            z = np.load(path)
            if key in z.files and len(z[key]) == F:
                out[name] = z[key].astype(np.float64)
    return out


def evidence(players, team, ball):
    """What the decoder reads from a tracking with one ball (NaN where missing), the same on a
    simulated DFL match and on the provided one: dead (ball missing), baseline labels (nearest
    +/- 50), near, dist, touches, rule labels (touch rule), distance of each team from the ball."""
    _, dfl_bench, _, dfl_control = _bench()
    from baseline import smooth
    team = np.broadcast_to(team, players.shape[:2])
    dead = ~np.isfinite(ball[:, 0])
    near, dist = dfl_control.nearest(players, team, ball)
    codes = np.where(near == "H", 0, np.where(near == "A", 1, -1))
    w50 = smooth(np.where(dead, -1, codes), 50).copy()
    w50[dead] = "D"
    tc = dfl_control.touches(ball, dist, TOUCH_T)
    rule = dfl_control.run_rule(near, dist, dead, None, 0, tc)
    dd = np.linalg.norm((players - ball[:, None]) * dfl_bench.SCALE, axis=2)
    dd = np.where(np.isnan(dd), np.inf, dd)
    d_home = np.where(team == "H", dd, np.inf).min(1)
    d_away = np.where(team == "A", dd, np.inf).min(1)
    return {"dead": dead, "baseline": w50, "near": near, "dist": dist, "touch": tc, "rule": rule,
            "ball_missing": dead, "d_home": d_home, "d_away": d_away}


def simulate(d, seed, level=None):
    """Evidence for one DFL match seen as the provided tracking would give it (dfl_sim.inputs)."""
    s = _bench()[2].inputs(d, seed, level)
    ev = evidence(s["players"], s["team"], s["ball"])
    if os.environ.get("DFL_BALL", "metrica") != "metrica":  # tocchi non cercati vicino a Metrica / ai voli
        import pandas as pd
        near = pd.Series(s["src"] == 2).rolling(5, center=True, min_periods=1).max().to_numpy().astype(bool)
        ev["touch"] = ev["touch"] & ~near
        ev["rule"] = _bench()[3].run_rule(ev["near"], ev["dist"], ev["dead"], None, 0, ev["touch"])
    return ev


def match_logp(ev, ext=None, net_weight=NET_WEIGHT):
    ext = ext or {}
    return frame_logp(ev["near"], ev["dist"], ball_missing=ev["ball_missing"], touch=ev["touch"],
                      rule_labels=ev["rule"], p_dead=ext.get("p_dead"), dead_mask=ev["dead"],
                      p_home_ext=ext.get("p_home"), p_home_net=ext.get("p_home_net"),
                      net_weight=net_weight)


def grid():
    return [(sc, {"H": ha, "A": ha, "D": dd}) for sc in SWITCH_COSTS for ha, dd in MIN_DURS]


def param_name(sc, md, w=None):
    """w: weight of C1's network, in the name only when it is one of the choices."""
    return "sw{:g}_min{}/{}".format(sc, md["H"], md["D"]) + ("" if w is None else "_net{:g}".format(w))


def predict(ev, logp, half, sc, md, extras=True):
    """Decoder labels for one parameter set (+ refined and touch-context variants if extras)."""
    out = {"decoder": decode(logp, half, sc, md)}
    if extras:
        tev = np.zeros((len(logp), K))
        tev[ev["touch"] & (ev["near"] == "H"), H] = 1
        tev[ev["touch"] & (ev["near"] == "A"), A] = 1
        out["decoder+refine"] = refine_boundaries(out["decoder"], tev, REFINE_WINDOW, half)
        out["decoder+touch_ctx"] = decode(logp, half, sc, md, touch_context(ev["touch"], ev["near"], len(logp), TOUCH_BONUS * sc))
        out["decoder+pressure_ctx"] = decode(logp, half, sc, md, pressure_context(ev["d_home"], ev["d_away"], sc))
    return out


def evaluate_dfl(seeds=SEEDS, use_files=True, save=True, verbose=True):
    """Baseline vs touch rule vs decoder on the DFL matches, decoder tuned leave-one-match-out.

    Returns {"summary": DataFrame (method x spellF1/macroF1 mean/min/max over matches),
             "params": {match: chosen param name}, "per_match": DataFrame, "sources": str}.
    Saves <match>_decoded.npz (seed seeds[0], parameters chosen on the other matches) and
    decoder_params.json (the best parameters on all the matches, to apply to another match).
    """
    import json
    import pandas as pd
    dfl_paths, dfl_bench, dfl_sim, _ = _bench()
    level = dfl_sim.level()
    rows, cache, used = [], {}, set()
    paths = dfl_paths.match_paths()
    if use_files and any(os.path.isfile(dfl_paths.out_path(dfl_paths.match_name(p), kind))
                         for p in paths for kind in ("deadball", "rule", "net")):
        # B1 / B2 / C1 computed their files on the inputs of the first seed only: on another seed
        # they would have seen the ball where this simulation has a gap, and inflate the score.
        seeds = seeds[:1]
    has_net = use_files and any(os.path.isfile(dfl_paths.out_path(dfl_paths.match_name(p), "net")) for p in paths)
    G = [(sc, md, w) for sc, md in grid() for w in (NET_WEIGHTS if has_net else (None,))]
    for path in paths:
        match = dfl_paths.match_name(path)
        npz, gt = dfl_bench.load(path)
        d = {k: npz[k] for k in ("players", "team", "ball", "alive", "half")}
        hd = {"half": d["half"]}
        ext = _external(dfl_paths, match, len(gt)) if use_files else {}
        used |= set(ext)
        for seed in seeds:
            ev = simulate(d, seed, level)
            logps = {w: match_logp(ev, ext, NET_WEIGHT if w is None else w) for w in {g[2] for g in G}}
            cache[(match, seed)] = (ev, logps)
            rows.append((match, seed, "baseline nearest +/-50", "", *dfl_bench.score(hd, gt, ev["baseline"])))
            rows.append((match, seed, "touch rule T{:g}".format(TOUCH_T), "", *dfl_bench.score(hd, gt, ev["rule"])))
            for sc, md, w in G:
                pred = decode(logps[w], d["half"], sc, md)
                rows.append((match, seed, "grid", param_name(sc, md, w), *dfl_bench.score(hd, gt, pred)))
        if verbose:
            print(match, "done", flush=True)
    df = pd.DataFrame(rows, columns=["match", "seed", "method", "params", "macro_f1", "spell_f1"])

    grid_df = df[df.method == "grid"].groupby(["match", "params"])["spell_f1"].mean().unstack()
    chosen, final = {}, []
    by_name = {param_name(sc, md, w): (sc, md, w) for sc, md, w in G}
    for path in paths:
        match = dfl_paths.match_name(path)
        pool = dfl_paths.for_choice(list(grid_df.index))
        others = grid_df.loc[[n for n in pool if n != match] or pool]
        means = others.mean(axis=0)
        best = sorted(means.index, key=lambda n: (-means[n], n))[0]  # fixed tie-break by name
        chosen[match] = best
        sc, md, w = by_name[best]
        npz, gt = dfl_bench.load(path)
        half = npz["half"]
        for seed in seeds:
            ev, logps = cache[(match, seed)]
            for method, pred in predict(ev, logps[w], half, sc, md).items():
                final.append((match, seed, method + " (LOMO)", best, *dfl_bench.score({"half": half}, gt, pred)))
                if save and method == "decoder" and seed == seeds[0]:
                    np.savez_compressed(dfl_paths.out_path(match, "decoded"), labels=pred.astype("<U1"),
                                        variant=np.array("A1 decoder {} seed{} files:{} sim:{}".format(
                                            best, seed, "+".join(sorted(used)) or "none", level)))
    df = pd.concat([df, pd.DataFrame(final, columns=df.columns)], ignore_index=True)

    per_match = df[df.method != "grid"].groupby(["method", "match"], sort=False)[["spell_f1", "macro_f1"]].mean()
    summary = per_match.groupby(level="method", sort=False).agg(["mean", "min", "max"]).round(3)
    best_fixed = grid_df.mean(axis=0).sort_values(ascending=False).head(5).round(3)
    sources = "p_dead: {}, p_home B2: {}, p_home C1 net: {}, simulation: {}".format(
        "B1 file" if "p_dead" in used else "fallback (ball missing)",
        "yes" if "p_home" in used else "no",
        "yes (weight chosen among {})".format(NET_WEIGHTS) if "p_home_net" in used else "no", level)
    if save:
        means = grid_df.loc[dfl_paths.for_choice(list(grid_df.index))].mean(axis=0)
        sc, md, w = by_name[sorted(means.index, key=lambda n: (-means[n], n))[0]]
        with open(os.path.join(dfl_paths.models_dir(), "decoder_params.json"), "w") as fh:
            json.dump(dict(switch_cost=sc, min_dur=md, net_weight=NET_WEIGHT if w is None else w,
                           sources=sources, matches=len(paths)), fh, indent=1)
    if verbose:
        print("\nsources:", sources)
        print("\nspellF1 / macroF1 over {} matches x {} seeds (decoder: parameters chosen on the other matches)".format(
            len(paths), len(seeds)))
        print(summary.to_string())
        print("\nparameters chosen (LOMO), count of folds:")
        print(pd.Series(chosen).value_counts().to_string())
        print("\nbest fixed parameters on all matches (in-sample, for reference), mean spellF1:")
        print(best_fixed.to_string())
    return {"summary": summary, "params": chosen, "per_match": per_match, "grid": grid_df,
            "sources": sources, "cache": cache, "by_name": by_name}


def example(result, match=None, seed=None):
    """(gt, {method: labels}, half) for one match, from evaluate_dfl's result, to plot."""
    dfl_paths, dfl_bench, _, _ = _bench()
    paths = {dfl_paths.match_name(p): p for p in dfl_paths.match_paths()}
    match = match or sorted(result["params"])[0]
    seed = SEEDS[0] if seed is None else seed
    npz, gt = dfl_bench.load(paths[match])
    ev, logps = result["cache"][(match, seed)]
    best = result["params"][match]
    sc, md, w = result["by_name"][best]
    preds = {"baseline +/-50": ev["baseline"], "touch rule": ev["rule"]}
    preds["decoder " + best] = decode(logps[w], npz["half"], sc, md)
    return match, gt, preds


if __name__ == "__main__":
    evaluate_dfl()
