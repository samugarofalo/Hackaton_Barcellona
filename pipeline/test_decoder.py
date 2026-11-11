"""Tests for decoder.py on synthetic data only (no match data needed).

    python pipeline/test_decoder.py        # or: python -m pytest pipeline/test_decoder.py
"""

import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoder as dec  # noqa: E402

LAB = dec.LABELS


def onehot_logp(labels, p=0.8):
    codes = dec._codes(labels)
    q = np.full((len(codes), 3), (1 - p) / 2)
    q[np.arange(len(codes)), codes] = p
    return np.log(q)


def feasible(codes, half, L):
    for a, z in dec.half_blocks(half, len(codes)):
        n = z - a
        Le = [min(l, n) for l in L]
        w = codes[a:z]
        cut = np.flatnonzero(w[1:] != w[:-1]) + 1
        b = np.r_[0, cut, n]
        if any(e - s < Le[w[s]] for s, e in zip(b[:-1], b[1:])):
            return False
    return True


def brute(logp, half, sw, L, ctx=None):
    best = -np.inf
    for seq in itertools.product(range(3), repeat=len(logp)):
        c = np.array(seq)
        if feasible(c, half, L):
            best = max(best, dec.objective(logp, c, half, sw, ctx))
    return best


def test_flicker_removed():
    truth = np.array(["H"] * 200 + ["A"] * 150 + ["D"] * 60 + ["H"] * 200)
    rng = np.random.default_rng(0)
    noisy = truth.copy()
    flip = rng.random(len(truth)) < 0.25
    noisy[flip] = rng.choice(LAB, flip.sum())
    lab = dec.decode(onehot_logp(noisy), switch_cost=10, min_dur=25)
    runs = dec.run_lengths(lab)
    assert [r[2] for r in runs] == ["H", "A", "D", "H"], runs
    assert np.mean(lab == truth) > 0.97
    assert len(dec.run_lengths(noisy)) > 50  # the input really flickers


def test_min_dur_respected():
    rng = np.random.default_rng(1)
    logp = np.log(rng.dirichlet([0.3, 0.3, 0.3], 3000))
    md = {"H": 30, "A": 20, "D": 40}
    lab = dec.decode(logp, switch_cost=0.0, min_dur=md)
    for a, z, k in dec.run_lengths(lab):
        assert z - a >= md[k], (a, z, k)


def test_halves():
    # Evidence says H up to the cut, A after: the change must sit exactly at the cut, for free,
    # even with a prohibitive switch cost.
    half = np.r_[np.ones(100, int), np.full(100, 2)]
    truth = np.array(["H"] * 100 + ["A"] * 100)
    lab, score = dec.decode(onehot_logp(truth), half, switch_cost=1e6, min_dur=10, return_score=True)
    assert (lab == truth).all()
    assert np.isclose(score, dec.objective(onehot_logp(truth), truth, half, 1e6))
    # a run never spans the cut, and min_dur is counted inside each half
    lab = dec.decode(onehot_logp(np.array(["H"] * 200)), half, 5, 10)
    assert [(a, z) for a, z, _ in dec.run_lengths(lab, half)] == [(0, 100), (100, 200)]
    # a half shorter than min_dur is still decodable (min_dur clipped to the half)
    half = np.r_[np.ones(5, int), np.full(50, 2)]
    lab = dec.decode(onehot_logp(np.array(["A"] * 55)), half, 5, 25)
    assert (lab == "A").all()


def test_exact_vs_bruteforce():
    rng = np.random.default_rng(2)
    n_cases = 0
    for n in (1, 2, 3, 5, 7, 8):
        for _ in range(6):
            logp = np.log(rng.dirichlet([0.5, 0.5, 0.5], n))
            L = list(rng.integers(1, 4, 3))
            sw = rng.uniform(0, 2, (3, 3))
            half = None if rng.random() < 0.5 else np.sort(rng.integers(1, 3, n))
            ctx = rng.normal(0, 0.5, (n, 3, 3)) if rng.random() < 0.5 else None
            lab, sc = dec.decode(logp, half, sw, L, ctx, return_score=True)
            ref = brute(logp, half, sw, L, ctx)
            codes = dec._codes(lab)
            assert feasible(codes, half, L)
            assert np.isclose(sc, ref), (n, sc, ref)
            assert np.isclose(dec.objective(logp, lab, half, sw, ctx), ref)
            n_cases += 1
    assert n_cases == 36


def test_deterministic_ties():
    logp = np.zeros((40, 3))  # everything tied
    a = dec.decode(logp, switch_cost=1.0, min_dur=5)
    b = dec.decode(logp.copy(), switch_cost=1.0, min_dur=5)
    assert (a == b).all() and (a == "H").all()


def test_context_cost():
    # Weak evidence for an A stretch: too weak at the constant cost, accepted where the
    # context makes the H->A change cheap.
    truth = np.array(["H"] * 50 + ["A"] * 30 + ["H"] * 50)
    logp = onehot_logp(truth, p=0.4)  # log(0.4/0.3) = 0.29 per frame: 30 frames ~ 8.6
    sc = 6.0
    assert (dec.decode(logp, switch_cost=sc, min_dur=10) == "H").all()
    ctx = np.zeros((len(truth), 3, 3))
    ctx[50, dec.H, dec.A] = -sc
    ctx[80, dec.A, dec.H] = -sc
    assert (dec.decode(logp, switch_cost=sc, min_dur=10, context_cost=ctx) == truth).all()
    # pressure context: opponent close -> cheaper, nobody close -> dearer, D untouched
    pc = dec.pressure_context(np.array([1.0, 9.0, np.nan]), np.array([9.0, 1.0, np.nan]), 10.0)
    assert pc[0, dec.A, dec.H] == -5 and pc[0, dec.H, dec.A] == 5
    assert pc[1, dec.H, dec.A] == -5 and pc[1, dec.A, dec.H] == 5
    assert not pc[2].any() and not pc[:, :, dec.D].any() and not pc[:, dec.D, :].any()


def test_refine_boundaries():
    lab = np.array(["H"] * 50 + ["A"] * 50 + ["D"] * 30 + ["A"] * 20)
    ev = np.zeros(len(lab))
    ev[45] = 1.0   # touch 5 frames before the H->A change
    ev[127] = 1.0  # near the A->D change: ignored (team_only)
    out = dec.refine_boundaries(lab, ev, 10)
    assert dec.run_lengths(out) == [(0, 45, "H"), (45, 100, "A"), (100, 130, "D"), (130, 150, "A")]
    # out of the window: unchanged; per-label evidence: only the destination's column counts
    ev = np.zeros(len(lab)); ev[30] = 1
    assert (dec.refine_boundaries(lab, ev, 10) == lab).all()
    ev3 = np.zeros((len(lab), 3)); ev3[55, dec.H] = 1; ev3[53, dec.A] = 1
    out = dec.refine_boundaries(lab, ev3, 10)
    assert dec.run_lengths(out)[0] == (0, 53, "H")
    # never crosses a half or a neighbouring boundary
    lab = np.array(["H"] * 5 + ["A"] * 5 + ["H"] * 10)
    ev = np.zeros(20); ev[15] = 1
    out = dec.refine_boundaries(lab, ev, 20)
    assert [r[2] for r in dec.run_lengths(out)] == ["H", "A", "H"]


def test_frame_logp():
    near = np.array(["H", "A", "", "H", "A"])
    dmin = np.array([0.5, 0.5, np.inf, 10.0, 0.5])
    lp = dec.frame_logp(near, dmin, ball_missing=np.array([0, 0, 1, 0, 1], bool))
    p = np.exp(lp)
    assert np.allclose(p.sum(1), 1) and np.isfinite(lp).all()
    assert p[0, 0] > p[0, 1] and p[1, 1] > p[1, 0]
    assert np.isclose(p[2, 0], p[2, 1]) and np.isclose(p[4, 0], p[4, 1])  # missing: uniform H/A
    assert p[2, 2] > 0.5 and p[0, 2] < 0.05  # fallback p_dead from missingness
    assert p[0, 0] > p[3, 0] > p[3, 1]       # confidence fades with distance
    lp2 = dec.frame_logp(near, dmin, p_dead=np.array([0, 1, 0.5, 0.5, 0.5]))
    assert np.isfinite(lp2).all()            # clipped, no log(0)
    lp3 = dec.frame_logp(near, dmin, touch=np.array([1, 0, 0, 0, 0], bool), p_home_ext=np.full(5, 0.5))
    assert np.exp(lp3)[0, 0] > np.exp(lp)[0, 0]


def test_speed():
    rng = np.random.default_rng(3)
    logp = np.log(rng.dirichlet([1, 1, 1], 150_000))
    half = np.r_[np.ones(75_000, int), np.full(75_000, 2)]
    t = time.time()
    dec.decode(logp, half, 16.0, {"H": 50, "A": 50, "D": 25})
    dt = time.time() - t
    print("  decode 150k frames: {:.2f} s".format(dt))
    assert dt < 30


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("{} tests passed".format(len(tests)))
