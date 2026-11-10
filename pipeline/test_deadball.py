"""End-to-end check of pipeline/deadball.py on synthetic matches (no real data needed).

Writes 3 small fake matches with the convert_dfl.py layout to a temporary folder (never under
data/), then runs features, the leave-one-match-out training, the grid, the saving of the
<match>_deadball.npz files and of the final model, and applies the saved model again.

    python pipeline/test_deadball.py      (or: pytest pipeline/test_deadball.py)
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fake_match(path, seed, half_frames=3000, n_players=24):
    rng = np.random.default_rng(seed)
    F = 2 * half_frames
    alive = np.zeros(F, np.int8)
    i = 0
    while i < F:  # alternate alive 20-60 s / dead 4-25 s
        L = int(rng.uniform(20, 60) * 25)
        alive[i:i + L] = 1
        i += L + int(rng.uniform(4, 25) * 25)
    team = np.array(["H"] * 12 + ["A"] * 12)[:n_players]
    pos = np.empty((F, n_players, 2))
    pos[0] = rng.uniform([10, 5], [95, 63], (n_players, 2))
    v = np.zeros((n_players, 2))
    for t in range(1, F):
        sigma = 0.6 if alive[t] else 0.15
        v = 0.95 * v + rng.normal(0, sigma, v.shape) * 0.2
        pos[t] = np.clip(pos[t - 1] + v / 25, [-2, -2], [107, 70])
    pos[:, 22:] = np.nan  # substitutes, never on the pitch
    ball = np.zeros((F, 3))
    k = 0
    for t in range(F):
        if alive[t]:
            if t % 75 == 0:
                k = rng.integers(0, 22)
            ball[t, :2] = pos[t, k] + rng.normal(0, 0.5, 2)
        else:
            ball[t, :2] = (0.0, 0.0) if (t // 500) % 2 else (52.5, 34.0)
    possession = np.where((np.arange(F) // 400) % 2 == 0, "H", "A")
    possession = np.where(alive == 1, possession, "")
    scale = np.array([105.0, 68.0])
    np.savez_compressed(path, frame=np.arange(F), half=np.repeat([1, 2], half_frames),
                        players=(pos / scale).astype(np.float32), team=team,
                        keeper=np.zeros(n_players, bool),
                        referees=np.full((F, 1, 2), np.nan, np.float32),
                        ball=np.c_[ball[:, :2] / scale, ball[:, 2]].astype(np.float32),
                        possession=possession.astype("<U1"), alive=alive)


def test_end_to_end():
    tmp = tempfile.mkdtemp(prefix="deadball_test_")
    conv, out, models = (os.path.join(tmp, s) for s in ("converted", "outputs", "models"))
    os.makedirs(conv)
    os.environ.update(DFL_CONVERTED=conv, DFL_OUT=out, DEADBALL_MODELS=models)
    for s in range(3):
        fake_match(os.path.join(conv, "DFL-MAT-FAKE{}.npz".format(s)), s)
    sys.path.insert(0, HERE)
    import deadball
    import dfl_paths

    d = dict(np.load(dfl_paths.match_paths()[0]))
    ball, p, t = deadball.degraded_inputs(d, 0)
    X = deadball.features(d, ball, p, t)
    names = deadball.feature_names()
    assert X.shape == (len(d["half"]), len(names)) and X.dtype == np.float32
    assert len(set(names)) == len(names)
    assert not np.isinf(X).any(), "inf in features"
    assert np.isfinite(X).mean() > 0.5

    res = deadball.run(stride=5, windows=(1, 91), thresholds=(0.4, 0.5), n_rounds=30, verbose=False)
    for path in dfl_paths.match_paths():
        name = dfl_paths.match_name(path)
        z = np.load(dfl_paths.out_path(name, "deadball"))
        F = len(np.load(path)["half"])
        assert z["p_dead"].shape == (F,) and z["p_dead"].dtype == np.float32
        assert 0 <= z["p_dead"].min() and z["p_dead"].max() <= 1
        assert str(z["variant"])
    assert os.path.exists(res["final"]) and os.path.exists(os.path.join(models, "deadball_meta.json"))
    assert res["importance"] is not None and len(res["importance"]) == len(names)
    s = res["summary"]
    assert {"ball missing (baseline)", "detector", "oracle D (gt)"} <= set(s.index)
    # the synthetic dead phases are easy: the detector must learn them
    assert s.loc["detector", "frame_acc"] > 0.8, s

    pd_, dead, meta = deadball.predict_p_dead(ball, p, t, d["half"])
    assert pd_.shape == (len(d["half"]),) and dead.dtype == bool and meta["features"] == names
    print(s.round(3).to_string())
    print(res["importance"].head(10).round(4).to_string())
    print("OK, temporary folder:", tmp)


if __name__ == "__main__":
    test_end_to_end()
