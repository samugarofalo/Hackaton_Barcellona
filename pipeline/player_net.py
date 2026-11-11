"""C1: a small network on the players that tells which team has the ball, also when the ball is lost.

Why. The provided ball is also lost in live play (DATA.md), for seconds at a time: without the
ball there is no touch and no nearest player, so the decoder has no H/A evidence there. But the
players still move like the team in possession (it advances
and spreads out, the other one falls back and closes the ball carrier down). Borghesi et al. 2023
reach 86% possession accuracy on optical tracking with a TCN per object and a permutation-
invariant aggregation (ablation: ~50% from the ball alone, ~85% with ball + team of the objects);
Kim et al. 2023 ("Ball Radar", Set Transformer + Bi-LSTM) reach 85% team possession with the ball
completely hidden, 97% with 80% of it hidden. Implemented here from scratch, small, in PyTorch.

Data (other games, so training here is allowed): the 7 DFL matches as the provided tracking would
give them (dfl_sim.inputs, the simulation shared with B1, B2 and A1, seed TRAIN_SEED, level in
$DFL_SIM): one ball, the provided one, blurred and also missing in live play; players with noise,
drop-outs and tracks that change slot, as the provided slots do. That ball goes in the first ball
slot (called "ours" below); the second one stays empty. Examples = windows of W frames centred on alive frames (GT alive == 1,
possession H/A), every `stride` frames. Target = DFL possession of the centre frame; auxiliary
target = a change of possession within +/- 0.5 s.

Input per object and per step of the window (T = 51 steps, every W/50 frames), up to P_MAX
players + 2 balls (ours, Metrica), padding + mask:
    x, y (metres, rotated so that home attacks +x in each half: the direction comes from the
    teams' centroids at the start of the half, no label), vx, vy, speed (from positions smoothed
    over 5 frames, 3 for the ball), one-hot home / away / ball, Metrica flag, missing flag,
    position relative to the reference ball (ours if seen, else Metrica) and its distance, and a
    flag "no ball at all". A missing ball is an object whose missing flag is on.
Ball masking in training: on top of the simulated gaps, on a share of the examples our ball is
hidden in blocks (lengths dfl_sim.MASK_GAPS, up to x3), on some examples for the whole window, and on
some both balls: the network must also work with no ball.

Architecture (PlayerNet): a TCN shared by all objects (1x1 input projection, 5 residual blocks
with dilations 1..16, kernel 3, GroupNorm, GELU; receptive field 63 steps >= the window) applied to
each object's sequence; the embedding at the centre step; one interaction step (each object sees
the mean of all objects); attention pooling separately over home players, away players, balls
(permutation invariant: the order of the slots does not matter, as with real tracking without
identities); MLP -> 2 logits (H/A) + 1 auxiliary logit (change nearby). encoder="gru" replaces the
TCN with a 1-layer Bi-GRU. W and the channels are hyperparameters compared leave-one-match-out:
W in {100 (4 s: Borghesi 2023, the +/-4 s context of Lang 2022), 250 (10 s: Kim 2023, PathCRF)},
channels in {32, 64, 128}. Final choice by a fixed rule: best mean accuracy with the ball in a
gap over the folds; within TIE (0.5 points) of it, the smallest network.

Training: leave-one-match-out (test 1, early stopping on 1 of the other 6, train on 5), AdamW,
cross-entropy (weight 1 where our ball is seen at the centre, W_GAP elsewhere) + AUX_WEIGHT x
binary cross-entropy of the auxiliary target. Augmentation: swap home/away (with a 180 degree
rotation, so home still attacks +x, label swapped), mirror y. Fixed seeds, deterministic
algorithms (warn only), DataLoader with num_workers=0, CUDA + mixed precision if available. The
final model is trained on all 7 matches for the median of the best epochs of the chosen config
and saved to models_dir(): player_net.pt + player_net.json.

Inference: predict_p_home(...) -> P(home) per frame (sliding window, batches, every `stride`
frames and interpolated in log-odds, 0.5 on dead frames); works on real tracking too (slots
without identity, NaN, team per frame), on CPU.

Outputs: <match>_net.npz (p_home (F,) float32, out-of-fold, on every frame: whether the ball is
dead is B1's and the decoder's business; variant) via dfl_paths.out_path.
Evaluation (aggregates only): accuracy of the possession per match on alive frames, split between
our ball seen and our ball in a gap, against the nearest player to the ball of the A1 chain on the
same frames; the same with no ball at all; spellF1 of the A1 decoder with and without p_home.

    python pipeline/player_net.py [--quick] [--epochs N] [--device cpu|cuda] [--grid-w 100 250]
                                  [--grid-ch 32 64 128] [--no-downstream]
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import decoder as dec  # noqa: E402
import dfl_paths  # noqa: E402
from dfl_bench import SCALE, load, score  # noqa: E402
from dfl_control import nearest  # noqa: E402
from dfl_degrade import FPS  # noqa: E402
from dfl_sim import MASK_GAPS, inputs, level  # noqa: E402

# -- parameters ----------------------------------------------------------------------------------
P_MAX = 24               # player slots per window (22 on the pitch + substitutions / fragments)
N_BALL = 2               # ours, Metrica
STEPS_HALF = 25          # T = 51 steps per window, every W / 50 frames
HALF_X, HALF_Y = 52.5, 34.0
V_NORM, D_NORM = 10.0, 20.0
FEATURES = ("x", "y", "vx", "vy", "speed", "home", "away", "ball", "metrica", "missing",
            "dx_ball", "dy_ball", "dist_ball", "no_ball")
C_IN = len(FEATURES)
TRAIN_SEED = 0           # degradation seed of the training inputs (= decoder.SEEDS[0])
AUX_FRAMES = int(0.5 * FPS)
TIE = 0.005              # 0.5 points of accuracy: within it, the smallest network wins
GRID_W = (100, 250)
GRID_CH = (32, 64, 128)
DEFAULTS = dict(W=100, channels=64, layers=5, kernel=3, encoder="tcn", dropout=0.1,
                lr=2e-3, weight_decay=1e-4, batch=256, epochs=15, patience=3, stride=5,
                aux_weight=0.3, w_gap=0.5, mask_share=0.5, hide_all=0.15, hide_both=0.10,
                seed=0)
QUICK = dict(epochs=3, stride=20, folds=2, grid_w=(100,), grid_ch=(64,))
PRED_STRIDE = 2          # out-of-fold predictions every 2 frames (a 1-frame offset is free on spellF1)


def models_dir():
    path = os.environ.get("PLAYER_NET_MODELS") or (
        "/kaggle/working/models" if dfl_paths.on_kaggle() else os.path.join(ROOT, "models"))
    os.makedirs(path, exist_ok=True)
    return path


def cfg_name(cfg):
    return "W{}_ch{}{}".format(cfg["W"], cfg["channels"], "" if cfg["encoder"] == "tcn" else "_" + cfg["encoder"])


def setup(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def pick_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def offsets(W):
    step = max(1, int(W) // (2 * STEPS_HALF))
    return np.arange(-STEPS_HALF, STEPS_HALF + 1) * step, step


# -- simulated inputs ----------------------------------------------------------------------------

def degraded_inputs(d, seed):
    """One DFL match as the provided tracking would give it (dfl_sim.inputs, as decoder.simulate).
    Returns players (F,S,2), team (F,S), ours (F,2) = the provided ball, metrica (F,2) = empty
    second ball slot, normalised 0-1, dead (F,) = no frame is skipped."""
    s = inputs(d, seed)
    return dict(players=s["players"], team=s["team"], ours=s["ball"],
                metrica=np.full_like(s["ball"], np.nan), dead=np.zeros(len(s["ball"]), bool))


def chain_nearest(inp):
    """Team of the player nearest the ball: the evidence the decoder has, for the comparison."""
    near, _ = nearest(inp["players"], inp["team"], inp["ours"])
    return near


# -- features ------------------------------------------------------------------------------------

def team_codes(team, F, P):
    """1 home, 2 away, 0 unknown; team (P,) or (F,P) of 'H'/'A'/''."""
    t = np.broadcast_to(np.asarray(team), (F, P))
    return np.where(t == "H", 1, np.where(t == "A", 2, 0)).astype(np.int8)


def half_bounds(half, F):
    """Per frame: start and end (exclusive) of its half block."""
    lo, hi = np.zeros(F, np.int64), np.zeros(F, np.int64)
    for a, z in dec.half_blocks(half, F):
        lo[a:z], hi[a:z] = a, z
    return lo, hi


def attack_sign(players, team, half, n_frames=25, min_players=5):
    """+1 / -1 per frame: rotate by 180 degrees (sign -1) the halves in which home attacks -x.
    From the teams' mean x over the first n_frames frames of the half with >= min_players of each
    team (at kick-off each team is in its own half). No label used."""
    F, P = players.shape[:2]
    codes = team_codes(team, F, P)
    x = players[:, :, 0]
    fin = np.isfinite(x)
    sign = np.ones(F, np.float32)
    half = np.ones(F, int) if half is None else np.asarray(half)
    for a, z in dec.half_blocks(half, F):
        h = (codes[a:z] == 1) & fin[a:z]
        w = (codes[a:z] == 2) & fin[a:z]
        ok = np.flatnonzero((h.sum(1) >= min_players) & (w.sum(1) >= min_players))[:n_frames]
        if not len(ok):
            continue
        xs = np.where(fin[a:z][ok], x[a:z][ok], 0.0)
        hx = (xs * h[ok]).sum(1) / h[ok].sum(1)
        ax = (xs * w[ok]).sum(1) / w[ok].sum(1)
        sign[a:z] = 1.0 if np.mean(hx - ax) <= 0 else -1.0
    return sign


def _kinematics(xy, half, n):
    """(F,K,2) metres -> (F,K,4) x, y, vx, vy; velocity from positions smoothed over n frames,
    inside each half; NaN position = missing, velocity 0 where it cannot be computed."""
    F, K = xy.shape[:2]
    out = np.full((F, K, 4), np.nan, np.float32)
    out[..., :2] = xy
    for a, z in dec.half_blocks(half, F):
        if z - a < 2:
            out[a:z, :, 2:] = 0
            continue
        sm = pd.DataFrame(xy[a:z].reshape(z - a, -1)).rolling(n, center=True, min_periods=1).mean().to_numpy()
        v = np.gradient(sm, axis=0) * FPS
        out[a:z, :, 2:] = v.reshape(z - a, K, 2)
    out[..., 2:] = np.clip(np.nan_to_num(out[..., 2:]), -40, 40)
    return out


def prepare(players, team, ours=None, metrica=None, half=None, sign=None):
    """Per-frame arrays of one sequence (normalised 0-1 inputs, NaN = missing).
    players (F,P,2), team (P,) or (F,P), ours / metrica (F,2) or None, half (F,) or None."""
    players = np.asarray(players, np.float32)
    F, P = players.shape[:2]
    half = np.ones(F, int) if half is None else np.asarray(half)
    if sign is None:
        sign = attack_sign(players, team, half)
    codes = team_codes(team, F, P)
    pos = (players - 0.5) * SCALE * sign[:, None, None]
    present = np.isfinite(pos[..., 0])
    codes = np.where(present, codes, 0).astype(np.int8)
    nanb = np.full((F, 2), np.nan, np.float32)
    balls = np.stack([nanb if ours is None else np.asarray(ours, np.float32),
                      nanb if metrica is None else np.asarray(metrica, np.float32)], axis=1)
    balls = (balls - 0.5) * SCALE * sign[:, None, None]
    return dict(pk=_kinematics(pos, half, 5), codes=codes, bk=_kinematics(balls, half, 3),
                present=present, half=half, F=F, P=P)


def select_columns(present, centres, lo, hi, p_max=P_MAX, chunk=20000):
    """(n, p_max) player columns per window [lo, hi): the most present ones, -1 = padding."""
    F, P = present.shape
    cs = np.zeros((F + 1, P), np.int32)
    np.cumsum(present, axis=0, out=cs[1:])
    out = np.full((len(centres), p_max), -1, np.int64)
    k = min(p_max, P)
    for s in range(0, len(centres), chunk):
        cnt = cs[hi[s:s + chunk]] - cs[lo[s:s + chunk]]
        order = np.argsort(-cnt, axis=1, kind="stable")[:, :k]
        got = np.take_along_axis(cnt, order, 1)
        out[s:s + chunk, :k] = np.where(got > 0, order, -1)
    return out


class Store:
    """Per-frame arrays of several sequences, concatenated, as tensors on the device. One extra
    all-NaN player column (index `null`) serves the padding slots."""

    def __init__(self, preps, device):
        Pall = max(p["P"] for p in preps)
        self.offset, F = [], 0
        pk, cd, bk = [], [], []
        for p in preps:
            self.offset.append(F)
            F += p["F"]
            a = np.full((p["F"], Pall + 1, 4), np.nan, np.float32)
            a[:, :p["P"]] = p["pk"]
            c = np.zeros((p["F"], Pall + 1), np.int8)
            c[:, :p["P"]] = p["codes"]
            pk.append(a)
            cd.append(c)
            bk.append(p["bk"])
        self.F, self.null = F, Pall
        self.pk = torch.from_numpy(np.concatenate(pk)).to(device)
        self.codes = torch.from_numpy(np.concatenate(cd)).to(device)
        self.bk = torch.from_numpy(np.concatenate(bk)).to(device)
        self.device = device


def _ball_masks(B, T, step, gen, cfg, gaps, dev):
    """Training-time hiding of the balls: blocks of our ball (lengths dfl_sim.MASK_GAPS x 1-3, half of
    them over the centre), our ball for the whole window, or both balls for the whole window."""
    def rand(n):
        return torch.rand(n, generator=gen, device=dev)
    t = torch.arange(T, device=dev)
    m = rand(B) < cfg["mask_share"]
    L = gaps[torch.randint(0, len(gaps), (B,), generator=gen, device=dev)] * \
        torch.randint(1, 4, (B,), generator=gen, device=dev)
    Ls = torch.clamp((L + step - 1) // step, 1, T)
    start_any = (rand(B) * (T + Ls).float()).long() - Ls
    start_c = T // 2 - (rand(B) * Ls.float()).long()
    start = torch.where(rand(B) < 0.5, start_c, start_any)
    blk = (t[None] >= start[:, None]) & (t[None] < (start + Ls)[:, None]) & m[:, None]
    all_o = rand(B) < cfg["hide_all"]
    both = rand(B) < cfg["hide_both"]
    hide_o = blk | all_o[:, None] | both[:, None]
    hide_m = both[:, None].expand(B, T)
    return hide_o, hide_m


def make_batch(store, c, lo, hi, cols, offs, step, train=False, gen=None, cfg=None, gaps=None):
    """Tensors for the network. c, lo, hi (B,) global frame of the centre and bounds of its half;
    cols (B, P_MAX) global player columns (-1 padding); offs (T,) frame offsets.
    Returns x (B,N,T,C_IN), groups (B,N) 0 home 1 away 2 ball, mask (B,N), extra (B,2) balls seen
    at the centre, swap (B,) bool or None (home/away swapped by augmentation)."""
    dev = store.device
    B, T, Pm = c.shape[0], offs.shape[0], cols.shape[1]
    fr = c[:, None] + offs[None, :]
    valid = (fr >= lo[:, None]) & (fr < hi[:, None])
    fr = fr.clamp(0, store.F - 1)
    cx = torch.where(cols < 0, torch.full_like(cols, store.null), cols)
    pk = store.pk[fr[:, :, None], cx[:, None, :]]           # (B,T,Pm,4)
    cd = store.codes[fr[:, :, None], cx[:, None, :]].long()   # (B,T,Pm)
    bk = store.bk[fr]                                        # (B,T,2,4)
    pres_p = torch.isfinite(pk[..., 0]) & valid[:, :, None]
    pres_b = torch.isfinite(bk[..., 0]) & valid[:, :, None]
    cd = cd * pres_p
    swap = None
    if train:
        hide_o, hide_m = _ball_masks(B, T, step, gen, cfg, gaps, dev)
        pres_b = pres_b & torch.stack([~hide_o, ~hide_m], -1)
        swap = torch.rand(B, generator=gen, device=dev) < 0.5
        flipy = torch.rand(B, generator=gen, device=dev) < 0.5
        s = 1.0 - 2.0 * swap.float()
        fy = 1.0 - 2.0 * flipy.float()
        one = torch.ones_like(s)
        mult = (s[:, None] * torch.stack([one, fy, one, fy], 1))[:, None, None, :]
        pk, bk = pk * mult, bk * mult
        cd = torch.where(swap[:, None, None] & (cd > 0), 3 - cd, cd)
    pos = torch.nan_to_num(torch.cat([pk, bk], 2))           # (B,T,N,4)
    pres = torch.cat([pres_p, pres_b], 2)
    pos = pos * pres[..., None]
    ours, metr = pos[:, :, Pm, :2], pos[:, :, Pm + 1, :2]
    ref = torch.where(pres_b[..., 0:1], ours, metr)
    ref_ok = pres_b.any(-1)
    rel = (pos[..., :2] - ref[:, :, None, :]) * (ref_ok[:, :, None] & pres)[..., None]
    dist = rel.norm(dim=-1)
    speed = pos[..., 2:4].norm(dim=-1)
    N = Pm + N_BALL
    zb = torch.zeros(B, T, N_BALL, device=dev)
    home = torch.cat([(cd == 1).float(), zb], 2)
    away = torch.cat([(cd == 2).float(), zb], 2)
    isball = torch.zeros(B, T, N, device=dev)
    isball[..., Pm:] = 1
    ismetr = torch.zeros(B, T, N, device=dev)
    ismetr[..., Pm + 1] = 1
    x = torch.stack([pos[..., 0] / HALF_X, pos[..., 1] / HALF_Y, pos[..., 2] / V_NORM, pos[..., 3] / V_NORM,
                     torch.clamp(speed / V_NORM, max=4.0), home, away, isball, ismetr, (~pres).float(),
                     rel[..., 0] / D_NORM, rel[..., 1] / D_NORM, torch.clamp(dist / D_NORM, max=3.0),
                     (~ref_ok).float()[:, :, None].expand(B, T, N)], -1)
    x = x.permute(0, 2, 1, 3).contiguous()                   # (B,N,T,C)
    nh, na = (cd == 1).sum(1), (cd == 2).sum(1)
    groups = torch.cat([(na > nh).long(), torch.full((B, N_BALL), 2, device=dev, dtype=torch.long)], 1)
    mask = torch.cat([pres_p.any(1), torch.ones(B, N_BALL, dtype=torch.bool, device=dev)], 1)
    extra = pres_b[:, T // 2, :].float()
    return x, groups, mask, extra, swap


# -- the network ---------------------------------------------------------------------------------

class TCN(nn.Module):
    def __init__(self, c_in, ch, layers, kernel, dropout):
        super().__init__()
        self.inp = nn.Conv1d(c_in, ch, 1)
        g = 8 if ch % 8 == 0 else 1
        self.blocks = nn.ModuleList()
        for i in range(layers):
            d = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, kernel, dilation=d, padding=d * (kernel - 1) // 2),
                nn.GroupNorm(g, ch), nn.GELU(), nn.Dropout(dropout)))

    def forward(self, x):                  # x (M, T, C) -> (M, ch) at the centre step
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        return h[:, :, h.shape[-1] // 2]


class BiGRU(nn.Module):
    def __init__(self, c_in, ch):
        super().__init__()
        self.gru = nn.GRU(c_in, ch // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, x.shape[1] // 2]


class PlayerNet(nn.Module):
    """Per-object encoder -> interaction -> attention pooling per group (home, away, ball) -> MLP."""

    def __init__(self, cfg):
        super().__init__()
        cfg = dict(DEFAULTS, **cfg)
        self.cfg = cfg
        ch = cfg["channels"]
        self.enc = (BiGRU(C_IN, ch) if cfg["encoder"] == "gru"
                    else TCN(C_IN, ch, cfg["layers"], cfg["kernel"], cfg["dropout"]))
        self.inter = nn.Sequential(nn.Linear(2 * ch, ch), nn.GELU(), nn.Linear(ch, ch))
        self.att = nn.Linear(ch, 3)
        self.head = nn.Sequential(nn.Linear(3 * ch + 4, 2 * ch), nn.GELU(), nn.Dropout(cfg["dropout"]),
                                  nn.Linear(2 * ch, 3))

    def forward(self, x, groups, mask, extra):
        B, N, T, C = x.shape
        h = self.enc(x.reshape(B * N, T, C)).reshape(B, N, -1)
        m = mask.to(h.dtype)[..., None]
        g = (h * m).sum(1) / m.sum(1).clamp(min=1)
        h = h + self.inter(torch.cat([h, g[:, None].expand_as(h)], -1))
        s = self.att(h)
        pooled = []
        for k in range(3):
            sel = mask & (groups == k)
            w = torch.softmax(s[..., k].float().masked_fill(~sel, -1e4), 1) * sel
            w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
            pooled.append((w[..., None].to(h.dtype) * h).sum(1))
        cnt = torch.stack([(mask & (groups == 0)).sum(1), (mask & (groups == 1)).sum(1)], 1).to(h.dtype) / 11.0
        out = self.head(torch.cat(pooled + [cnt, extra.to(h.dtype)], -1))
        return out[:, :2], out[:, 2]


def n_params(model_or_cfg):
    m = model_or_cfg if isinstance(model_or_cfg, nn.Module) else PlayerNet(model_or_cfg)
    return int(sum(p.numel() for p in m.parameters()))


# -- examples and training -----------------------------------------------------------------------

def examples(gt, half, present, W, stride, offset=0):
    """Training windows of one match: centres on alive frames with possession H/A, every stride."""
    F = len(gt)
    ok = np.isin(gt, ["H", "A"]) & (np.arange(F) % stride == 0)
    c = np.flatnonzero(ok)
    lo, hi = half_bounds(half, F)
    ha = np.isin(gt, ["H", "A"])
    ch = np.zeros(F, np.int64)
    ch[1:] = (gt[1:] != gt[:-1]) & ha[1:] & ha[:-1] & (np.asarray(half)[1:] == np.asarray(half)[:-1])
    cs = np.cumsum(ch)
    aux = (cs[np.minimum(c + AUX_FRAMES, F - 1)] - cs[np.maximum(c - AUX_FRAMES, 0)]) > 0
    wlo = np.maximum(c - W // 2, lo[c])
    whi = np.minimum(c + W // 2 + 1, hi[c])
    cols = select_columns(present, c, wlo, whi)
    return dict(c=c + offset, lo=lo[c] + offset, hi=hi[c] + offset, cols=cols,
                y=(gt[c] == "A").astype(np.int64), aux=aux.astype(np.float32))


def _cat(exs):
    return {k: np.concatenate([e[k] for e in exs]) for k in exs[0]}


class _Batches(torch.utils.data.Dataset):
    """One item = one batch of example indices (so the gather stays vectorised)."""

    def __init__(self, n, batch, shuffle_seed=None):
        idx = np.arange(n)
        if shuffle_seed is not None:
            idx = np.random.default_rng(shuffle_seed).permutation(n)
        self.chunks = [idx[s:s + batch] for s in range(0, n, batch)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return torch.from_numpy(self.chunks[i])


def _loader(n, batch, shuffle_seed=None):
    return torch.utils.data.DataLoader(_Batches(n, batch, shuffle_seed), batch_size=None,
                                       shuffle=False, num_workers=0)


def _to_dev(ex, device):
    return {k: torch.from_numpy(v).to(device) for k, v in ex.items()}


def _autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")


def _scaler(device):
    try:
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def _evaluate_examples(model, store, ex, offs, step, batch):
    """Mean cross-entropy and accuracy on examples (no masking, no augmentation)."""
    model.eval()
    tot, n, ok = 0.0, 0, 0
    with torch.no_grad():
        for idx in _loader(len(ex["c"]), batch):
            idx = idx.to(store.device)
            x, g, m, e, _ = make_batch(store, ex["c"][idx], ex["lo"][idx], ex["hi"][idx], ex["cols"][idx], offs, step)
            with _autocast(store.device):
                logit, _ = model(x, g, m, e)
            y = ex["y"][idx]
            tot += float(Fn.cross_entropy(logit.float(), y, reduction="sum"))
            ok += int((logit.argmax(1) == y).sum())
            n += len(idx)
    return tot / max(n, 1), ok / max(n, 1)


def train_model(store, ex_train, ex_val, cfg, gaps, verbose=True):
    """Train one PlayerNet. ex_* dicts of numpy arrays with global indices (ex_val None: fixed
    number of epochs, no early stopping). Returns (model on CPU, info)."""
    cfg = dict(DEFAULTS, **cfg)
    dev = store.device
    setup(cfg["seed"])
    model = PlayerNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = _scaler(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg["seed"])
    offs_np, step = offsets(cfg["W"])
    offs = torch.from_numpy(offs_np).to(dev)
    gaps_t = torch.from_numpy(np.asarray(gaps, np.int64)).to(dev)
    tr = _to_dev(ex_train, dev)
    va = _to_dev(ex_val, dev) if ex_val is not None else None
    best, best_state, best_epoch, bad, hist = np.inf, None, 0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for idx in _loader(len(ex_train["c"]), cfg["batch"], shuffle_seed=cfg["seed"] * 1000 + epoch):
            idx = idx.to(dev)
            x, g, m, e, swap = make_batch(store, tr["c"][idx], tr["lo"][idx], tr["hi"][idx], tr["cols"][idx],
                                          offs, step, True, gen, cfg, gaps_t)
            y = torch.where(swap, 1 - tr["y"][idx], tr["y"][idx])
            with _autocast(dev):
                logit, aux = model(x, g, m, e)
            w = cfg["w_gap"] + (1.0 - cfg["w_gap"]) * (e[:, 0] > 0).float()
            ce = Fn.cross_entropy(logit.float(), y, reduction="none")
            loss = (w * ce).sum() / w.sum() + cfg["aux_weight"] * \
                Fn.binary_cross_entropy_with_logits(aux.float(), tr["aux"][idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            nb += 1
        row = dict(epoch=epoch, train_loss=tot / max(nb, 1), sec=time.time() - t0)
        if va is not None:
            row["val_ce"], row["val_acc"] = _evaluate_examples(model, store, va, offs, step, 2 * cfg["batch"])
            if row["val_ce"] < best - 1e-4:
                best, best_epoch, bad = row["val_ce"], epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                bad += 1
        hist.append(row)
        if verbose:
            print("    epoca {:>2}: loss {:.3f}{} ({:.0f} s)".format(
                epoch, row["train_loss"],
                "  val CE {:.3f} acc {:.3f}".format(row["val_ce"], row["val_acc"]) if va is not None else "",
                row["sec"]), flush=True)
        if va is not None and bad >= cfg["patience"]:
            break
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_epoch = len(hist)
    model = PlayerNet(cfg)
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(best_epoch=best_epoch, history=hist)


# -- inference -----------------------------------------------------------------------------------

def predict_p_home(model, players, team, ball=None, ball_metrica=None, half=None, dead=None,
                   stride=1, batch=512, device=None, chunk=20000):
    """P(home | ball alive) per frame, from any tracking shaped like convert_dfl's npz.

    players       (F, P, 2) normalised 0-1, NaN where a slot is empty (slots need no identity:
                  each window takes its P_MAX most present slots).
    team          (P,) or (F, P) 'H' / 'A' / '' (per frame for tracks whose team changes).
    ball          (F, 2) our ball, NaN where missing (None: no ball at all).
    ball_metrica  (F, 2) the provider's ball, or None.
    half          (F,) or None; windows never cross a change of half.
    dead          (F,) bool or None: frames to skip, written as 0.5.
    stride        predict every `stride` frames and interpolate the log-odds in between.
    Returns (F,) float32. Runs on CPU (device='cpu') as well as CUDA.
    """
    dev = pick_device(device)
    model = model.to(dev).eval()
    cfg = model.cfg
    players = np.asarray(players, np.float32)
    F, P = players.shape[:2]
    half = np.ones(F, int) if half is None else np.asarray(half)
    dead = np.zeros(F, bool) if dead is None else np.asarray(dead, bool)
    team = np.asarray(team)
    sign = attack_sign(players, team, half)
    hlo, hhi = half_bounds(half, F)
    offs_np, step = offsets(cfg["W"])
    offs = torch.from_numpy(offs_np).to(dev)
    centres = []
    for a, z in dec.half_blocks(half, F):
        live = np.flatnonzero(~dead[a:z]) + a
        if len(live):
            centres.append(np.unique(np.r_[live[::stride], live[-1]]))
    centres = np.concatenate(centres) if centres else np.zeros(0, np.int64)
    logit = np.full(F, np.nan)
    margin = int(cfg["W"]) // 2 + 8
    for a in range(0, F, chunk):
        sel = centres[(centres >= a) & (centres < a + chunk)]
        if not len(sel):
            continue
        s0, s1 = max(0, a - margin), min(F, a + chunk + margin)
        tm = team[s0:s1] if team.ndim == 2 else team
        prep = prepare(players[s0:s1], tm, None if ball is None else ball[s0:s1],
                       None if ball_metrica is None else ball_metrica[s0:s1], half[s0:s1], sign[s0:s1])
        store = Store([prep], dev)
        c = sel - s0
        lo, hi = np.maximum(hlo[sel], s0) - s0, np.minimum(hhi[sel], s1) - s0
        cols = select_columns(prep["present"], c, np.maximum(c - cfg["W"] // 2, lo),
                              np.minimum(c + cfg["W"] // 2 + 1, hi))
        ex = _to_dev(dict(c=c, lo=lo, hi=hi, cols=cols), dev)
        out = []
        with torch.no_grad():
            for idx in _loader(len(c), batch):
                idx = idx.to(dev)
                x, g, m, e, _ = make_batch(store, ex["c"][idx], ex["lo"][idx], ex["hi"][idx], ex["cols"][idx], offs, step)
                with _autocast(dev):
                    lg, _ = model(x, g, m, e)
                lg = lg.float()
                out.append((lg[:, 0] - lg[:, 1]).cpu().numpy())
        logit[sel] = np.concatenate(out)
        del store
    p = np.full(F, 0.5, np.float32)
    for a, z in dec.half_blocks(half, F):
        live = np.flatnonzero(~dead[a:z]) + a
        known = live[np.isfinite(logit[live])]
        if len(known):
            z_ = np.interp(live, known, logit[known])
            p[live] = 1.0 / (1.0 + np.exp(-z_))
    p[dead] = 0.5
    return p.astype(np.float32)


def save_model(model, extra=None, path=None):
    path = path or os.path.join(models_dir(), "player_net.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg}, path)
    meta = dict(cfg=model.cfg, n_params=n_params(model), features=list(FEATURES), p_max=P_MAX,
                n_ball=N_BALL, steps=2 * STEPS_HALF + 1,
                normalisation=dict(x=HALF_X, y=HALF_Y, v=V_NORM, d=D_NORM, metres=SCALE.tolist(),
                                   orientation="rotated 180 deg in the halves where home attacks -x "
                                               "(team centroids at the start of the half)"),
                output="p_home = P(home | ball alive); class 0 = H, 1 = A")
    meta.update(extra or {})
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    return path


def load_model(path=None, device="cpu"):
    path = path or os.path.join(models_dir(), "player_net.pt")
    z = torch.load(path, map_location=device, weights_only=False)
    model = PlayerNet(z["cfg"])
    model.load_state_dict(z["state_dict"])
    return model.to(device).eval()


# -- evaluation ----------------------------------------------------------------------------------

def frame_metrics(gt, p, inp, near=None):
    """Accuracy of P(home) on alive frames with possession (not estimated dead), split by our
    ball seen / in a gap, and of the nearest player of the A1 chain on the same frames.
    p == 0.5 and an empty nearest count as half right (a coin)."""
    ok = np.isin(gt, ["H", "A"]) & ~inp["dead"]
    gap = ~np.isfinite(inp["ours"][:, 0])
    hit = np.where(p > 0.5, gt == "H", gt == "A") * 1.0
    hit[p == 0.5] = 0.5
    out = dict(frames=int(ok.sum()), gap_share=float(gap[ok].mean()) if ok.any() else np.nan)
    for name, sel in (("all", ok), ("seen", ok & ~gap), ("gap", ok & gap)):
        out["acc_" + name] = float(hit[sel].mean()) if sel.any() else np.nan
        if near is not None:
            nh = (near == gt) * 1.0
            nh[near == ""] = 0.5
            out["near_" + name] = float(nh[sel].mean()) if sel.any() else np.nan
    return out


def choose(table, tie=TIE):
    """Fixed rule: best mean acc_gap over the folds; within `tie` of it, the fewest parameters.
    (acc_all where the simulation has no gap in live play, level "easy".)"""
    col = "acc_gap" if table["acc_gap"].notna().all() else "acc_all"
    best = table[col].max()
    cand = table[table[col] >= best - tie]
    return cand.sort_values(["params", col], ascending=[True, False]).index[0]


def downstream(data, oof, matches, net_weight=dec.NET_WEIGHT, verbose=True):
    """spellF1 / macroF1 of the A1 decoder with and without p_home of the network, decoder
    parameters chosen leave-one-match-out (as decoder.evaluate_dfl), seed TRAIN_SEED (the seed of
    the network's inputs). B1 / B2 files are used in both variants when present."""
    G = dec.grid()
    rows = []
    for match in matches:
        d, gt = data[match]["d"], data[match]["gt"]
        ev = dec.simulate(d, TRAIN_SEED)
        ext = dec._external(dfl_paths, match, len(gt))
        ext.pop("p_home_net", None)
        lps = {"A1 senza rete": dec.match_logp(ev, ext),
               "A1 + rete": dec.match_logp(ev, dict(ext, p_home_net=oof[match]), net_weight)}
        for variant, lp in lps.items():
            for sc, md in G:
                pred = dec.decode(lp, d["half"], sc, md)
                rows.append((match, variant, dec.param_name(sc, md), *score({"half": d["half"]}, gt, pred)))
        if verbose:
            print("  decoder:", match, "fatto", flush=True)
    df = pd.DataFrame(rows, columns=["match", "variant", "params", "macro_f1", "spell_f1"])
    held = []
    for variant, g in df.groupby("variant", sort=False):
        piv = g.pivot_table(index="match", columns="params", values="spell_f1")
        for m in piv.index:
            others = piv.drop(index=m) if len(piv) > 1 else piv
            means = others.mean(0)
            best = sorted(means.index, key=lambda n: (-means[n], n))[0]
            r = g[(g.match == m) & (g.params == best)].iloc[0]
            held.append(dict(variant=variant, match=m, params=best, spell_f1=r.spell_f1, macro_f1=r.macro_f1))
    held = pd.DataFrame(held)
    summ = held.groupby("variant", sort=False).agg(spell_f1=("spell_f1", "mean"), spell_min=("spell_f1", "min"),
                                                   spell_max=("spell_f1", "max"), macro_f1=("macro_f1", "mean"))
    return summ.round(3), held


def run(quick=False, epochs=None, device=None, grid_w=None, grid_ch=None, stride=None, folds=None,
        do_downstream=True, save=True, pred_stride=PRED_STRIDE, base_cfg=None, verbose=True):
    """The whole C1 pipeline on the DFL matches. Returns a dict of aggregate tables (+ the arrays
    of each test match needed for the plot)."""
    dev = pick_device(device)
    setup(0)
    q = QUICK if quick else {}
    grid_w = tuple(grid_w or q.get("grid_w", GRID_W))
    grid_ch = tuple(grid_ch or q.get("grid_ch", GRID_CH))
    base = dict(DEFAULTS, **(base_cfg or {}))
    base["epochs"] = epochs or q.get("epochs", base["epochs"])
    base["stride"] = stride or q.get("stride", base["stride"])
    paths = dfl_paths.match_paths()
    names = [dfl_paths.match_name(p) for p in paths]
    n_folds = min(len(names), folds or q.get("folds", len(names)))
    if verbose:
        print("dispositivo: {} | partite: {} | fold: {} | simulazione: {}".format(
            dev, len(names), n_folds, level()), flush=True)
    data, preps = {}, []
    for p, m in zip(paths, names):
        npz, gt = load(p)
        d = {k: npz[k] for k in ("players", "team", "ball", "alive", "half")}
        inp = degraded_inputs(d, TRAIN_SEED)
        prep = prepare(inp["players"], inp["team"], inp["ours"], inp["metrica"], d["half"])
        data[m] = dict(d=d, gt=gt, inp=inp, near=chain_nearest(inp))
        preps.append(prep)
    store = Store(preps, dev)
    offset = dict(zip(names, store.offset))
    present = {m: pr["present"] for m, pr in zip(names, preps)}
    del preps
    cfgs = [dict(base, W=w, channels=c) for w in grid_w for c in grid_ch]
    exs = {}
    rows, models, oof, infos = [], {}, {}, {}
    test_matches = names[:n_folds]
    for cfg in cfgs:
        name = cfg_name(cfg)
        if cfg["W"] not in exs:
            exs[cfg["W"]] = {m: examples(data[m]["gt"], data[m]["d"]["half"], present[m], cfg["W"],
                                         cfg["stride"], offset[m]) for m in names}
        ex = exs[cfg["W"]]
        if verbose:
            print("\nconfigurazione {} ({} parametri)".format(name, n_params(cfg)), flush=True)
        for i, test in enumerate(test_matches):
            others = [m for m in names if m != test]
            val = names[(i + 1) % len(names)] if len(names) > 2 else None
            train = [m for m in others if m != val] if val else others
            if verbose:
                print("  fold {}/{}: test {}, validazione {}, {} partite di training".format(
                    i + 1, n_folds, i + 1, (i + 1) % len(names) + 1 if val else "-", len(train)), flush=True)
            model, info = train_model(store, _cat([ex[m] for m in train]), ex[val] if val else None, cfg,
                                      MASK_GAPS, verbose)
            inp = data[test]["inp"]
            p = predict_p_home(model, inp["players"], inp["team"], inp["ours"], inp["metrica"],
                               data[test]["d"]["half"], inp["dead"], stride=pred_stride, device=dev)
            met = frame_metrics(data[test]["gt"], p, inp, data[test]["near"])
            rows.append(dict(cfg=name, fold=i + 1, match=test, params=n_params(model),
                             best_epoch=info["best_epoch"], **met))
            models[(name, test)], oof[(name, test)], infos[(name, test)] = model, p, info
            if verbose:
                print("    accuratezza: tutti {acc_all:.3f}, palla vista {acc_seen:.3f}, palla nel vuoto {acc_gap:.3f}"
                      " | più vicino: {near_all:.3f} / {near_seen:.3f} / {near_gap:.3f}".format(**met), flush=True)
    per_match = pd.DataFrame(rows)
    cols = ["params", "acc_all", "acc_seen", "acc_gap", "near_all", "near_seen", "near_gap", "gap_share", "best_epoch"]
    table = per_match.groupby("cfg", sort=False)[cols].mean()
    table["acc_gap_min"] = per_match.groupby("cfg", sort=False)["acc_gap"].min()
    table["params"] = table["params"].astype(int)
    chosen = choose(table)
    ccfg = next(c for c in cfgs if cfg_name(c) == chosen)
    if verbose:
        print("\nconfigurazione scelta (miglior accuratezza nel vuoto, entro {:.1f} punti la più piccola): {}".format(
            100 * TIE, chosen))

    # no ball at all, chosen configuration
    abl = []
    for test in test_matches:
        inp = data[test]["inp"]
        nan = np.full_like(inp["ours"], np.nan)
        p0 = predict_p_home(models[(chosen, test)], inp["players"], inp["team"], nan, nan,
                            data[test]["d"]["half"], inp["dead"], stride=2 * pred_stride, device=dev)
        met = frame_metrics(data[test]["gt"], p0, inp, data[test]["near"])
        abl.append(dict(match=test, acc_con_palla=float(per_match[(per_match.cfg == chosen) & (per_match.match == test)].acc_all.iloc[0]),
                        acc_senza_palla=met["acc_all"], acc_senza_palla_nel_vuoto=met["acc_gap"],
                        near_nel_vuoto=met["near_gap"]))
    ablation = pd.DataFrame(abl)

    written = []
    if save:
        for test in test_matches:
            path = dfl_paths.out_path(test, "net")
            np.savez_compressed(path, p_home=oof[(chosen, test)].astype(np.float32),
                                variant=np.array("C1 player_net {} | out-of-fold seed{} | sim:{}".format(
                                    chosen, TRAIN_SEED, level())))
            written.append(path)
        if verbose:
            print("scritti {} file _net.npz in {}".format(len(written), dfl_paths.out_dir()))

    ds = None
    if do_downstream:
        ds, _ = downstream(data, {m: oof[(chosen, m)] for m in test_matches}, test_matches, verbose=verbose)

    model_path = None
    if save:
        n_ep = max(1, int(np.median([infos[(chosen, m)]["best_epoch"] for m in test_matches])))
        fcfg = dict(ccfg, epochs=n_ep)
        if verbose:
            print("\nmodello finale {} su tutte le {} partite, {} epoche".format(chosen, len(names), n_ep), flush=True)
        final, _ = train_model(store, _cat([exs[ccfg["W"]][m] for m in names]), None, fcfg, MASK_GAPS, verbose)
        model_path = save_model(final, dict(chosen=chosen, selection_rule="max mean acc_gap; within {} the fewest params".format(TIE),
                                            grid=table.round(4).reset_index().to_dict("records"),
                                            sim=level(), train_seed=TRAIN_SEED, matches=len(names)))
        if verbose:
            print("modello salvato in", model_path)
    examples_ = {m: dict(gt=data[m]["gt"], p=oof[(chosen, m)], gap=~np.isfinite(data[m]["inp"]["ours"][:, 0]),
                         dead=data[m]["inp"]["dead"], near=data[m]["near"]) for m in test_matches}
    del store
    return dict(grid=table.round(3), per_match=per_match, chosen=chosen, ablation=ablation.round(3),
                downstream=ds, written=written, model_path=model_path, examples=examples_,
                device=str(dev), sim=level())


# -- plot ----------------------------------------------------------------------------------------

COLORS = {"H": "#2a6fdb", "A": "#e8833a", "D": "#c8c8c8"}


def plot_example(res, match=None, seconds=90, start_s=None, ax=None):
    """p_home of the network vs DFL truth on one stretch, gaps of our ball shaded."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    match = match or sorted(res["examples"])[0]
    e = res["examples"][match]
    gt, p, gap = e["gt"], e["p"], e["gap"] & np.isin(e["gt"], ["H", "A"])
    win = seconds * FPS
    if start_s is None:  # the stretch with the most alive frames in a gap, after 10 minutes
        cs = np.r_[0, np.cumsum(gap)]
        s = np.arange(min(10 * 60 * FPS, max(len(gt) - win - 1, 0)), max(len(gt) - win, 1))
        a = int(s[np.argmax(cs[s + win] - cs[s])]) if len(s) else 0
    else:
        a = int(start_s * FPS)
    z = min(a + win, len(gt))
    t = np.arange(a, z) / FPS
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.6))
    seg = gt[a:z]
    cuts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        ax.broken_barh([(t[s0], (s1 - s0) / FPS)], (1.04, 0.08), color=COLORS[seg[s0]], lw=0)
    g = gap[a:z]
    gc = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, len(g)]
    for s0, s1 in zip(gc[:-1], gc[1:]):
        if g[s0]:
            ax.axvspan(t[s0], t[min(s1, len(t) - 1)], color="#000000", alpha=0.08, lw=0)
    ax.plot(t, p[a:z], color="#333333", lw=1.2, label="P(home) rete")
    ax.axhline(0.5, color="#999999", lw=0.6, ls=":")
    ax.set_ylim(-0.02, 1.15)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("P(home)")
    ax.set_title("{} · banda in alto: verità DFL · grigio: nostra palla nel vuoto".format(res["chosen"]),
                 fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color=COLORS["H"], label="H"), Patch(color=COLORS["A"], label="A"),
                       Patch(color=COLORS["D"], label="D"), Patch(color="#000000", alpha=0.08, label="vuoto palla"),
                       plt.Line2D([], [], color="#333333", label="P(home) rete")],
              ncol=5, loc="upper right", bbox_to_anchor=(1, -0.2), frameon=False, fontsize=9)
    return ax


# -- CLI -----------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="2 fold, 3 epoche, solo W=100 e 64 canali")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None, choices=("cpu", "cuda"))
    ap.add_argument("--grid-w", type=int, nargs="+", default=None, help="finestre in frame (default 100 250)")
    ap.add_argument("--grid-ch", type=int, nargs="+", default=None, help="canali della TCN (default 32 64 128)")
    ap.add_argument("--stride", type=int, default=None, help="passo tra gli esempi di training, frame")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--encoder", default="tcn", choices=("tcn", "gru"))
    ap.add_argument("--no-downstream", action="store_true", help="salta lo spellF1 del decoder A1")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.quick, a.epochs, a.device, a.grid_w, a.grid_ch, a.stride, a.folds, not a.no_downstream,
              not a.no_save, base_cfg=dict(encoder=a.encoder))
    pd.set_option("display.width", 200)
    print("\nAccuratezza per configurazione (media sui fold):")
    print(res["grid"].to_string())
    print("\nscelta:", res["chosen"])
    print("\nSenza nessuna palla (configurazione scelta):")
    print(res["ablation"].to_string(index=False))
    if res["downstream"] is not None:
        print("\nDecoder A1 con e senza p_home della rete (parametri scelti leave-one-match-out):")
        print(res["downstream"].to_string())
    return res


if __name__ == "__main__":
    main()
