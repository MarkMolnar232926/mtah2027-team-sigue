"""Identity-free per-frame features from (ball, home players, away players).

The same function runs on degraded DFL data (training) and on the competition ATD (inference), so
the model never sees anything the ATD cannot give it: no player identities, no ball height, no
event data. All coordinates are metres with the pitch centre at the origin, and x is flipped so
that the HOME team attacks towards +x (direction estimated from the data itself, per half).
"""

import warnings

import numpy as np

from kalman import smooth_ball

warnings.filterwarnings("ignore", category=RuntimeWarning)

FPS = 25
HALF_L, HALF_W = 52.5, 34.0


def home_attacks_positive(home, away):
    """True if the home team's average x is below the away team's, i.e. home defends -x."""
    hx = np.nanmean(home[..., 0])
    ax = np.nanmean(away[..., 0])
    return hx < ax


def _ffill_idx(valid):
    idx = np.where(valid, np.arange(len(valid)), -1)
    return np.maximum.accumulate(idx)


def _bfill_idx(valid):
    n = len(valid)
    idx = np.where(valid, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _speed(xy, lag=5):
    """Speed in m/s from positions, central difference over +/- lag frames, NaN-aware."""
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    d = xy[2 * lag:] - xy[:-2 * lag]
    v[lag:-lag] = np.sqrt((d ** 2).sum(-1)) * FPS / (2 * lag)
    return np.clip(v, 0, 40)


def _velocity(xy, lag=3):
    vel = np.full(xy.shape, np.nan, np.float32)
    vel[lag:-lag] = (xy[2 * lag:] - xy[:-2 * lag]) * FPS / (2 * lag)
    return vel


def frame_features(ball, home, away, kalman=False, nearest=False):
    """ball (T,2) metres, NaN where unseen; home (T,P,2), away (T,Q,2). Returns (T,F) float32.

    kalman=True appends 7 features from a Kalman-smoothed ball track (N_FEATURES_K columns).
    nearest=True appends, per team, the NEAREST_K players closest to the ball: position and velocity
    relative to the ball, sorted by distance (N_FEATURES_P columns)."""
    T = len(ball)
    if not home_attacks_positive(home, away):
        ball = ball * np.array([-1, -1], np.float32)
        home = home * np.array([-1, -1], np.float32)
        away = away * np.array([-1, -1], np.float32)

    vis = np.isfinite(ball[:, 0])
    f = _ffill_idx(vis)
    b = _bfill_idx(vis)
    t = np.arange(T)
    since = np.where(f >= 0, t - f, 10 * FPS) / FPS
    until = np.where(b < T, b - t, 10 * FPS) / FPS
    last = np.where(f[:, None] >= 0, ball[np.clip(f, 0, T - 1)], 0.0)
    nxt = np.where(b[:, None] < T, ball[np.clip(b, 0, T - 1)], 0.0)
    # linear interpolation across gaps, constant at the ends
    w = np.where((f >= 0) & (b < T), since / np.maximum(since + until, 1e-6), 0.0)[:, None]
    interp = np.where(vis[:, None], ball,
                      np.where((f >= 0)[:, None] & (b < T)[:, None], last * (1 - w) + nxt * w,
                               np.where((f >= 0)[:, None], last, nxt)))
    interp = np.nan_to_num(interp)

    bvel = _velocity(ball)
    bspeed = np.sqrt((bvel ** 2).sum(-1))
    bspeed = np.nan_to_num(np.clip(bspeed, 0, 40))
    bacc = np.zeros(T, np.float32)
    acc = np.sqrt(((bvel[2:] - bvel[:-2]) ** 2).sum(-1)) * FPS / 2
    bacc[1:-1] = np.nan_to_num(np.clip(acc, 0, 400))

    if kalman:
        kpos, kvel, kstd = smooth_ball(ball)
        kspd = np.clip(np.sqrt((kvel ** 2).sum(-1)), 0, 40)
        kacc = np.zeros(T, np.float32)
        kacc[1:-1] = np.clip(np.sqrt(((kvel[2:] - kvel[:-2]) ** 2).sum(-1)) * FPS / 2, 0, 400)

    cols = [
        vis.astype(np.float32),
        np.where(vis, ball[:, 0], 0) / HALF_L,
        np.where(vis, ball[:, 1], 0) / HALF_W,
        bspeed / 10, np.log1p(bacc) / 5,
        np.minimum(since, 10) / 10, np.minimum(until, 10) / 10,
        interp[:, 0] / HALF_L, interp[:, 1] / HALF_W,
        (np.abs(interp[:, 0]) > HALF_L).astype(np.float32),
        (np.abs(interp[:, 1]) > HALF_W).astype(np.float32),
        (HALF_L - np.abs(interp[:, 0])) / 10, (HALF_W - np.abs(interp[:, 1])) / 10,
    ]

    dmins, krel, near_cols = [], [], []
    for side, P in ((1, home), (-1, away)):
        n_seen = np.isfinite(P[..., 0]).sum(1)
        spd = _speed(P)
        vel = _velocity(P)
        d = np.sqrt(((P - interp[:, None, :]) ** 2).sum(-1))
        d = np.where(np.isfinite(d), d, 99.0)
        order = np.argsort(d, axis=1)
        ds = np.take_along_axis(d, order, 1)
        d1 = np.minimum(ds[:, 0], 40)
        d2 = np.minimum(ds[:, 1], 40) if ds.shape[1] > 1 else d1
        dmins.append(d1)
        near = order[:, 0]
        nvel = vel[t, near]
        rel = np.sqrt(((nvel - np.nan_to_num(bvel)) ** 2).sum(-1))
        rel = np.where(vis & np.isfinite(rel), np.minimum(rel, 30), 30.0)
        nspd = np.nan_to_num(spd[t, near], nan=0.0)
        if nearest:
            ball_v = np.nan_to_num(bvel)
            for j in range(NEAREST_K):
                if j < P.shape[1]:
                    idx = order[:, j]
                    ok = ds[:, j] < 98
                    rp = P[t, idx] - interp
                    rv = np.nan_to_num(vel[t, idx]) - ball_v
                else:
                    ok = np.zeros(T, bool)
                    rp = rv = np.zeros((T, 2), np.float32)
                near_cols += [ok.astype(np.float32),
                              np.where(ok, np.clip(rp[:, 0] / 10, -4, 4), 0),
                              np.where(ok, np.clip(rp[:, 1] / 10, -4, 4), 0),
                              np.where(ok, np.clip(rv[:, 0] / 10, -3, 3), 0),
                              np.where(ok, np.clip(rv[:, 1] / 10, -3, 3), 0)]
        if kalman:
            krel.append(np.minimum(np.nan_to_num(
                np.sqrt(((nvel - kvel) ** 2).sum(-1)), nan=30.0), 30) / 10)
        with np.errstate(all="ignore"):
            cx = np.nanmean(P[..., 0], 1)
            cy = np.nanmean(P[..., 1], 1)
            sx = np.nanstd(P[..., 0], 1)
            sy = np.nanstd(P[..., 1], 1)
            mx = np.nanmax(P[..., 0] * side, 1)  # most advanced player in attacking direction
            mn = np.nanmin(P[..., 0] * side, 1)
            msp = np.nanmean(spd, 1)
            slow = np.nanmean(np.where(np.isfinite(spd), spd < 1.0, np.nan), 1)
            fast = np.nanmax(spd, 1)
        cols += [
            d1 / 10, d2 / 10,
            (d < 2).sum(1).astype(np.float32), (d < 5).sum(1).astype(np.float32) / 3,
            (d < 10).sum(1).astype(np.float32) / 5,
            rel / 10, nspd / 5,
            n_seen.astype(np.float32) / 11,
            np.nan_to_num(cx) / HALF_L, np.nan_to_num(cy) / HALF_W,
            np.nan_to_num(sx) / 20, np.nan_to_num(sy) / 15,
            np.nan_to_num(mx) / HALF_L, np.nan_to_num(mn) / HALF_L,
            np.nan_to_num(msp) / 3, np.nan_to_num(slow), np.nan_to_num(fast) / 8,
        ]
    cols.append((dmins[0] - dmins[1]) / 10)
    cols += near_cols
    if kalman:
        cols += [np.clip(kpos[:, 0] / HALF_L, -1.5, 1.5), np.clip(kpos[:, 1] / HALF_W, -1.5, 1.5),
                 kspd / 10, np.log1p(kacc) / 5, np.minimum(kstd, 20) / 10] + krel
    X = np.stack(cols, 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


N_FEATURES = 13 + 2 * 17 + 1
N_FEATURES_K = N_FEATURES + 7
NEAREST_K = 4
N_FEATURES_P = N_FEATURES + 2 * NEAREST_K * 5
