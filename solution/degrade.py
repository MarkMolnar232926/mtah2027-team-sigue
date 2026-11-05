"""Make clean DFL tracking look like automatic, uncorrected tracking (domain randomisation).

Every knob is drawn from a broad range per call, so the model learns to be robust to a family of
trackers rather than fitted to one. The ranges were chosen from the *structure* of automatic
tracking (unlabelled statistics only; no ground truth, no video): the ball is lost in long gaps
rather than flickers, tracks fragment, balls in the air are seen projected onto the grass.
"""

import numpy as np
from scipy.stats import norm

FPS = 25


def _ar1(rng, n, tau):
    # vectorised AR(1) via scipy-free recursive filter
    from scipy.signal import lfilter
    a = np.exp(-1.0 / max(tau, 1))
    e = rng.standard_normal(n) * np.sqrt(1 - a * a)
    return lfilter([1.0], [1.0, -a], e).astype(np.float32)


def degrade(ball3, home, away, status, rng):
    """ball3 (T,3) x,y,z metres; home/away (T,P,2). Returns ball (T,2) with NaN, home, away."""
    T = len(ball3)
    ball = ball3[:, :2].copy()
    z = np.nan_to_num(ball3[:, 2])

    # 1. airborne ball seen projected onto the grass from a high camera at the side of the pitch
    cam = np.array([rng.uniform(-10, 10), rng.choice([-1, 1]) * rng.uniform(40, 70)], np.float32)
    cam_h = rng.uniform(12, 35)
    if rng.random() < 0.8:
        ray = ball - cam
        k = z / np.maximum(cam_h - z, 1.0)
        ball = ball + ray * k[:, None]

    # 2. position noise, temporally correlated
    s_b = rng.uniform(0.1, 0.6)
    ball = ball + s_b * np.stack([_ar1(rng, T, 5), _ar1(rng, T, 5)], 1)

    # 3. ball visibility: long gaps placed preferentially on hard frames
    target_missing = rng.uniform(0.25, 0.55)
    dist_h = np.nanmin(np.sqrt(((home - ball3[:, None, :2]) ** 2).sum(-1)), 1)
    dist_a = np.nanmin(np.sqrt(((away - ball3[:, None, :2]) ** 2).sum(-1)), 1)
    crowd = (np.nan_to_num(dist_h, nan=9) < 2) & (np.nan_to_num(dist_a, nan=9) < 2)
    score = (rng.uniform(0, 2.0) * (z > 1.5) + rng.uniform(0, 1.5) * crowd
             + rng.uniform(-0.5, 2.0) * (status == 0)
             + rng.uniform(0, 1.0) * (np.abs(ball3[:, 0]) > 40))
    tau = rng.uniform(2, 12) * FPS
    from scipy.ndimage import uniform_filter1d
    score = uniform_filter1d(score.astype(np.float32), int(rng.uniform(1, 6) * FPS))
    score = score + rng.uniform(0.8, 2.0) * _ar1(rng, T, tau)
    thr = np.quantile(score, 1 - target_missing)
    missing = score > thr
    # bridge very short visible islands and very short gaps, like a tracker with interpolation
    from scipy.ndimage import binary_closing, binary_opening
    w = int(rng.uniform(0.5, 3) * FPS)
    missing = binary_opening(binary_closing(missing, np.ones(w)), np.ones(w))
    ball[missing] = np.nan

    # 4. rare spurious ball detections near a random player
    n_spur = rng.poisson(T * rng.uniform(0, 0.0005) / 10)
    for _ in range(n_spur):
        i = rng.integers(0, T - 10)
        P = home if rng.random() < 0.5 else away
        j = rng.integers(0, P.shape[1])
        if np.isfinite(P[i, j, 0]):
            ball[i:i + 10] = P[i:i + 10, j] + rng.normal(0, 0.5, 2)

    # 5. players: fragmented tracks, missing stretches, jitter, ID-free
    out = []
    p_off = rng.uniform(0.02, 0.15)
    for P in (home, away):
        P = P.copy()
        s_p = rng.uniform(0.05, 0.4)
        P = P + rng.normal(0, s_p, P.shape).astype(np.float32)
        for j in range(P.shape[1]):
            if not np.isfinite(P[:, j, 0]).any():
                continue
            m = _ar1(rng, T, rng.uniform(2, 20) * FPS) > norm.ppf(1 - p_off)
            P[m, j] = np.nan
        # occasional ghost duplicate of a player
        if rng.random() < 0.3:
            j = rng.integers(0, P.shape[1])
            ghost = P[:, j:j + 1] + rng.normal(0, 1.0, (T, 1, 2)).astype(np.float32)
            keep = _ar1(rng, T, 100) > 1.5
            ghost[~keep] = np.nan
            P = np.concatenate([P, ghost], 1)
        out.append(P)
    return ball.astype(np.float32), out[0], out[1]
