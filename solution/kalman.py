"""Constant-velocity Kalman filter + Rauch-Tung-Striebel smoother for the ball track.

Input: ball (T,2) metres with NaN where unseen. Output: smoothed position (T,2), velocity (T,2) and
position standard deviation (T,) -- the last one says how sure the track is, which grows inside gaps.
Detections far from the prediction (Mahalanobis gate) are treated as outliers, unless the track keeps
disagreeing for several frames, in which case the filter re-initialises (a kick or a real jump).

Noise parameters were chosen on EXTERNAL data only (solution/kalman_tune.py: degraded DFL copies
compared with the clean DFL ball of training matches), never on the provided game.
"""
import numpy as np

FPS = 25
# set by solution/kalman_tune.py on external data
Q_ACC = 30.0     # white-acceleration spectral density, m^2/s^3
R_POS = 0.3      # measurement noise std, m
GATE = 25.0      # chi-square gate (2 dof)
RESET_AFTER = 4  # consecutive rejected detections before re-initialising


def smooth_ball(ball, q=None, r=None, gate=None, reset_after=None):
    q = Q_ACC if q is None else q
    r = R_POS if r is None else r
    gate = GATE if gate is None else gate
    reset_after = RESET_AFTER if reset_after is None else reset_after
    T = len(ball)
    dt = 1.0 / FPS
    F = np.array([[1, dt], [0, 1]])
    Qm = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    R = r * r
    # both axes share covariance (same model, same measurement mask), so filter them together
    xs_f = np.zeros((T, 2, 2))      # filtered state per axis: [axis][pos, vel]
    Ps_f = np.zeros((T, 2, 2))
    xs_p = np.zeros((T, 2, 2))
    Ps_p = np.zeros((T, 2, 2))
    vis = np.isfinite(ball[:, 0])
    first = np.argmax(vis) if vis.any() else 0
    x = np.zeros((2, 2))
    if vis.any():
        x[:, 0] = ball[first]
    P = np.array([[r * r, 0], [0, 100.0]])
    big = np.array([[1e4, 0], [0, 100.0]])
    if not vis.any():
        P = big
    rejected = 0
    for t in range(T):
        if t > 0:
            x = x @ F.T
            P = F @ P @ F.T + Qm
        xs_p[t], Ps_p[t] = x, P
        if vis[t]:
            z = ball[t]
            S = P[0, 0] + R
            innov = z - x[:, 0]
            d2 = (innov ** 2).sum() / S
            if d2 > gate and rejected < reset_after and t >= first + 1:
                rejected += 1
            else:
                if d2 > gate:   # persistent disagreement: re-initialise on the detections
                    x = np.stack([z, np.zeros(2)], 1)
                    P = np.array([[R, 0], [0, 100.0]])
                else:
                    K = P[:, 0] / S
                    x = x + np.outer(innov, K)
                    P = P - np.outer(K, P[0, :])
                rejected = 0
        xs_f[t], Ps_f[t] = x, P
    # RTS backward pass
    xs = xs_f.copy()
    Ps = Ps_f.copy()
    for t in range(T - 2, -1, -1):
        C = Ps_f[t] @ F.T @ np.linalg.inv(Ps_p[t + 1])
        xs[t] = xs_f[t] + (xs[t + 1] - xs_p[t + 1]) @ C.T
        Ps[t] = Ps_f[t] + C @ (Ps[t + 1] - Ps_p[t + 1]) @ C.T
    pos = xs[:, :, 0].astype(np.float32)
    vel = xs[:, :, 1].astype(np.float32)
    std = np.sqrt(np.maximum(Ps[:, 0, 0], 0)).astype(np.float32)
    return pos, vel, std
