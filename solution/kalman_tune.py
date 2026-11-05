"""Choose the ball Kalman noise parameters on EXTERNAL data only.

For DFL training matches 0-4 (never the benchmark matches 5-6, never the provided game), degrade the
clean ball track exactly as in training, smooth it, and compare the smoothed velocity with the clean
ball's velocity on live-ball frames. No possession labels are used.

    python solution/kalman_tune.py
"""
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from degrade import degrade        # noqa: E402
from features import _velocity    # noqa: E402
from kalman import smooth_ball    # noqa: E402

PROC = os.path.join(os.path.dirname(HERE), "external", "processed")


def cases():
    files = sorted(glob.glob(os.path.join(PROC, "DFL-MAT-*.npz")))[:5]
    for mi, f in enumerate(files):
        z = np.load(f)
        h = 1 + mi % 2
        rng = np.random.default_rng(777 + mi)
        st = z["h%d_status" % h]
        b, hp, ap_ = degrade(z["h%d_ball" % h], z["h%d_home" % h], z["h%d_away" % h], st, rng)
        clean = z["h%d_ball" % h][:, :2].astype(np.float64)
        v_true = np.full_like(clean, np.nan)
        v_true[1:-1] = (clean[2:] - clean[:-2]) * 25 / 2
        speed = np.sqrt((v_true ** 2).sum(-1))
        # live, visible frames with a physically plausible reference speed (the clean track has
        # occasional jumps that would otherwise dominate the error)
        mask = np.isfinite(b[:, 0]) & (st > 0) & np.isfinite(speed) & (speed < 40)
        yield b, v_true, mask


def main():
    data = list(cases())
    base = [np.nanmedian(np.sqrt(((_velocity(b) - vt)[m] ** 2).sum(-1))) for b, vt, m in data]
    print("finite-difference velocity (current features): median error %.2f m/s" % np.mean(base),
          flush=True)
    for q in (10, 30, 60, 150, 400):
        for r in (0.3, 0.6):
            t0 = time.time()
            errs = []
            for b, vt, m in data:
                _, v, _ = smooth_ball(b, q=q, r=r)
                errs.append(np.median(np.sqrt(((v - vt)[m] ** 2).sum(-1))))
            print("q %5.0f r %.1f  median velocity error %.2f m/s  (%.0fs)" % (q, r, np.mean(errs),
                                                                    time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
