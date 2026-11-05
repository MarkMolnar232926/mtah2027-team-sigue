"""Build the training set: K randomly degraded copies of every DFL half -> features + labels.

    python solution/build_train.py --copies 6
writes external/processed/train_feats.npz  (only external data inside)
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from degrade import degrade          # noqa: E402
from dfl_labels import controlled, raw_codes  # noqa: E402
from features import frame_features  # noqa: E402

PROC = os.path.join(os.path.dirname(HERE), "external", "processed")
MIN_CONTROL = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sc-copies", type=int, default=0,
                    help="augmented copies of each SkillCorner half (match index = 100 + k)")
    ap.add_argument("--met-copies", type=int, default=0,
                    help="augmented copies of each Metrica sample half (match index = 200 + k)")
    ap.add_argument("--met-games", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--out", default="train_feats.npz")
    ap.add_argument("--kalman", action="store_true", help="add Kalman-smoothed ball features")
    ap.add_argument("--nearest", action="store_true", help="add per-player features of the nearest players")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    X, Y, G = [], [], []   # G = (match index, half, copy)
    files = sorted(f for f in glob.glob(os.path.join(PROC, "DFL-MAT-*.npz")))
    for mi, f in enumerate(files):
        z = np.load(f)
        for h in (1, 2):
            ball3, home, away = z["h%d_ball" % h], z["h%d_home" % h], z["h%d_away" % h]
            status = z["h%d_status" % h]
            y = controlled(raw_codes(status, z["h%d_poss" % h]), MIN_CONTROL)
            for c in range(args.copies):
                t0 = time.time()
                b, hp, ap_ = degrade(ball3, home, away, status, rng)
                yy = y.copy()
                if c % 2 == 1:   # team swap augmentation: away plays the "home" role
                    hp, ap_ = ap_, hp
                    yy = np.where(y == 2, 2, 1 - y).astype(np.int8)
                if rng.random() < 0.5:   # mirror across the long axis
                    b = b * np.array([1, -1], np.float32)
                    hp = hp * np.array([1, -1], np.float32)
                    ap_ = ap_ * np.array([1, -1], np.float32)
                X.append(frame_features(b, hp, ap_, kalman=args.kalman, nearest=args.nearest).astype(np.float16))
                Y.append(yy)
                G.append(np.array([mi, h, c]))
                print(os.path.basename(f), h, c, "ball missing %.2f" % np.isnan(b[:, 0]).mean(),
                      "%.1fs" % (time.time() - t0), flush=True)
    # SkillCorner: real automatic tracking already -- only light augmentation on top
    sc_files = sorted(glob.glob(os.path.join(PROC, "SC-*.npz"))) if args.sc_copies else []
    met_files = [os.path.join(PROC, "MET-%d.npz" % g) for g in args.met_games] if args.met_copies else []
    jobs = [(100 + k, f, args.sc_copies) for k, f in enumerate(sc_files)]
    jobs += [(200 + g - 1, f, args.met_copies) for g, f in zip(args.met_games, met_files)]
    for gid, f, n_copies in jobs:
        z = np.load(f)
        for h in (1, 2):
            if "h%d_label" % h not in z:
                continue
            b0, h0, a0 = z["h%d_ball" % h], z["h%d_home" % h], z["h%d_away" % h]
            hd, ad, y = z["h%d_hdet" % h], z["h%d_adet" % h], z["h%d_label" % h]
            for c in range(n_copies):
                p_hide = rng.uniform(0.0, 1.0)   # how many extrapolated players to hide
                hp = np.where((~hd & (rng.random(hd.shape) < p_hide))[..., None], np.nan, h0)
                ap_ = np.where((~ad & (rng.random(ad.shape) < p_hide))[..., None], np.nan, a0)
                b = b0 + rng.normal(0, rng.uniform(0, 0.3), b0.shape).astype(np.float32)
                yy = y.copy()
                if c % 2 == 1:
                    hp, ap_ = ap_, hp
                    yy = np.where(y == 2, 2, 1 - y).astype(np.int8)
                if rng.random() < 0.5:
                    b = b * np.array([1, -1], np.float32)
                    hp = hp * np.array([1, -1], np.float32)
                    ap_ = ap_ * np.array([1, -1], np.float32)
                X.append(frame_features(b, hp, ap_, kalman=args.kalman, nearest=args.nearest).astype(np.float16))
                Y.append(yy)
                G.append(np.array([gid, h, c]))
            print(os.path.basename(f), "done", flush=True)
    lens = np.array([len(x) for x in X])
    np.savez(os.path.join(PROC, args.out), X=np.concatenate(X), Y=np.concatenate(Y),
             lens=lens, G=np.stack(G), files=np.array([os.path.basename(f) for f in files]))


if __name__ == "__main__":
    main()
