"""Playground 3: BallNet estimates plus the touch and motion features in one possession model.

    playground/.venv/Scripts/python playground/playground3.py run      # train, tune on match 4, score the benchmark
    playground/.venv/Scripts/python playground/playground3.py verify   # are the old files untouched?

Reads external data only (the 7 DFL matches); never the provided game, its ground truth or video.
"""

# =================================================================================================
#  ATTEKINTES (a csapatnak)
# -------------------------------------------------------------------------------------------------
#  A kerdes: ha a playground.py labdabecsloje (BallNet, 11 jellemzo) es a playground2.py erintes- es
#  mozgas-jellemzoi (19 jellemzo) egyszerre kerulnek a modellbe, no-e a spellF1?
#
#  Merkozesek ugyanazok, mint a playground2-ben: 0-3 tanitas, 4 hangolas, 5-6 teszt.
#  A BallNet is csak a 0-3 meccset latja, keresztillesztve: a (0, 2) meccsek becsleset az a halo adja,
#  amelyik csak (1, 3)-on tanult, es forditva. A hangolo es a teszt meccset egyik halo sem latta.
#
#  A tanito masolatok es a seedek megegyeznek a playground2-vel, ezert az o eredmenyei
#  (report2.json) a kiindulopontok, nem kell oket ujra tanitani. A GPU-s tanitas nem teljesen
#  determinisztikus, ezert par ezrelekes kulonbseget nem szabad komolyan venni.
# =================================================================================================

import argparse
import json
import os
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from scipy.ndimage import maximum_filter1d  # noqa: E402

import playground as pg  # noqa: E402  (also puts solution/ and evaluation/ on the path)
import playground2 as p2  # noqa: E402
from degrade import degrade  # noqa: E402
from dfl_labels import controlled, raw_codes  # noqa: E402

FOLDS = ((0, 2), (1, 3))  # cross-fitting of BallNet over the four training matches
BASELINES = ("old 48 features | fixed cost", "+ touch and motion features | fixed cost")


def make_seq(mi, b, hp, ap_, y, truth):
    seq = pg.make_sequence(mi, b, hp, ap_, y, truth)
    seq["touch"] = p2.extra_features(b, hp, ap_).astype(np.float16)
    change = np.zeros(len(y), bool)
    change[1:] = y[1:] != y[:-1]
    seq["change"] = maximum_filter1d(change.astype(np.uint8), 2 * p2.CHANGE_RADIUS + 1).astype(np.int8)
    return seq


def train_seqs(copies, seed, source):
    """The same random draws, in the same order, as playground2.train_seqs."""
    rng = np.random.default_rng(seed)
    out = []
    for mi in p2.TRAIN:
        for ball3, home, away, status, poss in pg.load_dfl(mi, source):
            y = controlled(raw_codes(status, poss), pg.MIN_CONTROL)
            for c in range(copies):
                b, hp, ap_ = degrade(ball3, home, away, status, rng)
                yy, truth = y.copy(), ball3[:, :2].copy()
                if c % 2 == 1:
                    hp, ap_ = ap_, hp
                    yy = np.where(y == 2, 2, 1 - y).astype(np.int8)
                if rng.random() < 0.5:
                    flip = np.array([1, -1], np.float32)
                    b, hp, ap_, truth = b * flip, hp * flip, ap_ * flip, truth * flip
                out.append(make_seq(mi, b, hp, ap_, yy, truth))
        print("  match {} built ({} sequences)".format(pg.MATCHES[mi], len(out)), flush=True)
    return out


def held_out(indices, source):
    """The fixed degraded copies of playground2.held_out, in the same order."""
    out = []
    for mi in indices:
        for h, (ball3, home, away, status, poss) in enumerate(pg.load_dfl(mi, source), start=1):
            y = controlled(raw_codes(status, poss), pg.MIN_CONTROL)
            for c in range(2):
                b, hp, ap_ = degrade(ball3, home, away, status, np.random.default_rng(1000 * mi + 10 * h + c))
                out.append(make_seq(mi, b, hp, ap_, y, ball3[:, :2]))
    return out


def view(seq, ball_extra):
    """What playground2.train_net reads: all 78 columns in X, so it runs with use_extra=False."""
    return dict(X=np.concatenate([seq["X"], seq["touch"], ball_extra], 1), extra=seq["touch"],
                y=seq["y"], change=seq["change"])


def estimate_all(ball_nets, seqs):
    """Views of sequences that no BallNet saw, using the mean estimate of both, and the ball errors."""
    views, interp, net = [], [], []
    for s in seqs:
        estimates = [pg.ball_estimate(n, s) for n in ball_nets]
        delta, log_sigma = np.mean([e[0] for e in estimates], 0), np.mean([e[1] for e in estimates], 0)
        hidden = s["hidden"] & np.isfinite(s["target"][:, 0])
        interp.append(np.hypot(*(s["interp"] - s["target"])[hidden].T))
        net.append(np.hypot(*(s["interp"] + delta - s["target"])[hidden].T))
        views.append(view(s, pg.extra_features(s, delta, log_sigma)))
    return views, np.concatenate(interp), np.concatenate(net)


def run(args):
    os.makedirs(pg.OUT, exist_ok=True)
    print("building sequences on {} ...".format(p2.DEVICE), flush=True)
    seqs = train_seqs(args.copies, 0, args.source)
    val_seqs, test_seqs = held_out(p2.VAL, args.source), held_out(p2.TEST, args.source)

    # views stay in build order: playground2.train_net draws crops by index, as in playground2
    ball_nets, train_views = [], [None] * len(seqs)
    for k, fold in enumerate(FOLDS):
        print("BallNet {}/{}, trained without matches {}".format(k + 1, len(FOLDS), fold), flush=True)
        net = pg.train_ball_net([s for s in seqs if s["match"] not in fold], args.steps_ball, k)
        ball_nets.append(net)
        for i, s in enumerate(seqs):
            if s["match"] in fold:
                train_views[i] = view(s, pg.extra_features(s, *pg.ball_estimate(net, s)))
    del seqs

    val_views = estimate_all(ball_nets, val_seqs)[0]
    test_views, err_interp, err_net = estimate_all(ball_nets, test_seqs)
    print("ball error on hidden frames (benchmark, metres): interpolation median {:.1f}, BallNet median {:.1f}".format(
        np.median(err_interp), np.median(err_net)), flush=True)

    nets = []
    for seed in range(args.seeds):
        print("48 + 19 + 11 features, seed {}".format(seed), flush=True)
        nets.append(p2.train_net(train_views, False, args.steps, seed))
    val, test = p2.ensemble(nets, val_views, False), p2.ensemble(nets, test_views, False)
    setting = p2.best_setting(val, (0.0,), (0,))
    scores = p2.spells(test, *setting)
    result = dict(spell=float(scores[:, 0].mean()), macro=float(scores[:, 1].mean()),
                  per_copy=scores[:, 0].tolist(), setting=list(setting),
                  ball_median_interp=float(np.median(err_interp)), ball_median_net=float(np.median(err_net)))
    print("48 + 19 + 11 features: spellF1 {:.4f}  macroF1 {:.4f}  setting {}".format(
        result["spell"], result["macro"], setting), flush=True)

    with open(os.path.join(pg.OUT, "report2.json")) as handle:
        earlier = json.load(handle)
    for name in BASELINES:
        base = np.array(earlier[name]["per_copy"])
        print("  against '{}': delta {:+.4f}  better on {}/{} copies".format(
            name, scores[:, 0].mean() - base.mean(), int((scores[:, 0] > base).sum()), len(base)), flush=True)
    with open(os.path.join(pg.OUT, "report3.json"), "w") as handle:
        json.dump(result, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("run", "verify"))
    parser.add_argument("--source", default=pg.DFL_SOURCE)
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--steps-ball", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        pg.verify()


if __name__ == "__main__":
    main()
