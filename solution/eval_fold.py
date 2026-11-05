"""Fixed external benchmark for comparing variants: DFL matches 5,6 (degraded, fixed seeds),
SkillCorner matches 15-19 (real automatic tracking) and Metrica sample game 3 (Metrica tracking,
EPTS format; match index 202). Models must not have trained on these.

    python solution/eval_fold.py --models models/expB_base.pt [--tta]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from degrade import degrade  # noqa: E402
from dfl_labels import controlled, raw_codes  # noqa: E402
from eval_external import load_models, sc_halves  # noqa: E402
from features import frame_features  # noqa: E402
from model import predict, viterbi  # noqa: E402

PROC = os.path.join(ROOT, "external", "processed")
DFL_VAL, SC_VAL, MET_VAL = (5, 6), (15, 16, 17, 18, 19), (2,)


def benchmark_halves():
    files = sorted(glob.glob(os.path.join(PROC, "DFL-MAT-*.npz")))
    for mi in DFL_VAL:
        z = np.load(files[mi])
        for h in (1, 2):
            y = controlled(raw_codes(z["h%d_status" % h], z["h%d_poss" % h]), 2.0)
            for c in range(2):
                rng = np.random.default_rng(1000 * mi + 10 * h + c)
                b, hp, ap_ = degrade(z["h%d_ball" % h], z["h%d_home" % h], z["h%d_away" % h],
                                     z["h%d_status" % h], rng)
                yield "DFL", b, hp, ap_, y
    for k, h, b, hp, ap_, y in sc_halves(set(SC_VAL)):
        yield "SC", b, hp, ap_, y
    for k, h, b, hp, ap_, y in sc_halves(set(MET_VAL), "MET-*.npz"):
        yield "MET", b, hp, ap_, y


def probs(nets, b, hp, ap_, dev, tta):
    views = [(b, hp, ap_, False)]
    if tta:
        flip = np.array([1, -1], np.float32)
        views += [(b * flip, hp * flip, ap_ * flip, False), (b, ap_, hp, True),
                  (b * flip, ap_ * flip, hp * flip, True)]
    out = []
    for bb, hh, aa, swapped in views:
        X = frame_features(bb, hh, aa, kalman=nets[0].kalman, nearest=nets[0].nearest)
        for n in nets:
            p = np.exp(predict(n, X, dev))
            if swapped:
                p = p[:, [1, 0, 2]]
            out.append(p)
    return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--switch", nargs=2, type=float, default=None)
    ap.add_argument("--save", default=None, help="save probabilities for decoder experiments")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dec = json.load(open(os.path.join(ROOT, "models", "decoder.json")))
    sh, sa = args.switch or (dec["switch_ha"], dec["switch_alive"])
    nets = load_models(args.models, dev)
    res = {"DFL": [], "SC": [], "MET": []}
    saved = []
    for src, b, hp, ap_, y in benchmark_halves():
        p = probs(nets, b, hp, ap_, dev, args.tta)
        saved.append((src, p, y))
        r = pm.evaluate(y, viterbi(np.log(np.clip(p, 1e-6, 1)), sh, sa))
        res[src].append((r["spell_f1_mean"], r["macro_f1"]))
    for src, v in res.items():
        v = np.array(v)
        print("%-3s spellF1 %.4f  macroF1 %.4f  (%d halves)" % (src, *v.mean(0), len(v)))
    if args.save:
        np.save(os.path.join(PROC, args.save), np.array(saved, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
