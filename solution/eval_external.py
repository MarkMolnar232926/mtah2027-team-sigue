"""Score trained models on SkillCorner matches (real automatic tracking, external data).

    python solution/eval_external.py --models models/final_s0.pt [--matches 0-19]
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
from features import N_FEATURES_K, N_FEATURES_P, frame_features  # noqa: E402
from model import PossessionNet, predict, viterbi  # noqa: E402

PROC = os.path.join(ROOT, "external", "processed")


def sc_halves(ids=None, pattern="SC-*.npz"):
    files = sorted(glob.glob(os.path.join(PROC, pattern)))
    for k, f in enumerate(files):
        if ids is not None and k not in ids:
            continue
        z = np.load(f)
        for h in (1, 2):
            if "h%d_label" % h in z:
                yield k, h, z["h%d_ball" % h], z["h%d_home" % h], z["h%d_away" % h], z["h%d_label" % h]


def load_models(paths, dev):
    nets = []
    for m in paths:
        ck = torch.load(os.path.join(ROOT, m), map_location="cpu")
        net = PossessionNet(ck["n_feat"], hidden=ck["hidden"]).to(dev)
        net.load_state_dict(ck["state"])
        net.kalman = ck["n_feat"] == N_FEATURES_K
        net.nearest = ck["n_feat"] == N_FEATURES_P
        nets.append(net)
    return nets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--matches", nargs="*", type=int, default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dec = json.load(open(os.path.join(ROOT, "models", "decoder.json")))
    nets = load_models(args.models, dev)
    res = []
    for k, h, b, hp, ap_, y in sc_halves(set(args.matches) if args.matches else None):
        X = frame_features(b, hp, ap_, kalman=nets[0].kalman, nearest=nets[0].nearest)
        p = np.mean([np.exp(predict(n, X, dev)) for n in nets], 0)
        path = viterbi(np.log(np.clip(p, 1e-6, 1)), dec["switch_ha"], dec["switch_alive"])
        r = pm.evaluate(y, path)
        res.append((r["spell_f1_mean"], r["macro_f1"], (p.argmax(1) == y).mean()))
        print("SC match %2d half %d  spellF1 %.3f  macroF1 %.3f  frame acc %.3f" % ((k, h) + res[-1]),
              flush=True)
    r = np.array(res)
    print("MEAN spellF1 %.4f  macroF1 %.4f  frame acc %.4f  (%d halves)" % (*r.mean(0), len(r)))


if __name__ == "__main__":
    main()
