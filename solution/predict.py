"""Label the competition match with the externally trained model(s).

    python solution/predict.py --models models/final_bw3_s0.pt ... --tta --out submission/Sigue.csv

Inputs: the tactical ATD only. Decoder penalties come from models/decoder.json, which was chosen
by cross-validation on DFL matches -- never on the provided game. --tta averages the
original, mirrored, team-swapped and mirrored+team-swapped views (same function as the external
benchmark in eval_fold.py).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from atd_parse import load_atd                 # noqa: E402
from eval_external import load_models          # noqa: E402
from eval_fold import probs                    # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from model import viterbi                      # noqa: E402

LABELS = np.array(["H", "A", "D"])


def team_arrays(pos, team, a, b):
    return pos[a:b][:, team == 0], pos[a:b][:, team == 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--decoder", default="models/decoder.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", action="store_true", help="average mirrored / team-swapped views")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    with open(os.path.join(ROOT, args.decoder)) as fh:
        dec = json.load(fh)

    nets = load_models(args.models, dev)

    ball, pos, team = load_atd()
    n = GAMES[GAME]["n_video_frames"]
    labels = np.full(n, "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        a, b = first, last + 1
        home, away = team_arrays(pos, team, a, b)
        # ensemble (and views) in probability space
        p = probs(nets, ball[a:b, :2], home, away, dev, args.tta)
        path = viterbi(np.log(np.clip(p, 1e-6, 1)), dec["switch_ha"], dec["switch_alive"])
        labels[a:b] = LABELS[path]
        alive = path != 2
        print("half", half, "frames", b - a, "dead %.2f" % (~alive).mean(),
              "home share of alive %.2f" % ((path == 0).sum() / max(alive.sum(), 1)))

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, l) for i, l in enumerate(labels))
    print("wrote", out)


if __name__ == "__main__":
    main()
