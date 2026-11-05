"""Rebuild the published models from the raw external data. Never reads the competition match.

    python solution/train_all.py

Steps (each skips nothing; ~45 min on an RTX 4060):
  1. parse DFL XML           -> external/processed/DFL-MAT-*.npz
  2. parse SkillCorner JSONL -> external/processed/SC-*.npz
  3. build degraded/augmented training features (seed 0)
  4. train three models (seeds 0, 1, 2) on all 7 DFL + 20 SkillCorner matches
The decoder penalties in models/decoder.json come from cross-validation on DFL and SkillCorner
(python solution/train.py --val <match ids> ...); see docs/TECHNICAL.md.

GPU training is not bit-for-bit deterministic, so retrained weights can differ slightly from the
published ones. The submission is regenerated exactly from the published weights by reproduce.py.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(*args):
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-u", *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    run("solution/dfl_parse.py")
    run("solution/skillcorner_parse.py")
    run("solution/metrica_parse.py")
    run("solution/build_train.py", "--copies", "8", "--sc-copies", "4", "--met-copies", "8",
        "--met-games", "1", "2", "3", "--seed", "0", "--out", "train_feats_final.npz")
    for seed in (0, 1, 2):
        run("solution/train.py", "--feats", "train_feats_final.npz", "--val", "none",
            "--steps", "6000", "--seed", str(seed), "--boundary-weight", "3",
            "--out", "models/final_met_s%d.pt" % seed)
