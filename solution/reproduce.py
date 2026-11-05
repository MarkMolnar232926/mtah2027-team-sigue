"""Regenerate the submission file from the provided match data and the published model weights.

    python solution/reproduce.py                 # -> submission/Sigue.csv

Only needs the tactical ATD (data/tactical/BAR-ATH_tactical_atd.{xml,txt}). The weights in
models/ were trained on external data only (see solution/train_all.py and docs/TECHNICAL.md).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = ["models/final_met_s0.pt", "models/final_met_s1.pt", "models/final_met_s2.pt"]
OUT = "submission/Sigue.csv"


def run(*args):
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    run("solution/atd_parse.py")
    run("solution/predict.py", "--models", *MODELS, "--tta", "--out", out)
    run("evaluation/validate_submission.py", out)
