"""Read the competition ATD (tactical) into arrays. No labels, no ground truth.

Output (cache/BAR-ATH_atd.npz, gitignored -- it is derived match data and must not be published):
    ball    (N, 3)   x, y in metres (pitch centre = origin, like DFL), ball speed; NaN if unseen
    pos     (N, 46, 2) track slot positions in metres, NaN if unseen
    team    (46,)    0 = home slot, 1 = away slot
Row i = video frame i (ATD frame_id - 1).
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

CACHE = os.path.join(ROOT, "cache")


def track_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")],
                    np.int8)


def load_atd(force=False):
    out = os.path.join(CACHE, GAME + "_atd.npz")
    if os.path.exists(out) and not force:
        z = np.load(out)
        return z["ball"], z["pos"], z["team"]
    meta, raw = require(["atd_meta", "atd_raw"])
    team = track_teams(meta)
    n = GAMES[GAME]["n_video_frames"]
    L, W = GAMES[GAME]["pitch"]
    ball = np.full((n, 3), np.nan, np.float32)
    pos = np.full((n, len(team), 2), np.nan, np.float32)
    with open(raw) as handle:
        for line in handle:
            parts = line.rstrip().split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            b = parts[2].split(",")
            if b[0] != "NaN":
                ball[i] = [float(b[0]), float(b[1]), float(b[2]) if b[2] != "NaN" else np.nan]
            for s, g in enumerate(parts[1].split(";")):
                if g[0] != "N":
                    x, y, _ = g.split(",")
                    pos[i, s] = (float(x), float(y))
    # normalised 0-1 -> metres centred on the pitch, the DFL convention
    ball[:, 0] = (ball[:, 0] - 0.5) * L
    ball[:, 1] = (ball[:, 1] - 0.5) * W
    pos[..., 0] = (pos[..., 0] - 0.5) * L
    pos[..., 1] = (pos[..., 1] - 0.5) * W
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out, ball=ball, pos=pos, team=team)
    return ball, pos, team


if __name__ == "__main__":
    ball, pos, team = load_atd(force=True)
    print("frames", len(ball), "ball seen", np.isfinite(ball[:, 0]).mean().round(3),
          "home slots", (team == 0).sum(), "away slots", (team == 1).sum())
