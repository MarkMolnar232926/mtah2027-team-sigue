"""Parse SkillCorner Open Data (MIT, github.com/SkillCorner/opendata) into the same per-half arrays
as the DFL parser, upsampled 10 fps -> 25 fps. External data only.

Labels come from phases_of_play.csv: inside a phase the ball is alive and `team_in_possession_id`
has it; outside every phase the ball is dead (SkillCorner defines phases only while in play).
Ball positions are kept only where SkillCorner says the ball was actually detected -- extrapolated
ball positions are dropped, which makes the ball as patchy as real automatic tracking.
Players keep their extrapolated positions plus a per-frame `detected` mask, so the training loader
can decide how many unseen players to hide.

    python solution/skillcorner_parse.py      # writes external/processed/SC-<id>.npz
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "external", "skillcorner", "data", "matches")
OUT = os.path.join(ROOT, "external", "processed")
SRC_FPS, FPS = 10, 25


def upsample(arr, n_src, n_dst):
    """Linear interpolation along axis 0 from n_src samples at 10 fps to n_dst at 25 fps; NaN-aware."""
    t = np.arange(n_dst) * SRC_FPS / FPS
    i0 = np.clip(np.floor(t).astype(int), 0, n_src - 1)
    i1 = np.clip(i0 + 1, 0, n_src - 1)
    w = (t - i0).reshape(-1, *([1] * (arr.ndim - 1)))
    return (arr[i0] * (1 - w) + arr[i1] * w).astype(np.float32)


def parse_match(mdir):
    mid = os.path.basename(mdir)
    meta = json.load(open(os.path.join(mdir, mid + "_match.json"), encoding="utf-8"))
    home_id = meta["home_team"]["id"]
    team_of = {p["id"]: (0 if p["team_id"] == home_id else 1) for p in meta["players"]}
    phases = pd.read_csv(os.path.join(mdir, mid + "_phases_of_play.csv"))

    rows = [json.loads(l) for l in open(os.path.join(mdir, mid + "_tracking_extrapolated.jsonl"))]
    frames = np.array([r["frame"] for r in rows])
    out = {}
    for period in (1, 2):
        sel = [k for k, r in enumerate(rows) if r.get("period") == period]
        if not sel:
            continue
        f0, f1 = frames[sel[0]], frames[sel[-1]]
        n = f1 - f0 + 1
        ball = np.full((n, 2), np.nan, np.float32)
        home = np.full((n, 16, 2), np.nan, np.float32)
        away = np.full((n, 16, 2), np.nan, np.float32)
        hdet = np.zeros((n, 16), bool)
        adet = np.zeros((n, 16), bool)
        slot = {}
        for k in sel:
            r = rows[k]
            i = r["frame"] - f0
            b = r.get("ball_data") or {}
            if b.get("x") is not None and b.get("is_detected"):
                ball[i] = (b["x"], b["y"])
            cnt = [0, 0]
            for p in r.get("player_data") or []:
                t = team_of.get(p["player_id"])
                if t is None or p.get("x") is None:
                    continue
                s = slot.setdefault(p["player_id"], None)
                j = cnt[t]
                cnt[t] += 1
                if j >= 16:
                    continue
                (home if t == 0 else away)[i, j] = (p["x"], p["y"])
                (hdet if t == 0 else adet)[i, j] = bool(p.get("is_detected"))
        # labels at 10 fps
        lab = np.full(n, 2, np.int8)
        ph = phases[phases.period == period]
        for _, p in ph.iterrows():
            a, b = int(p.frame_start) - f0, int(p.frame_end) - f0
            lab[max(a, 0):max(min(b, n), 0)] = 0 if p.team_in_possession_id == home_id else 1
        n25 = int(n * FPS / SRC_FPS)
        idx = np.clip(np.round(np.arange(n25) * SRC_FPS / FPS).astype(int), 0, n - 1)
        # ball: interpolate only between two detected samples, never across a gap
        b25 = upsample(ball, n, n25)
        out["h%d_ball" % period] = b25
        out["h%d_home" % period] = upsample(home, n, n25)
        out["h%d_away" % period] = upsample(away, n, n25)
        out["h%d_hdet" % period] = hdet[idx]
        out["h%d_adet" % period] = adet[idx]
        out["h%d_label" % period] = lab[idx]
    return mid, out


def main():
    os.makedirs(OUT, exist_ok=True)
    for mdir in sorted(glob.glob(os.path.join(SC, "*"))):
        if not os.path.isdir(mdir):
            continue
        mid, out = parse_match(mdir)
        np.savez_compressed(os.path.join(OUT, "SC-%s.npz" % mid), **out)
        l1 = out["h1_label"]
        print(mid, "half1 frames(25fps)", len(l1), "alive %.2f" % (l1 != 2).mean(),
              "ball seen %.2f" % np.isfinite(out["h1_ball"][:, 0]).mean(), flush=True)


if __name__ == "__main__":
    main()
