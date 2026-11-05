"""Parse the open DFL dataset (Bassek et al., 2025, CC BY 4.0) into compact per-half arrays.

External training data only -- nothing here touches the competition match.

For every match and half we store, at 25 fps:
    ball      (T, 3)  x, y, z in metres, pitch centre = origin
    status    (T,)    1 = ball alive, 0 = dead          (DFL BallStatus)
    poss      (T,)    1 = home team, 2 = away team       (DFL BallPossession)
    home/away (T, P, 2) player positions in metres, NaN when the player is not on the pitch

    python solution/dfl_parse.py            # writes external/processed/<match>.npz
"""

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "external", "raw")
OUT = os.path.join(ROOT, "external", "processed")

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?')
BALLATTR = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def match_teams(info_path):
    root = ET.parse(info_path).getroot()
    home = away = None
    for team in root.iter("Team"):
        if team.get("Role") == "home":
            home = team.get("TeamId")
        elif team.get("Role") == "guest":
            away = team.get("TeamId")
    return home, away


def parse_positions(path):
    """{half: {key: (frames, xs, ys, zs, poss, status)}} streamed line by line."""
    data = {}
    current = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = FRAME.match(line)
                buf = current[1]
                buf[0].append(int(m.group(1)))
                buf[1].append(float(m.group(2)))
                buf[2].append(float(m.group(3)))
                if current[0] == "BALL":
                    buf[3].append(float(m.group(4) or 0.0))
                    b = BALLATTR.search(line)
                    buf[4].append(int(b.group(1)))
                    buf[5].append(int(b.group(2)))
            elif line.startswith("<FrameSet"):
                m = FRAMESET.match(line)
                half, team, person = m.groups()
                buf = ([], [], [], [], [], [])
                data.setdefault(half, {})[(team, person)] = buf
                current = (team if team == "BALL" else "P", buf)
    return data


def build_half(sets, home, away):
    ball_key = [k for k in sets if k[0] == "BALL"][0]
    bf = np.array(sets[ball_key][0])
    start, stop = bf.min(), bf.max() + 1
    n = stop - start
    ball = np.full((n, 3), np.nan, np.float32)
    status = np.zeros(n, np.int8)
    poss = np.zeros(n, np.int8)
    idx = bf - start
    b = sets[ball_key]
    ball[idx, 0], ball[idx, 1], ball[idx, 2] = b[1], b[2], b[3]
    poss[idx], status[idx] = b[4], b[5]

    teams = {}
    for side, tid in (("home", home), ("away", away)):
        keys = [k for k in sets if k[0] == tid]
        arr = np.full((n, len(keys), 2), np.nan, np.float32)
        for j, k in enumerate(keys):
            f = np.array(sets[k][0]) - start
            ok = (f >= 0) & (f < n)
            arr[f[ok], j, 0] = np.array(sets[k][1])[ok]
            arr[f[ok], j, 1] = np.array(sets[k][2])[ok]
        teams[side] = arr
    return dict(ball=ball, status=status, poss=poss, home=teams["home"], away=teams["away"],
                first_frame=np.int64(start))


def main():
    os.makedirs(OUT, exist_ok=True)
    for pos_path in sorted(glob.glob(os.path.join(RAW, "DFL_04_03_positions_raw_observed_*.xml"))):
        match_id = pos_path.rsplit("_", 1)[-1][:-4]
        out_path = os.path.join(OUT, match_id + ".npz")
        if os.path.exists(out_path):
            print("skip", match_id)
            continue
        info = glob.glob(os.path.join(RAW, "DFL_02_01_matchinformation_*" + match_id + ".xml"))[0]
        home, away = match_teams(info)
        data = parse_positions(pos_path)
        arrays = {}
        for h, half in enumerate(("firstHalf", "secondHalf"), start=1):
            for k, v in build_half(data[half], home, away).items():
                arrays["h{}_{}".format(h, k)] = v
        np.savez_compressed(out_path, **arrays)
        s = arrays["h1_status"]
        print(match_id, "half1 frames", len(s), "alive", round(s.mean(), 3),
              "players", arrays["h1_home"].shape[1], arrays["h1_away"].shape[1])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
