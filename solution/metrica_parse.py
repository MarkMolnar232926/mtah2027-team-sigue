"""Parse the Metrica Sports open sample data (3 matches) into per-half arrays + possession labels.

Source: https://github.com/metrica-sports/sample-data ("sample data to play around with to the
broader football analytics community"; acknowledge the source). Put the files in external/metrica/.
Writes external/processed/MET-<game>.npz with the same keys as the SkillCorner files:
h{1,2}_ball (T,2), _home / _away (T,P,2) metres, _hdet / _adet (all True), _label (T,) 0 H 1 A 2 D.

Tracking is Metrica's own (25 fps, real ball gaps). Labels come from Metrica's EVENT data -- not
from the provided game -- with the same controlled-possession rule as DFL (solution/dfl_labels.py):
  * alive from a SET PIECE (or goal-kick pass) until BALL OUT (its end frame), a FAULT RECEIVED,
    a goal, a shot out of play, an offside or the end of the half;
  * while alive, possession = team of the last controlled action (PASS, CARRY, SHOT, RECOVERY,
    SET PIECE, BALL LOST at its start); CHALLENGE (duels) and CARD do not change possession;
  * then spells shorter than 2 s inside a live stretch are merged into the previous team.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dfl_labels import controlled  # noqa: E402

SRC = os.path.join(ROOT, "external", "metrica")
OUT = os.path.join(ROOT, "external", "processed")
L, W = 105.0, 68.0
CONTROL = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST"}


def to_m(xy):
    xy = np.asarray(xy, np.float32).copy()
    xy[..., 0] = (xy[..., 0] - 0.5) * L
    xy[..., 1] = (xy[..., 1] - 0.5) * W
    return xy


def labels_from_events(ev, frames):
    """ev: list of dicts (type, sub, team 0/1, start, end) for one period; frames: tracking frame
    numbers of that period. Returns per-frame codes 0/1/2 before the 2 s rule."""
    changes = []   # (frame, kind, value) -- kind 'dead' or 'team'
    for e in sorted(ev, key=lambda e: (e["start"], e["end"])):
        t, sub = e["type"], e["sub"] or ""
        if t in CONTROL:
            changes.append((e["start"], "team", e["team"]))
        if t == "BALL OUT":
            changes.append((max(e["end"], e["start"]), "dead", None))
        elif t == "FAULT RECEIVED":
            changes.append((e["start"], "dead", None))
        elif t == "SHOT" and (("GOAL" in sub.upper() and "WOODWORK" not in sub.upper())
                              or sub.upper().endswith("OUT")):
            changes.append((max(e["end"], e["start"]), "dead", None))   # goal, or shot out of play
        elif t == "BALL LOST" and sub.upper() in ("OFFSIDE", "END HALF"):
            changes.append((max(e["end"], e["start"]), "dead", None))
    changes.sort(key=lambda c: c[0])
    codes = np.full(len(frames), 2, np.int8)
    last_frame = max(max(e["end"], e["start"]) for e in ev)
    state, ci = 2, 0
    for i, f in enumerate(frames):
        while ci < len(changes) and changes[ci][0] <= f:
            _, kind, val = changes[ci]
            state = 2 if kind == "dead" else val
            ci += 1
        codes[i] = state if f <= last_frame else 2
    return codes


def game_csv(g):
    def team(side):
        t = pd.read_csv(os.path.join(SRC, "Sample_Game_%d_RawTrackingData_%s_Team.csv" % (g, side)),
                        skiprows=2)
        xy = t.iloc[:, 3:-2].to_numpy(np.float32).reshape(len(t), -1, 2)
        return t.Period.to_numpy(), t.Frame.to_numpy(), xy, t.iloc[:, -2:].to_numpy(np.float32)
    per, frm, home, ball = team("Home")
    _, _, away, _ = team("Away")
    e = pd.read_csv(os.path.join(SRC, "Sample_Game_%d_RawEventsData.csv" % g))
    ev = [dict(period=int(p), type=t, sub=s if isinstance(s, str) else "",
               team=0 if tm == "Home" else 1, start=int(a), end=int(b))
          for p, t, s, tm, a, b in zip(e.Period, e.Type, e.Subtype, e.Team, e["Start Frame"],
                                       e["End Frame"])]
    return per, frm, home, away, ball, ev


def game_epts(g):
    meta = open(os.path.join(SRC, "Sample_Game_%d_metadata.xml" % g), encoding="utf-8").read()
    pteam = dict(re.findall(r'<Player id="([^"]+)" teamId="([^"]+)"', meta))
    chan = dict(re.findall(r'<PlayerChannel channelId="x" id="([^"]+)" playerId="([^"]+)"', meta))
    teams = list(dict.fromkeys(re.findall(r'<Team id="([^"]+)"', meta)))
    specs = []
    for a, b, body in re.findall(r'<DataFormatSpecification endFrame="(\d+)" separator=":" '
                                 r'startFrame="(\d+)">(.*?)</DataFormatSpecification>', meta, re.S):
        xs = [c for c in re.findall(r'playerChannelId="([^"]+)"', body) if c.endswith("_x")]
        specs.append((int(b), int(a), [teams.index(pteam[chan[c]]) for c in xs]))
    frames, home_l, away_l, ball_l = [], [], [], []
    with open(os.path.join(SRC, "Sample_Game_%d_tracking.txt" % g)) as fh:
        for line in fh:
            f, pl, bl = line.strip().split(":")
            f = int(f)
            tm = next(s[2] for s in specs if s[0] <= f <= s[1])
            p = np.array([v.split(",") for v in pl.split(";")], np.float32)
            tm = np.array(tm)
            h, a = p[tm == 0], p[tm == 1]
            home_l.append(np.pad(h, ((0, 14 - len(h)), (0, 0)), constant_values=np.nan))
            away_l.append(np.pad(a, ((0, 14 - len(a)), (0, 0)), constant_values=np.nan))
            ball_l.append(np.array(bl.split(",")[:2], np.float32))
            frames.append(f)
    d = json.load(open(os.path.join(SRC, "Sample_Game_%d_events.json" % g)))["data"]
    ev = []
    for x in d:
        subs = x["subtypes"]
        subs = subs if isinstance(subs, list) else ([subs] if subs else [])
        ev.append(dict(period=x["period"], type=x["type"]["name"],
                       sub="-".join(s["name"] for s in subs),
                       team=teams.index(x["team"]["id"]), start=int(x["start"]["frame"]),
                       end=int(x["end"]["frame"] or 0)))
    frames = np.array(frames)
    # periods from the events: a frame belongs to the period whose events span it
    per = np.zeros(len(frames), int)
    for p in (1, 2):
        fs = [e["start"] for e in ev if e["period"] == p]
        per[(frames >= min(fs) - 25 * 5) & (frames <= max(fs) + 25 * 5)] = p
    return per, frames, np.stack(home_l), np.stack(away_l), np.stack(ball_l), ev


def main():
    os.makedirs(OUT, exist_ok=True)
    for g in (1, 2, 3):
        per, frm, home, away, ball, ev = game_csv(g) if g < 3 else game_epts(g)
        out = {}
        for p in (1, 2):
            m = per == p
            evp = [e for e in ev if e["period"] == p]
            raw = labels_from_events(evp, frm[m])
            lab = controlled(raw, 2.0)
            h, a = to_m(home[m]), to_m(away[m])
            out["h%d_ball" % p] = to_m(ball[m])
            out["h%d_home" % p], out["h%d_away" % p] = h, a
            out["h%d_hdet" % p] = np.ones(h.shape[:2], bool)
            out["h%d_adet" % p] = np.ones(a.shape[:2], bool)
            out["h%d_label" % p] = lab
            alive = lab != 2
            nposs = int(((lab[1:] != lab[:-1]) & alive[1:]).sum() + alive[0])
            print("game %d half %d frames %d  alive %.2f  home share %.2f  ball seen %.2f  "
                  "possessions %d  (raw spells %d)" % (
                      g, p, m.sum(), alive.mean(), (lab == 0).sum() / max(alive.sum(), 1),
                      np.isfinite(ball[m][:, 0]).mean(), nposs,
                      int(((raw[1:] != raw[:-1]) & (raw[1:] != 2)).sum())), flush=True)
        np.savez_compressed(os.path.join(OUT, "MET-%d.npz" % g), **out)


if __name__ == "__main__":
    main()
