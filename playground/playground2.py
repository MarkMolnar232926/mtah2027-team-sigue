"""Playground 2: three candidates for a higher spellF1, each measured on external data.

    playground/.venv/Scripts/python playground/playground2.py run      # train, tune on match 4, score the benchmark
    playground/.venv/Scripts/python playground/playground2.py verify   # are the old files untouched?

Reads external data only (the 7 DFL matches); never the provided game, its ground truth or video.
"""

# =================================================================================================
#  ATTEKINTES (a csapatnak)
# -------------------------------------------------------------------------------------------------
#  Harom uj funkcio, mindegyik kulon merve:
#
#  1. UJ JELLEMZOK (19 darab, extra_features). Ket csoport:
#     - Erintes-sorozat: ki ert a labdahoz az elozo 3 es a kovetkezo 3 erintesnel, mennyi ideje,
#       es hany egymas utani erintes volt azonos csapattol. A kiiras szerint a birtoklas
#       kontrollt jelent ("tobb egymast koveto passz"), egy erintes meg nem az.
#     - Csapatmozgas: a csapat atlagsebessege a sajat tamadasi iranyaban, es annak valtozasa.
#       Labdavesztesnel a ket csapat hirtelen megfordul, ez labda nelkul is latszik.
#
#  2. BEMENETFUGGO VALTASI KOLTSEG (viterbi_tv). A halo kap egy masodik kimenetet: "kozel van-e
#     itt egy birtokvaltas". A Viterbi ott olcsobban valt, ahol a halo valtast var, maskor
#     dragabban. A regi dekoder minden frame-en ugyanazt a 16/24 buntetest hasznalja.
#
#  3. HATAR-IGAZITAS (snap_boundaries). A Viterbi utan minden valtast a valtas-kimenet legnagyobb
#     erteke fele tolunk egy kis ablakon belul. A spellF1 atfedest (IoU) mer, ezert szamit a pontos
#     hatar.
#
#  Merkozesek: 0-3 tanitas, 4 hangolas (dekoder-beallitasok), 5-6 teszt (a repo benchmarkja).
#  Minden beallitasrol a 4-es meccs dont, a tesztmeccs csak a vegen ad szamot.
#  Csak DFL-en merunk, mert a SkillCorner es a Metrica nincs meg ezen a gepen.
# =================================================================================================

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.ndimage import maximum_filter1d, uniform_filter1d  # noqa: E402

import playground as pg  # noqa: E402  (also puts solution/ and evaluation/ on the path)
from degrade import degrade  # noqa: E402
from dfl_labels import controlled, raw_codes  # noqa: E402
from features import _velocity, frame_features, home_attacks_positive  # noqa: E402
from model import PossessionNet  # noqa: E402

DEVICE = pg.DEVICE
TRAIN, VAL, TEST = (0, 1, 2, 3), (4,), (5, 6)
TOUCH_RADIUS = 2.0        # m, the closest band solution/features.py already uses
TOUCH_MERGE = 5           # frames: touches of one team this close together are one touch
N_EXTRA = 19
CHANGE_RADIUS = 12        # frames either side of a label change that count as "change here"
GRID_SWITCH = (8.0, 12.0, 16.0, 24.0)
GRID_ALIVE = (16.0, 24.0, 32.0)
GRID_ALPHA = (0.0, 0.25, 0.5, 0.75, 0.9)
GRID_SNAP = (0, 12, 25, 50)
LABEL_LOSS_CHANGE = 0.5


# =================================================================================================
#  1. RESZ -- UJ JELLEMZOK
# =================================================================================================

def touch_features(ball, home, away):
    """10 columns from the sequence of touches: who touched before and after this frame."""
    T = len(ball)
    vis = np.isfinite(ball[:, 0])
    dist = [np.fmin.reduce(np.sqrt(((P - ball[:, None, :]) ** 2).sum(-1)), axis=1) for P in (home, away)]
    near = np.zeros(T, np.int8)                   # +1 home, -1 away, 0 nobody
    with np.errstate(invalid="ignore"):
        near[vis & (dist[0] < TOUCH_RADIUS) & ~(dist[1] < dist[0])] = 1
        near[vis & (dist[1] < TOUCH_RADIUS) & ~(dist[0] <= dist[1])] = -1
    frames, teams = [], []
    for t in np.flatnonzero(near):
        if frames and teams[-1] == near[t] and t - last <= TOUCH_MERGE:
            last = t
            continue
        frames.append(t)
        teams.append(int(near[t]))
        last = t
    out = np.zeros((T, 10), np.float32)
    out[:, 6:8] = 1.0                             # "a long time since / until a touch"
    if not frames:
        return out
    frames, teams = np.array(frames), np.array(teams, np.float32)
    n = len(frames)
    back, fwd = np.ones(n), np.ones(n)
    for j in range(1, n):
        if teams[j] == teams[j - 1]:
            back[j] = back[j - 1] + 1
    for j in range(n - 2, -1, -1):
        if teams[j] == teams[j + 1]:
            fwd[j] = fwd[j + 1] + 1
    t = np.arange(T)
    i = np.searchsorted(frames, t, side="right")  # touches [0, i) are at or before t
    for k in range(3):
        ok = i - 1 - k >= 0
        out[ok, k] = teams[i[ok] - 1 - k]
        ok = i + k < n
        out[ok, 3 + k] = teams[i[ok] + k]
    ok = i > 0
    out[ok, 6] = np.minimum((t[ok] - frames[i[ok] - 1]) / 250.0, 1.0)
    out[ok, 8] = np.minimum(back[i[ok] - 1] / 5.0, 1.0)
    ok = i < n
    out[ok, 7] = np.minimum((frames[i[ok]] - t[ok]) / 250.0, 1.0)
    out[ok, 9] = np.minimum(fwd[i[ok]] / 5.0, 1.0)
    return out


def motion_features(home, away):
    """9 columns: each team's velocity along its attacking direction, and how it is changing."""
    cols, short = [], []
    for side, P in ((1.0, home), (-1.0, away)):
        vel = _velocity(P) * side
        with np.errstate(all="ignore"):
            mean = np.nan_to_num(np.nanmean(vel[..., 0], axis=1))
            forward = np.nan_to_num(np.nanmean(np.where(np.isfinite(vel[..., 0]), vel[..., 0] > 1.0, np.nan), axis=1))
        v1 = uniform_filter1d(mean, 25)
        v3 = uniform_filter1d(mean, 75)
        change = np.zeros_like(v1)
        change[25:-25] = v1[50:] - v1[:-50]
        cols += [v1 / 5.0, v3 / 5.0, change / 5.0, forward]
        short.append(v1)
    cols.append((short[0] - short[1]) / 5.0)
    return np.stack(cols, 1).astype(np.float32)


def extra_features(ball, home, away):
    if not home_attacks_positive(home, away):
        flip = np.array([-1, -1], np.float32)
        ball, home, away = ball * flip, home * flip, away * flip
    return np.nan_to_num(np.column_stack([touch_features(ball, home, away), motion_features(home, away)]))


def make_seq(b, hp, ap_, y):
    change = np.zeros(len(y), bool)
    change[1:] = y[1:] != y[:-1]
    near = maximum_filter1d(change.astype(np.uint8), 2 * CHANGE_RADIUS + 1)
    return dict(X=frame_features(b, hp, ap_).astype(np.float16), extra=extra_features(b, hp, ap_).astype(np.float16),
                y=y, change=near.astype(np.int8))


def train_seqs(copies, seed, source):
    rng = np.random.default_rng(seed)
    out = []
    for mi in TRAIN:
        for ball3, home, away, status, poss in pg.load_dfl(mi, source):
            y = controlled(raw_codes(status, poss), pg.MIN_CONTROL)
            for c in range(copies):
                b, hp, ap_ = degrade(ball3, home, away, status, rng)
                yy = y.copy()
                if c % 2 == 1:
                    hp, ap_ = ap_, hp
                    yy = np.where(y == 2, 2, 1 - y).astype(np.int8)
                if rng.random() < 0.5:
                    flip = np.array([1, -1], np.float32)
                    b, hp, ap_ = b * flip, hp * flip, ap_ * flip
                out.append(make_seq(b, hp, ap_, yy))
        print("  match {} built ({} sequences)".format(pg.MATCHES[mi], len(out)), flush=True)
    return out


def held_out(indices, source):
    """Fixed degraded copies, seeded like solution/eval_fold.py."""
    out = []
    for mi in indices:
        for h, (ball3, home, away, status, poss) in enumerate(pg.load_dfl(mi, source), start=1):
            y = controlled(raw_codes(status, poss), pg.MIN_CONTROL)
            for c in range(2):
                b, hp, ap_ = degrade(ball3, home, away, status, np.random.default_rng(1000 * mi + 10 * h + c))
                out.append(make_seq(b, hp, ap_, y))
    return out


# =================================================================================================
#  2. RESZ -- HALO VALTAS-KIMENETTEL, IDOFUGGO VITERBI, HATAR-IGAZITAS
# =================================================================================================

class TwoHeadNet(PossessionNet):
    """PossessionNet plus an output that says: a change of state is close to this frame."""

    def __init__(self, n_in, hidden=128):
        super().__init__(n_in, hidden=hidden)
        self.change = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.inp(x)
        h = h + torch.relu(self.conv(h.transpose(1, 2))).transpose(1, 2)
        g, _ = self.gru(h)
        both = torch.cat([g, h], -1)
        return self.head(both), self.change(both).squeeze(-1)


def inputs(seq, use_extra):
    x = np.asarray(seq["X"], np.float32)
    return np.concatenate([x, np.asarray(seq["extra"], np.float32)], 1) if use_extra else x


def train_net(seqs, use_extra, steps, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_in = seqs[0]["X"].shape[1] + (N_EXTRA if use_extra else 0)
    net = TwoHeadNet(n_in).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    ce, bce = nn.CrossEntropyLoss(reduction="none"), nn.BCEWithLogitsLoss()
    started, crop = time.time(), 3000
    for step in range(steps):
        net.train()
        xs, ys, cs = [], [], []
        for _ in range(16):
            s = seqs[rng.integers(len(seqs))]
            a = rng.integers(0, len(s["X"]) - crop)
            xs.append(inputs({"X": s["X"][a:a + crop], "extra": s["extra"][a:a + crop]}, use_extra))
            ys.append(s["y"][a:a + crop].astype(np.int64))
            cs.append(s["change"][a:a + crop].astype(np.float32))
        x, y, c = (torch.from_numpy(np.stack(v)).to(DEVICE) for v in (xs, ys, cs))
        logits, change = net(x)
        per = ce(logits.reshape(-1, 3), y.reshape(-1))
        yn = y.cpu().numpy()
        edge = np.zeros_like(yn, dtype=bool)
        edge[:, 1:] = yn[:, 1:] != yn[:, :-1]
        w = torch.from_numpy(np.where(maximum_filter1d(edge.astype(np.uint8), 51, axis=1) > 0, 3.0, 1.0)
                             .astype(np.float32)).to(DEVICE).reshape(-1)
        loss = (per * w).sum() / w.sum() + LABEL_LOSS_CHANGE * bce(change, c)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps - 1:
            print("    step {} loss {:.4f} ({:.0f} s)".format(step, loss.item(), time.time() - started), flush=True)
    return net.eval()


@torch.no_grad()
def predict_both(net, X, chunk=6000, overlap=1000):
    """(probabilities (T, 3), change probability (T,)), overlapping windows as in solution/model.py."""
    T = len(X)
    p, c = np.zeros((T, 3), np.float32), np.zeros(T, np.float32)
    step, start = chunk - 2 * overlap, 0
    while start < T:
        a, b = max(0, start - overlap), min(T, start + step + overlap)
        logits, change = net(torch.from_numpy(X[a:b])[None].to(DEVICE))
        n = min(step, T - start)
        p[start:start + n] = torch.softmax(logits[0], -1).cpu().numpy()[start - a:start - a + n]
        c[start:start + n] = torch.sigmoid(change[0]).cpu().numpy()[start - a:start - a + n]
        start += step
    return p, c


def viterbi_tv(p, change, switch, alive, alpha):
    """Viterbi over H/A/D where every change at frame t costs (1 - alpha * change[t]) times its base."""
    lp = np.log(np.clip(p, 1e-6, 1)).tolist()
    factor = (1.0 - alpha * change).tolist()
    n = len(lp)
    back = bytearray(3 * n)
    s0, s1, s2 = lp[0]
    for t in range(1, n):
        e0, e1, e2 = lp[t]
        sw, de = switch * factor[t], alive * factor[t]
        a, b, c = s0, s1 - sw, s2 - de
        if a >= b and a >= c:
            n0, p0 = a, 0
        elif b >= c:
            n0, p0 = b, 1
        else:
            n0, p0 = c, 2
        a, b, c = s0 - sw, s1, s2 - de
        if b >= a and b >= c:
            n1, p1 = b, 1
        elif a >= c:
            n1, p1 = a, 0
        else:
            n1, p1 = c, 2
        a, b, c = s0 - de, s1 - de, s2
        if c >= a and c >= b:
            n2, p2 = c, 2
        elif a >= b:
            n2, p2 = a, 0
        else:
            n2, p2 = b, 1
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = p0, p1, p2
        s0, s1, s2 = n0 + e0, n1 + e1, n2 + e2
    path = np.empty(n, np.int8)
    state = int(np.argmax([s0, s1, s2]))
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[3 * t + state]
    return path


def snap_boundaries(path, change, window):
    """Move every change of state to the peak of the change output within +-window frames."""
    if window <= 0:
        return path
    edges = np.flatnonzero(path[1:] != path[:-1]) + 1
    if not len(edges):
        return path
    moved = edges.copy()
    for j, e in enumerate(edges):
        lo = max(e - window, moved[j - 1] + 1 if j else 1)
        hi = min(e + window, edges[j + 1] - 1 if j + 1 < len(edges) else len(path) - 1)
        if lo <= hi:
            moved[j] = lo + int(np.argmax(change[lo:hi + 1]))
    out = path.copy()
    starts = np.concatenate([[0], moved])
    states = np.concatenate([[path[0]], path[edges]])
    stops = np.concatenate([moved, [len(path)]])
    for a, b, s in zip(starts, stops, states):
        out[a:b] = s
    return out


# =================================================================================================
#  3. RESZ -- MERES
# =================================================================================================

def ensemble(nets, seqs, use_extra):
    out = []
    for s in seqs:
        x = inputs(s, use_extra)
        both = [predict_both(n, x) for n in nets]
        out.append((np.mean([b[0] for b in both], 0), np.mean([b[1] for b in both], 0), s["y"]))
    return out


def spells(preds, sw, al, alpha=0.0, window=0):
    """spellF1 per sequence under one decoder setting."""
    scores = []
    for p, c, y in preds:
        path = viterbi_tv(p, c, sw, al, alpha)
        scores.append(pg.spell_macro(y, snap_boundaries(path, c, window)))
    return np.array(scores)


def best_setting(val, alphas, windows):
    """The decoder setting with the best mean spellF1 on the tuning match."""
    best = None
    for sw in GRID_SWITCH:
        for al in GRID_ALIVE:
            for alpha in alphas:
                for w in windows:
                    score = spells(val, sw, al, alpha, w)[:, 0].mean()
                    if best is None or score > best[0]:
                        best = (score, sw, al, alpha, w)
    return best[1:]


def run(args):
    os.makedirs(pg.OUT, exist_ok=True)
    print("building sequences on {} ...".format(DEVICE), flush=True)
    seqs = train_seqs(args.copies, 0, args.source)
    val_seqs, test_seqs = held_out(VAL, args.source), held_out(TEST, args.source)

    results, baseline = {}, None
    for name, use_extra in (("old 48 features", False), ("+ touch and motion features", True)):
        nets = []
        for seed in range(args.seeds):
            print("{}, seed {}".format(name, seed), flush=True)
            nets.append(train_net(seqs, use_extra, args.steps, seed))
        val, test = ensemble(nets, val_seqs, use_extra), ensemble(nets, test_seqs, use_extra)
        stages = [("fixed cost", (0.0,), (0,))]
        if use_extra:
            stages += [("+ variable switch cost", GRID_ALPHA, (0,)),
                       ("+ variable cost + boundary snap", GRID_ALPHA, GRID_SNAP)]
        for stage, alphas, windows in stages:
            setting = best_setting(val, alphas, windows)
            scores = spells(test, *setting)
            label = "{} | {}".format(name, stage)
            results[label] = dict(spell=float(scores[:, 0].mean()), macro=float(scores[:, 1].mean()),
                                  per_copy=scores[:, 0].tolist(), setting=list(setting))
            if baseline is None:
                baseline = scores[:, 0]
            better = int((scores[:, 0] > baseline).sum())
            print("  {}: spellF1 {:.4f}  macroF1 {:.4f}  delta {:+.4f}  better on {}/{} copies  setting {}".format(
                label, scores[:, 0].mean(), scores[:, 1].mean(), scores[:, 0].mean() - baseline.mean(),
                better, len(baseline), setting), flush=True)
    with open(os.path.join(pg.OUT, "report2.json"), "w") as handle:
        json.dump(results, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("run", "verify"))
    parser.add_argument("--source", default=pg.DFL_SOURCE)
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        pg.verify()


if __name__ == "__main__":
    main()
