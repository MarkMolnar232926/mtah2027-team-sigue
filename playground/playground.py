"""Playground on top of solution/: ball trajectory inference from the players. Changes nothing outside playground/.

    playground/.venv/Scripts/python playground/playground.py oracle   # how much would a perfect ball be worth?
    playground/.venv/Scripts/python playground/playground.py run      # train BallNet, A/B on the DFL benchmark
    playground/.venv/Scripts/python playground/playground.py verify   # are the old files untouched?

Rules (organisers' clarification of 2026-11-03): this file reads external data only (the 7 DFL
matches) and never touches the provided game, its ground truth or its video.
"""

# =================================================================================================
#  ATTEKINTES (a csapatnak)
# -------------------------------------------------------------------------------------------------
#  Miert kulon mappa: minden, ami itt keszul, a playground/out/ mappaba kerul. A solution/ kodjat
#  csak importaljuk (degrade, features, PossessionNet, viterbi, a kesz sulyok), nem irjuk at. A
#  `verify` parancs ujjlenyomattal ellenorzi, hogy a playground/ mappan kivul semmi nem valtozott.
#
#  Mit ad hozza: a labda a meccs 41%-an hianyzik, gyakran hosszan. A BallNet a 22 jatekos
#  mozgasabol es a labda ismert darabjaibol megbecsli, hol van a labda A KIESESEK ALATT IS, es
#  azt is, mennyire biztos benne.
#
#    1. Tanitas: a DFL-en a valodi labda ismert. A degrade() elrontja a bemenetet, a halo pedig
#       a tiszta labdat tanulja meg visszaallitani.
#    2. Felhasznalas: a becsult labdabol 11 uj jellemzo lesz (hol van, mennyire bizonytalan, ki van
#       hozza kozel), ezek a regi 48 melle kerulnek, es a PossessionNet ezekkel tanul ujra.
#    3. Keresztillesztes: a birtoklas-halo tanitasahoz minden meccsre olyan BallNet ad becslest,
#       amely azt a meccset nem latta. Kulonben a birtoklas-halo tulsagosan megbizna benne.
#    4. A/B meres: ugyanazon a 5 meccsen tanitjuk a regi (48) es az uj (59 jellemzo) halot, ugyanazzal
#       a seed-del, es a repo fix DFL benchmarkjan (2 felretett meccs) hasonlitjuk ossze.
#
#  Korlat: itt most csak a DFL van meg, a SkillCorner es a Metrica nem. A szamok ezert a DFL-re
#  ervenyesek, es nem kozvetlenul hasonlithatok a repo vegleges, harom forrason tanitott modelljehez.
# =================================================================================================

import argparse
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True  # no __pycache__ next to the old code

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, _path)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.ndimage import maximum_filter1d  # noqa: E402

import possession_metrics as pm  # noqa: E402
from degrade import degrade  # noqa: E402
from dfl_labels import controlled, raw_codes  # noqa: E402
from eval_external import load_models  # noqa: E402
from eval_fold import probs  # noqa: E402
from features import (HALF_L, HALF_W, _bfill_idx, _ffill_idx, _velocity, frame_features,  # noqa: E402
                      home_attacks_positive)
from model import PossessionNet, predict, viterbi  # noqa: E402

OUT = os.path.join(HERE, "out")
MANIFEST = os.path.join(HERE, "baseline_manifest.md5")
DFL_SOURCE = r"D:\AI\BARCELONA\more-than-a-hack-2027\data\external\dfl"
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")  # sorted, as in the repo
TRAIN, TEST = (0, 1, 2, 3, 4), (5, 6)      # the repo's DFL benchmark holds out matches 5 and 6
FOLDS = ((0, 2, 4), (1, 3))                # cross-fitting of BallNet over the training matches
SWITCH_HA, SWITCH_ALIVE = 16, 24           # models/decoder.json
BENCHMARK_MODELS = ["models/expB_bw3met6k_s%d.pt" % s for s in range(3)]  # never saw the benchmark

STEP = 5                  # BallNet runs at 5 Hz, the estimate is interpolated back to 25 Hz
K_PLAYERS = 16            # players kept per team and frame
GAP_CAP_S = 20.0
N_BALL_IN, N_PLAYER_IN = 7, 6
N_EXTRA = 11
MIN_CONTROL = 2.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =================================================================================================
#  1. RESZ -- ADAT: a mar letoltott 7 DFL-meccs a solution/ formatumaban
# =================================================================================================

def load_dfl(index, source=DFL_SOURCE):
    """[(ball3, home, away, status, poss)] per half, metres, pitch centre = origin."""
    z = np.load(os.path.join(source, MATCHES[index] + ".npz"))
    length, width = z["pitch"]
    centre = np.array([length / 2, width / 2], np.float32)
    teams = z["teams"]
    halves = []
    for _, first, last in z["halves"]:
        part = slice(int(first), int(last) + 1)
        xy = np.stack([z["bx"][part], z["by"][part]], 1) - centre
        ball3 = np.column_stack([xy, np.nan_to_num(z["bz"][part])]).astype(np.float32)
        players = np.stack([z["px"][part], z["py"][part]], 2) - centre
        status = z["alive"][part].astype(np.int8)
        poss = np.where(z["owner"][part] == 0, 1, 2).astype(np.int8)  # DFL: 1 home, 2 away
        halves.append((ball3, players[:, teams == 0], players[:, teams == 1], status, poss))
    return halves


def spell_macro(y, path):
    """(spellF1, macroF1) of one half, with the official definitions."""
    y, path = y.astype(np.int8), path.astype(np.int8)
    runs = pm.valid_runs(y)
    spell = pm.spell_scores(y, pm.fill_abstentions(path, runs), runs)["mean_f1"]
    return spell, pm.macro_f1(pm.scored_confusion(pm.confusion(y, path)))


def decode(p):
    return viterbi(np.log(np.clip(p, 1e-6, 1)), SWITCH_HA, SWITCH_ALIVE)


def benchmark_copies(source):
    """The repo's fixed degraded copies of the held-out halves (same seeds as eval_fold.py)."""
    for mi in TEST:
        for h, (ball3, home, away, status, poss) in enumerate(load_dfl(mi, source), start=1):
            y = controlled(raw_codes(status, poss), MIN_CONTROL)
            for c in range(2):
                rng = np.random.default_rng(1000 * mi + 10 * h + c)
                b, hp, ap_ = degrade(ball3, home, away, status, rng)
                yield ball3, b, hp, ap_, y


# =================================================================================================
#  2. RESZ -- ORACLE: mennyit erne a tokeletes labda?
# -------------------------------------------------------------------------------------------------
#  A repo mar betanitott modelljei (ezek nem lattak a benchmarkot) kapnak egyszer zajos labdat,
#  egyszer a valodit. A kulonbseg a labdainformacio felso korlatja: ennel a BallNet nem hozhat
#  tobbet, es ha ez kicsi, akkor nem is erdemes vele foglalkozni.
# =================================================================================================

def oracle(source):
    nets = load_models(BENCHMARK_MODELS, DEVICE)
    noisy, perfect = [], []
    for ball3, b, hp, ap_, y in benchmark_copies(source):
        noisy.append(spell_macro(y, decode(probs(nets, b, hp, ap_, DEVICE, True))))
        perfect.append(spell_macro(y, decode(probs(nets, ball3[:, :2], hp, ap_, DEVICE, True))))
        print("  noisy ball {:.3f}   true ball {:.3f}".format(noisy[-1][0], perfect[-1][0]), flush=True)
    noisy, perfect = np.mean(noisy, 0), np.mean(perfect, 0)
    print("\nDFL benchmark, existing models:")
    print("  degraded ball: spellF1 {:.4f}  macroF1 {:.4f}".format(*noisy))
    print("  true ball:     spellF1 {:.4f}  macroF1 {:.4f}".format(*perfect))
    print("  ceiling of better ball information: {:+.4f} spellF1".format(perfect[0] - noisy[0]))


# =================================================================================================
#  3. RESZ -- BALLNET: a labda becslese a jatekosokbol
# -------------------------------------------------------------------------------------------------
#  Bemenet 5 Hz-en: a labda ismert darabjai (lathato-e, hol van, mennyi ideje nem latszik, linearis
#  interpolacio) es az osszes jatekos pozicioja + sebessege. A jatekosok sorrendje nem szamit
#  (halmaz-kodolo: minden jatekost ugyanaz a kis halo dolgoz fel, utana atlag es maximum
#  csapatonkent), mert az ATD-ben nincs jatekos-azonosito. Egy ketiranyu GRU idoben is osszekoti.
#  Kimenet: eltolas az interpolalt labdahoz kepest + bizonytalansag (Gauss-veszteseg).
# =================================================================================================

class BallNet(nn.Module):
    def __init__(self, hidden=128, player=64):
        super().__init__()
        self.player = nn.Sequential(nn.Linear(N_PLAYER_IN, player), nn.GELU(),
                                    nn.Linear(player, player), nn.GELU())
        self.inp = nn.Sequential(nn.Linear(N_BALL_IN + 4 * player, hidden), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * hidden, 3)

    def forward(self, ball, players, mask):
        h = self.player(players)                                   # (B, T, 2, K, player)
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(3) / m.sum(3).clamp(min=1)
        peak = (h - 1e6 * (1 - m)).max(3).values
        peak = peak.where(m.sum(3) > 0, torch.zeros_like(peak))
        pooled = torch.cat([mean, peak], -1).flatten(2)            # (B, T, 4 * player)
        g, _ = self.gru(self.inp(torch.cat([ball, pooled], -1)))
        out = self.head(g)
        return 20.0 * out[..., :2], out[..., 2].clamp(-1.0, 4.0)   # metres, log sigma


def canonical_sign(home, away):
    return 1.0 if home_attacks_positive(home, away) else -1.0


def ball_inputs(ball, home, away):
    """5 Hz BallNet inputs: (ball features, player features, mask, interpolated ball in metres)."""
    sign = canonical_sign(home, away)
    ball, home, away = ball * sign, home * sign, away * sign
    T = len(ball)
    vis = np.isfinite(ball[:, 0])
    f, b = _ffill_idx(vis), _bfill_idx(vis)
    t = np.arange(T)
    since = np.where(f >= 0, t - f, GAP_CAP_S * 25) / 25.0
    until = np.where(b < T, b - t, GAP_CAP_S * 25) / 25.0
    last = np.where(f[:, None] >= 0, ball[np.clip(f, 0, T - 1)], 0.0)
    nxt = np.where(b[:, None] < T, ball[np.clip(b, 0, T - 1)], 0.0)
    w = np.where((f >= 0) & (b < T), since / np.maximum(since + until, 1e-6), 0.0)[:, None]
    interp = np.where(vis[:, None], ball, np.where(
        (f >= 0)[:, None] & (b < T)[:, None], last * (1 - w) + nxt * w,
        np.where((f >= 0)[:, None], last, nxt)))
    interp = np.nan_to_num(interp).astype(np.float32)

    idx = np.arange(0, T, STEP)
    scale = np.array([HALF_L, HALF_W], np.float32)
    ball_x = np.column_stack([
        vis[idx].astype(np.float32),
        np.nan_to_num(np.where(vis[:, None], ball, 0.0)[idx] / scale),
        np.minimum(since[idx], GAP_CAP_S) / GAP_CAP_S, np.minimum(until[idx], GAP_CAP_S) / GAP_CAP_S,
        interp[idx] / scale])

    players, masks = [], []
    for P in (home, away):
        vel = _velocity(P)[idx]
        P = P[idx]
        seen = np.isfinite(P[..., 0])
        order = np.argsort(~seen, axis=1, kind="stable")[:, :K_PLAYERS]
        row = np.arange(len(idx))[:, None]
        pos, v, s = P[row, order], vel[row, order], seen[row, order]
        rel = (pos - interp[idx][:, None, :]) / 20.0
        one = np.concatenate([pos / scale, np.nan_to_num(v) / 8.0, rel], -1)
        one = np.nan_to_num(one)
        if one.shape[1] < K_PLAYERS:
            pad = K_PLAYERS - one.shape[1]
            one = np.pad(one, ((0, 0), (0, pad), (0, 0)))
            s = np.pad(s, ((0, 0), (0, pad)))
        players.append(one)
        masks.append(s)
    return (ball_x.astype(np.float32), np.stack(players, 1).astype(np.float32),
            np.stack(masks, 1), interp, sign)


def build_sequences(indices, copies, seed, source):
    """Degraded + augmented copies of the given matches, with everything both networks need."""
    rng = np.random.default_rng(seed)
    out = []
    for mi in indices:
        for ball3, home, away, status, poss in load_dfl(mi, source):
            y = controlled(raw_codes(status, poss), MIN_CONTROL)
            for c in range(copies):
                b, hp, ap_ = degrade(ball3, home, away, status, rng)
                yy, truth = y.copy(), ball3[:, :2].copy()
                if c % 2 == 1:   # team swap
                    hp, ap_ = ap_, hp
                    yy = np.where(y == 2, 2, 1 - y).astype(np.int8)
                if rng.random() < 0.5:   # mirror across the long axis
                    flip = np.array([1, -1], np.float32)
                    b, hp, ap_, truth = b * flip, hp * flip, ap_ * flip, truth * flip
                out.append(make_sequence(mi, b, hp, ap_, yy, truth))
        print("  match {} built ({} sequences)".format(MATCHES[mi], len(out)), flush=True)
    return out


def make_sequence(mi, b, hp, ap_, y, truth):
    bx, px, mask, interp, sign = ball_inputs(b, hp, ap_)
    idx = np.arange(0, len(b), STEP)
    return dict(match=mi, X=frame_features(b, hp, ap_).astype(np.float16), y=y, hp=hp, ap=ap_,
                ball_x=bx.astype(np.float16), players=px.astype(np.float16), mask=mask,
                interp=interp[idx], hidden=~np.isfinite(b[idx, 0]), sign=sign,
                target=(truth[idx] * sign).astype(np.float32))


def train_ball_net(seqs, steps, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = BallNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    started = time.time()
    crop = 300  # 60 s at 5 Hz
    for step in range(steps):
        batch = [seqs[rng.integers(len(seqs))] for _ in range(16)]
        starts = [rng.integers(0, len(s["ball_x"]) - crop) for s in batch]
        take = lambda key, kind: torch.from_numpy(np.stack(  # noqa: E731
            [np.asarray(s[key][a:a + crop], kind) for s, a in zip(batch, starts)])).to(DEVICE)
        ball, players, mask = take("ball_x", np.float32), take("players", np.float32), take("mask", bool)
        interp, target, hidden = take("interp", np.float32), take("target", np.float32), take("hidden", bool)
        delta, log_sigma = net(ball, players, mask)
        err2 = ((interp + delta - target) ** 2).sum(-1)
        nll = 0.5 * err2 * torch.exp(-2 * log_sigma) + 2 * log_sigma
        weight = torch.where(hidden, torch.ones_like(nll), 0.2 * torch.ones_like(nll))
        weight = weight * torch.isfinite(target[..., 0]).float()
        loss = (torch.nan_to_num(nll) * weight).sum() / weight.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            print("    ballnet step {} loss {:.3f} ({:.0f} s)".format(
                step, loss.item(), time.time() - started), flush=True)
    return net.eval()


@torch.no_grad()
def ball_estimate(net, seq, chunk=1500, overlap=250):
    """(delta, log sigma) over a whole 5 Hz sequence, in overlapping windows."""
    T = len(seq["ball_x"])
    delta, log_sigma = np.zeros((T, 2), np.float32), np.zeros(T, np.float32)
    step, start = chunk - 2 * overlap, 0
    while start < T:
        a, b = max(0, start - overlap), min(T, start + step + overlap)
        d, s = net(torch.from_numpy(np.asarray(seq["ball_x"][a:b], np.float32))[None].to(DEVICE),
                   torch.from_numpy(np.asarray(seq["players"][a:b], np.float32))[None].to(DEVICE),
                   torch.from_numpy(seq["mask"][a:b])[None].to(DEVICE))
        n = min(step, T - start)
        delta[start:start + n] = d[0, start - a:start - a + n].cpu().numpy()
        log_sigma[start:start + n] = s[0, start - a:start - a + n].cpu().numpy()
        start += step
    return delta, log_sigma


def extra_features(seq, delta, log_sigma):
    """The 11 new per-frame features from a BallNet estimate, at 25 Hz."""
    T = len(seq["y"]) if "y" in seq else len(seq["X"])
    coarse = np.arange(len(delta)) * STEP
    fine = np.arange(T)
    est5 = seq["interp"] + delta
    est = np.column_stack([np.interp(fine, coarse, est5[:, 0]), np.interp(fine, coarse, est5[:, 1])])
    shift = np.interp(fine, coarse, np.hypot(delta[:, 0], delta[:, 1]))
    sigma = np.interp(fine, coarse, log_sigma)
    sign = seq["sign"]
    cols = [est[:, 0] / HALF_L, est[:, 1] / HALF_W, sigma / 4.0, np.minimum(shift, 40.0) / 20.0]
    nearest = []
    for P in (seq["hp"], seq["ap"]):
        d = np.sqrt(((P * sign - est[:, None, :]) ** 2).sum(-1))
        d = np.where(np.isfinite(d), d, 99.0)
        two = np.sort(np.partition(d, 1, axis=1)[:, :2], axis=1)
        nearest.append(np.minimum(two[:, 0], 40.0))
        cols += [nearest[-1] / 10.0, np.minimum(two[:, 1], 40.0) / 10.0, (d < 5).sum(1) / 3.0]
    cols.append((nearest[0] - nearest[1]) / 10.0)
    return np.nan_to_num(np.stack(cols, 1)).astype(np.float16)


# =================================================================================================
#  4. RESZ -- A/B: regi 48 jellemzo vs 48 + 11 BallNet-jellemzo
# =================================================================================================

def train_possession(seqs, extra_key, steps, seed):
    """Same recipe as solution/train.py: 2-minute crops, boundary-weighted cross-entropy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_in = seqs[0]["X"].shape[1] + (N_EXTRA if extra_key else 0)
    net = PossessionNet(n_in, hidden=128).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(reduction="none")
    started, crop = time.time(), 3000
    for step in range(steps):
        net.train()
        xs, ys = [], []
        for _ in range(16):
            s = seqs[rng.integers(len(seqs))]
            a = rng.integers(0, len(s["X"]) - crop)
            x = np.asarray(s["X"][a:a + crop], np.float32)
            if extra_key:
                x = np.concatenate([x, np.asarray(s[extra_key][a:a + crop], np.float32)], 1)
            xs.append(x)
            ys.append(s["y"][a:a + crop].astype(np.int64))
        x, y = torch.from_numpy(np.stack(xs)).to(DEVICE), torch.from_numpy(np.stack(ys)).to(DEVICE)
        per = lossf(net(x).reshape(-1, 3), y.reshape(-1))
        yn = y.cpu().numpy()
        change = np.zeros_like(yn, dtype=bool)
        change[:, 1:] = yn[:, 1:] != yn[:, :-1]
        near = maximum_filter1d(change.astype(np.uint8), 51, axis=1)
        w = torch.from_numpy(np.where(near > 0, 3.0, 1.0).astype(np.float32)).to(DEVICE).reshape(-1)
        loss = (per * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps - 1:
            print("    possession step {} loss {:.4f} ({:.0f} s)".format(
                step, loss.item(), time.time() - started), flush=True)
    return net.eval()


def run(args):
    os.makedirs(OUT, exist_ok=True)
    print("building training sequences on {} ...".format(DEVICE), flush=True)
    seqs = build_sequences(TRAIN, args.copies, 0, args.source)

    # cross-fitting: every training match gets estimates from a BallNet that never saw it
    ball_nets = []
    for k, fold in enumerate(FOLDS):
        print("BallNet {}/{}, trained without matches {}".format(k + 1, len(FOLDS), fold), flush=True)
        net = train_ball_net([s for s in seqs if s["match"] not in fold], args.steps_ball, k)
        torch.save(net.state_dict(), os.path.join(OUT, "ballnet_fold%d.pt" % k))
        ball_nets.append(net)
        for s in seqs:
            if s["match"] in fold:
                s["extra"] = extra_features(s, *ball_estimate(net, s))

    print("benchmark copies ...", flush=True)
    tests = []
    errs_interp, errs_net = [], []
    for ball3, b, hp, ap_, y in benchmark_copies(args.source):
        seq = make_sequence(TEST[0], b, hp, ap_, y, ball3[:, :2])
        estimates = [ball_estimate(net, seq) for net in ball_nets]
        delta = np.mean([e[0] for e in estimates], 0)
        log_sigma = np.mean([e[1] for e in estimates], 0)
        seq["extra"] = extra_features(seq, delta, log_sigma)
        hidden = seq["hidden"] & np.isfinite(seq["target"][:, 0])
        errs_interp.append(np.hypot(*(seq["interp"] - seq["target"])[hidden].T))
        errs_net.append(np.hypot(*(seq["interp"] + delta - seq["target"])[hidden].T))
        tests.append(seq)
    errs_interp, errs_net = np.concatenate(errs_interp), np.concatenate(errs_net)
    print("\nball error on hidden frames (held-out matches, metres):")
    print("  linear interpolation: median {:.1f}  mean {:.1f}".format(np.median(errs_interp), errs_interp.mean()))
    print("  BallNet:              median {:.1f}  mean {:.1f}".format(np.median(errs_net), errs_net.mean()))

    report = dict(ball_median_interp=float(np.median(errs_interp)), ball_median_net=float(np.median(errs_net)))
    for name, key in (("A: old 48 features", None), ("B: 48 + BallNet", "extra")):
        nets = []
        for seed in range(args.seeds):
            print("{}, seed {}".format(name, seed), flush=True)
            nets.append(train_possession(seqs, key, args.steps_poss, seed))
        scores = []
        for s in tests:
            x = np.asarray(s["X"], np.float32)
            if key:
                x = np.concatenate([x, np.asarray(s[key], np.float32)], 1)
            p = np.mean([np.exp(predict(n, x, DEVICE)) for n in nets], 0)
            scores.append(spell_macro(s["y"], decode(p)))
        spell, macro = np.mean(scores, 0)
        report[name] = dict(spell=float(spell), macro=float(macro))
        print("  {}: benchmark spellF1 {:.4f}  macroF1 {:.4f}".format(name, spell, macro), flush=True)
    with open(os.path.join(OUT, "report.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print("\nA -> B: {:+.4f} spellF1".format(
        report["B: 48 + BallNet"]["spell"] - report["A: old 48 features"]["spell"]))


def verify():
    """Compare every file outside playground/ with the fingerprints taken before it existed."""
    bad = 0
    with open(MANIFEST) as handle:
        rows = [line.rstrip("\n").split(None, 1) for line in handle]
    listed = [(digest, os.path.normpath(name.lstrip("*"))) for digest, name in rows]
    for digest, name in listed:
        path = os.path.join(ROOT, name)
        current = hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.isfile(path) else None
        if current != digest:
            bad += 1
            print("CHANGED: " + name)
    known = {name for _, name in listed}
    for folder, dirs, files in os.walk(ROOT):
        rel = os.path.relpath(folder, ROOT)
        dirs[:] = [d for d in dirs if not (rel == "." and d in ("playground", ".git")) and d != "__pycache__"]
        for name in files:
            path = os.path.normpath(os.path.join(rel, name))
            if path not in known:
                bad += 1
                print("NEW: " + path)
    print("{} files checked: {}".format(len(listed), "all unchanged" if not bad else "%d differences" % bad))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("oracle", "run", "verify"))
    parser.add_argument("--source", default=DFL_SOURCE, help="folder with the cached DFL J03*.npz files")
    parser.add_argument("--copies", type=int, default=8, help="degraded copies of each training half")
    parser.add_argument("--steps-ball", type=int, default=6000)
    parser.add_argument("--steps-poss", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    if args.cmd == "oracle":
        oracle(args.source)
    elif args.cmd == "run":
        run(args)
    else:
        verify()


if __name__ == "__main__":
    main()
