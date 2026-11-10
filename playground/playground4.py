"""Playground 4: the strongest recipe measured so far, plus an audit of what still costs spellF1.

    playground/.venv/Scripts/python playground/playground4.py run      # train, tune on match 4, score the benchmark
    playground/.venv/Scripts/python playground/playground4.py verify   # are the old files untouched?

Reads external data only (the 7 DFL matches); never the provided game, its ground truth or video.
"""

# =================================================================================================
#  ATTEKINTES (a csapatnak)
# -------------------------------------------------------------------------------------------------
#  Cel: a leheto legmagasabb spellF1 a DFL-benchmarkon. Csak azt rakjuk ossze, aminek van alapja:
#
#    - 48 + 19 jellemzo (playground2). A BallNet kimaradt, mert a playground3 szerint nem adott hozza.
#    - Tobb tanito adat: 5 meccs (0-4) a 4 helyett, es felidonkent 16 rontott masolat a 8 helyett.
#    - Nagyobb ensemble: 5 seed a 3 helyett.
#    - TTA: minden masolatot 4 nezetben josolunk (eredeti, tukrozott, csapatcsere, mindketto).
#
#  Hangolas: a dekoder-buntetesekrol kulon, csak a 0-3 meccsen tanult ket halo donthet a 4-es meccsen.
#  A vegso halok latjak a 4-es meccset is, a 5-6 teszt-meccset senki. A teszt csak a vegen ad szamot.
#
#  A kimenet harom dolgot mutat:
#    1. mennyit hoz a tobb seed es a TTA kulon-kulon,
#    2. a plafont: ugyanezek a halok valodi labdaval mennyit ernenek el,
#    3. hibaaudit: milyen hosszu birtoklasokat veszitunk el, es rossz csapat vagy rossz hatar miatt-e.
# =================================================================================================

import argparse
import json
import os
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import playground as pg  # noqa: E402  (also puts solution/ and evaluation/ on the path)
import playground2 as p2  # noqa: E402
import possession_metrics as pm  # noqa: E402
from degrade import degrade  # noqa: E402
from dfl_labels import controlled, raw_codes  # noqa: E402

FINAL_TRAIN = (0, 1, 2, 3, 4)
SHORT_SEEDS = 3                      # the first stage: as many seeds as playground2
BUCKETS = ((0, 4), (4, 8), (8, 16), (16, 1e9))   # seconds
AUDIT_IOU = 0.5


def train_seqs(copies, seed, source):
    rng = np.random.default_rng(seed)
    out = []
    for mi in FINAL_TRAIN:
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
                seq = p2.make_seq(b, hp, ap_, yy)
                seq["match"] = mi
                out.append(seq)
        print("  match {} built ({} sequences)".format(pg.MATCHES[mi], len(out)), flush=True)
    return out


def held_out(indices, source):
    """The fixed degraded copies of playground2.held_out, in the same order."""
    out = []
    for mi in indices:
        for h, (ball3, home, away, status, poss) in enumerate(pg.load_dfl(mi, source), start=1):
            y = controlled(raw_codes(status, poss), pg.MIN_CONTROL)
            for c in range(2):
                b, hp, ap_ = degrade(ball3, home, away, status, np.random.default_rng(1000 * mi + 10 * h + c))
                out.append(dict(b=b, hp=hp, ap=ap_, truth=ball3[:, :2].astype(np.float32), y=y))
    return out


def view_inputs(b, hp, ap_):
    """All four views of one copy: [(X, teams swapped?)]."""
    flip = np.array([1, -1], np.float32)
    views = [(b, hp, ap_, False), (b * flip, hp * flip, ap_ * flip, False),
             (b, ap_, hp, True), (b * flip, ap_ * flip, hp * flip, True)]
    out = []
    for bb, hh, aa, swapped in views:
        s = p2.make_seq(bb, hh, aa, np.zeros(len(bb), np.int8))
        out.append((p2.inputs(s, True), swapped))
    return out


def per_net_probs(nets, b, hp, ap_, tta):
    """Probabilities (nets, T, 3); with tta each net's output is the mean over the four views."""
    views = view_inputs(b, hp, ap_)
    if not tta:
        views = views[:1]
    out = []
    for net in nets:
        one = []
        for x, swapped in views:
            p = p2.predict_both(net, x)[0]
            one.append(p[:, [1, 0, 2]] if swapped else p)
        out.append(np.mean(one, 0))
    return np.stack(out)


def as_preds(probs, ys):
    """The (p, change, y) triples playground2.spells expects, with a change output that is never used."""
    return [(p, np.zeros(len(p), np.float32), y) for p, y in zip(probs, ys)]


def audit(pairs):
    """What the decoded output gets wrong, over all copies: [(y, path)]."""
    rows = {b: [0, 0, 0] for b in BUCKETS}         # recovered, wrong team, wrong boundaries
    counts = {t: [0, 0, 0] for t in pm.SPELL_IOUS}  # matched, predicted, true
    for y, path in pairs:
        y, path = y.astype(np.int8), path.astype(np.int8)
        runs = pm.valid_runs(y)
        filled = pm.fill_abstentions(path, runs)
        scores = pm.spell_scores(y, filled, runs)
        for t, v in scores["per_iou"].items():
            counts[t][0] += v["n_matched"]
            counts[t][1] += v["n_matched"] + v["n_spurious"]
            counts[t][2] += v["n_matched"] + v["n_missed"]
        pred = pm.possession_spells(filled, runs)
        for a, b, code in pm.possession_spells(y, runs):
            same, other = 0.0, 0.0
            for c, d, q in pred:
                inter = min(b, d) - max(a, c)
                if inter > 0:
                    iou = inter / float(max(b, d) - min(a, c))
                    if q == code:
                        same = max(same, iou)
                    else:
                        other = max(other, iou)
            seconds = (b - a) / 25.0
            row = next(rows[k] for k in BUCKETS if k[0] <= seconds < k[1])
            row[0 if same >= AUDIT_IOU else 1 if other >= AUDIT_IOU else 2] += 1
    return rows, counts


def run(args):
    os.makedirs(pg.OUT, exist_ok=True)
    print("building sequences on {} ...".format(p2.DEVICE), flush=True)
    seqs = train_seqs(args.copies, 0, args.source)
    val_set, test_set = held_out(p2.VAL, args.source), held_out(p2.TEST, args.source)

    # the decoder is chosen by nets that have not seen the tuning match
    tune_nets = []
    for seed in range(args.tune_seeds):
        print("tuning net, seed {} (matches 0-3)".format(seed), flush=True)
        tune_nets.append(p2.train_net([s for s in seqs if s["match"] in p2.TRAIN], True, args.steps, seed))
    val_probs = [per_net_probs(tune_nets, s["b"], s["hp"], s["ap"], True).mean(0) for s in val_set]
    setting = p2.best_setting(as_preds(val_probs, [s["y"] for s in val_set]), (0.0,), (0,))
    print("decoder setting from match 4: switch {} alive {}".format(*setting[:2]), flush=True)
    del tune_nets

    nets = []
    for seed in range(args.seeds):
        print("final net, seed {} (matches 0-4)".format(seed), flush=True)
        nets.append(p2.train_net(seqs, True, args.steps, seed))
    del seqs

    ys = [s["y"] for s in test_set]
    plain = [per_net_probs(nets, s["b"], s["hp"], s["ap"], False) for s in test_set]
    tta = [per_net_probs(nets, s["b"], s["hp"], s["ap"], True) for s in test_set]
    ceiling = [per_net_probs(nets, s["truth"], s["hp"], s["ap"], True) for s in test_set]

    n_short = min(SHORT_SEEDS, args.seeds)
    stages = (("5 matches, 16 copies, 3 seeds", [p[:n_short].mean(0) for p in plain]),
              ("+ 5 seeds", [p.mean(0) for p in plain]),
              ("+ 4-view TTA", [p.mean(0) for p in tta]),
              ("true ball (ceiling, not achievable)", [p.mean(0) for p in ceiling]))

    with open(os.path.join(pg.OUT, "report2.json")) as handle:
        earlier = json.load(handle)
    base48 = np.array(earlier["old 48 features | fixed cost"]["per_copy"])
    base67 = np.array(earlier["+ touch and motion features | fixed cost"]["per_copy"])
    report = dict(setting=list(setting))
    for name, probs in stages:
        scores = p2.spells(as_preds(probs, ys), *setting)
        spell = scores[:, 0]
        report[name] = dict(spell=float(spell.mean()), macro=float(scores[:, 1].mean()), per_copy=spell.tolist())
        print("  {}: spellF1 {:.4f}  macroF1 {:.4f} | vs 48 features {:+.4f} ({}/8) | vs 48+19 {:+.4f} ({}/8)".format(
            name, spell.mean(), scores[:, 1].mean(), spell.mean() - base48.mean(), int((spell > base48).sum()),
            spell.mean() - base67.mean(), int((spell > base67).sum())), flush=True)

    # audit of the real model: 5 seeds with TTA
    pairs = [(y, p2.viterbi_tv(p.mean(0), np.zeros(len(y), np.float32), setting[0], setting[1], 0.0))
             for y, p in zip(ys, tta)]
    rows, counts = audit(pairs)
    print("\nspells over the 8 copies, by IoU (matched / predicted / true):")
    for t, (m, q, g) in counts.items():
        print("  IoU {:.2f}: {} / {} / {}   precision {:.3f}  recall {:.3f}".format(t, m, q, g, m / max(q, 1), m / max(g, 1)))
    print("\ntrue spells by length, at IoU {}:".format(AUDIT_IOU))
    for (lo, hi), (ok, team, bound) in rows.items():
        total = max(ok + team + bound, 1)
        print("  {:>2}-{:<4} s: {:4d} spells | recovered {:.1%} | wrong team {:.1%} | boundaries or split {:.1%}".format(
            lo, "inf" if hi > 1000 else int(hi), ok + team + bound, ok / total, team / total, bound / total))
    report["audit"] = dict(by_iou={str(t): v for t, v in counts.items()},
                           by_length={"{}-{}".format(*k): v for k, v in rows.items()})
    with open(os.path.join(pg.OUT, "report4.json"), "w") as handle:
        json.dump(report, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("run", "verify"))
    parser.add_argument("--source", default=pg.DFL_SOURCE)
    parser.add_argument("--copies", type=int, default=16)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--tune-seeds", type=int, default=2)
    args = parser.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        pg.verify()


if __name__ == "__main__":
    main()
