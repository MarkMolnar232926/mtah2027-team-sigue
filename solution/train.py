"""Train on degraded DFL halves, validate on held-out DFL matches, tune the decoder there.

    python solution/train.py --val 0 1 --steps 4000          # one CV fold
    python solution/train.py --val none --steps 4000 --out models/final.pt   # final model

Everything in here is external data. The competition match is never read.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from model import PossessionNet, predict, viterbi  # noqa: E402

PROC = os.path.join(ROOT, "external", "processed")
FEATS = "train_feats.npz"
SWITCH_HA = (2, 4, 6, 8, 12, 16, 24, 32)
SWITCH_ALIVE = (2, 4, 6, 8, 12, 16, 24)


def load():
    z = np.load(os.path.join(PROC, FEATS))
    X, Y, lens, G = z["X"], z["Y"], z["lens"], z["G"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    seqs = [(X[offs[i]:offs[i + 1]], Y[offs[i]:offs[i + 1]], G[i]) for i in range(len(lens))]
    return seqs


def score(gt, pred):
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"]


def tune_decoder(val_preds):
    """Grid over the two penalties, maximising mean spellF1 over the held-out halves."""
    best = None
    table = {}
    for sh in SWITCH_HA:
        for sa in SWITCH_ALIVE:
            s = [score(y, viterbi(lp, sh, sa)) for lp, y in val_preds]
            sf, mf = np.mean([a for a, _ in s]), np.mean([b for _, b in s])
            table[(sh, sa)] = (sf, mf)
            if best is None or sf > best[0]:
                best = (sf, mf, sh, sa)
    return best, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="+", default=["0"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--crop", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--feats", default="train_feats.npz")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--boundary-weight", type=float, default=1.0,
                    help="loss weight for frames within --boundary-radius of a label change")
    ap.add_argument("--boundary-radius", type=int, default=25)
    args = ap.parse_args()
    global FEATS
    FEATS = args.feats
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    seqs = load()
    val_ids = set() if args.val == ["none"] else {int(v) for v in args.val}
    train = [s for s in seqs if s[2][0] not in val_ids]
    val = [s for s in seqs if s[2][0] in val_ids]
    print("train halves", len(train), "val halves", len(val), "device", dev, flush=True)

    n_feat = seqs[0][0].shape[1]
    model = PossessionNet(n_feat, hidden=args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps)
    lossf = torch.nn.CrossEntropyLoss(reduction="none")
    from scipy.ndimage import maximum_filter1d
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        xb, yb = [], []
        for _ in range(args.batch):
            x, y, _ = train[rng.integers(len(train))]
            a = rng.integers(0, len(x) - args.crop)
            xb.append(np.asarray(x[a:a + args.crop], np.float32))
            yb.append(y[a:a + args.crop].astype(np.int64))
        xb = torch.from_numpy(np.stack(xb)).to(dev)
        yb = torch.from_numpy(np.stack(yb)).to(dev)
        per = lossf(model(xb).reshape(-1, 3), yb.reshape(-1))
        if args.boundary_weight != 1.0:
            yn = yb.cpu().numpy()
            change = np.zeros_like(yn, dtype=bool)
            change[:, 1:] = yn[:, 1:] != yn[:, :-1]
            near = maximum_filter1d(change.astype(np.uint8), 2 * args.boundary_radius + 1, axis=1)
            w = torch.from_numpy(np.where(near > 0, args.boundary_weight, 1.0)
                                 .astype(np.float32)).to(dev).reshape(-1)
            loss = (per * w).sum() / w.sum()
        else:
            loss = per.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            print("step %d loss %.4f  %.0fs" % (step, loss.item(), time.time() - t0), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
        torch.save({"state": model.state_dict(), "n_feat": n_feat, "hidden": args.hidden},
                   os.path.join(ROOT, args.out))

    if val:
        # each held-out half, only its degraded copies 0..3 to keep tuning fast
        val_preds = [(predict(model, x, dev), y) for x, y, g in val if g[2] < 4]
        val_src = ["SC" if g[0] >= 100 else "DFL" for x, y, g in val if g[2] < 4]
        frame_acc = np.mean([(lp.argmax(1) == y).mean() for lp, y in val_preds])
        raw = np.mean([score(y, lp.argmax(1).astype(np.int8))[0] for lp, y in val_preds])
        if args.no_tune:
            from json import load as _l
            d = _l(open(os.path.join(ROOT, "models", "decoder.json")))
            s = [score(y, viterbi(lp, d["switch_ha"], d["switch_alive"])) for lp, y in val_preds]
            for src in ("DFL", "SC"):
                ss = [v for v, k in zip(s, val_src) if k == src]
                if ss:
                    print("VAL %-3s spellF1 %.4f macroF1 %.4f (%d halves)" % (
                        src, np.mean([a for a, _ in ss]), np.mean([b for _, b in ss]), len(ss)))
            print("VAL frame acc %.4f | decoded spellF1 %.4f macroF1 %.4f (fixed decoder)" % (
                frame_acc, np.mean([a for a, _ in s]), np.mean([b for _, b in s])))
            return
        best, table = tune_decoder(val_preds)
        print("VAL frame acc %.4f | argmax spellF1 %.4f | decoded spellF1 %.4f macroF1 %.4f "
              "@ switch_ha=%s switch_alive=%s" % (frame_acc, raw, best[0], best[1], best[2], best[3]))
        tag = "_".join(sorted(args.val))
        with open(os.path.join(PROC, "cv_%s.json" % tag), "w") as fh:
            json.dump({"%s,%s" % k: v for k, v in table.items()}, fh)
        np.save(os.path.join(PROC, "cv_preds_%s.npy" % tag),
                np.array([lp for lp, _ in val_preds], dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
