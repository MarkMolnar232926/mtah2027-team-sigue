"""Score a network on external matches -- the same pipeline as solution/eval_fold.py, with details.

Two splits, both external data only (the provided match is never read here):

    benchmark   DFL 5-6 (degraded, fixed seeds), SkillCorner 15-19, Metrica game 3.
                The fixed held-out benchmark of docs/TECHNICAL.md. Final numbers only; never shown
                to a writer.
    dev         DFL 0 (degraded, fixed seeds), SkillCorner 0-2, Metrica game 1.
                What a writer gets feedback on in the experiments that allow feedback.

Features of the four symmetric views are cached per half under external/processed/wsp_cache/
(gitignored), so scoring a new network only costs the forward passes and the decoder.

    python wsp/harness/evaluate.py --split benchmark --models models/expB_bw3met6k_s0.pt
    python wsp/harness/evaluate.py --split benchmark --reference nearest
    python wsp/harness/evaluate.py --split benchmark --reference zero
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "solution"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import eval_fold  # noqa: E402
from eval_external import load_models  # noqa: E402
from features import frame_features  # noqa: E402
from model import PossessionNet, predict, viterbi  # noqa: E402
import netspec  # noqa: E402

torch.set_num_threads(int(os.environ.get("WSP_THREADS", "4")))   # more threads do not help a GRU
CACHE = os.path.join(ROOT, "external", "processed", "wsp_cache")
SPLITS = {
    "benchmark": dict(dfl=eval_fold.DFL_VAL, sc=eval_fold.SC_VAL, met=eval_fold.MET_VAL),
    "dev": dict(dfl=(0,), sc=(0, 1, 2), met=(0,)),
}
SOURCES = ("DFL", "SC", "MET")
LABELS = ("H", "A", "D")
FLIP = np.array([1, -1], np.float32)


def decoder():
    with open(os.path.join(ROOT, "models", "decoder.json")) as fh:
        d = json.load(fh)
    return d["switch_ha"], d["switch_alive"]


def _views(b, hp, ap_):
    """The four physically equivalent views of eval_fold.probs: (features, H/A outputs swapped)."""
    out = []
    for bb, hh, aa, swapped in ((b, hp, ap_, False), (b * FLIP, hp * FLIP, ap_ * FLIP, False),
                                (b, ap_, hp, True), (b * FLIP, ap_ * FLIP, hp * FLIP, True)):
        out.append((frame_features(bb, hh, aa), swapped))
    return out


def build_cache(split):
    folder = os.path.join(CACHE, split)
    os.makedirs(folder, exist_ok=True)
    index = []
    for i, (src, b, hp, ap_, y) in enumerate(eval_fold.benchmark_halves(**SPLITS[split])):
        views = _views(b, hp, ap_)
        np.savez(os.path.join(folder, "%02d.npz" % i), y=y.astype(np.int8),
                 swapped=np.array([s for _, s in views]),
                 **{"X%d" % v: X for v, (X, _) in enumerate(views)})
        index.append({"src": src, "frames": int(len(y))})
        print("cached %s half %d (%s, %d frames)" % (split, i, src, len(y)), flush=True)
    with open(os.path.join(folder, "index.json"), "w") as fh:
        json.dump(index, fh)
    return index


def split_index(split):
    path = os.path.join(CACHE, split, "index.json")
    if not os.path.exists(path):
        return build_cache(split)
    with open(path) as fh:
        return json.load(fh)


def load_half(split, i):
    """dict(src, y, views=[(X, swapped), ...]) for half i of a split."""
    index = split_index(split)
    z = np.load(os.path.join(CACHE, split, "%02d.npz" % i))
    return {"src": index[i]["src"], "y": z["y"],
            "views": [(z["X%d" % v], bool(s)) for v, s in enumerate(z["swapped"])]}


def net_from_state(state):
    net = PossessionNet(netspec.N_FEAT, hidden=netspec.HIDDEN)
    net.load_state_dict(netspec.to_torch_state(state), strict=True)
    net.kalman = net.nearest = False
    return net.eval()


def predict_batched(net, Xs, chunk=6000, overlap=1000):
    """model.predict for several equally long sequences at once, windows batched.

    model.predict runs the windows of a half one after another; on a CPU the recurrent layers then
    cost about 1 ms per frame, i.e. hours for the benchmark. The windows are independent, so here
    all windows of the same length (from all the given views) go through the network as one batch.
    Same windows, same stitching, same numbers up to float round-off (tested against predict)."""
    T = len(Xs[0])
    step = chunk - 2 * overlap
    windows, start = [], 0
    while start < T:
        windows.append((start, max(0, start - overlap), min(T, start + step + overlap)))
        start += step
    out = [np.zeros((T, 3), np.float32) for _ in Xs]
    by_len = {}
    for w in windows:
        by_len.setdefault(w[2] - w[1], []).append(w)
    net.eval()
    with torch.no_grad():
        for group in by_len.values():
            x = np.stack([np.asarray(X[a:b], np.float32) for X in Xs for _, a, b in group])
            lp = torch.log_softmax(net(torch.from_numpy(x)), -1).numpy()
            for v in range(len(Xs)):
                for g, (s, a, b) in enumerate(group):
                    n = min(step, T - s)
                    out[v][s:s + n] = lp[v * len(group) + g, s - a:s - a + n]
    return out


def half_probs(nets, half, tta=True):
    """Ensemble and views averaged in probability space, exactly like eval_fold.probs."""
    views = half["views"][:4 if tta else 1]
    out = []
    for net in nets:
        for lp, (_, swapped) in zip(predict_batched(net, [X for X, _ in views]), views):
            p = np.exp(lp)
            out.append(p[:, [1, 0, 2]] if swapped else p)
    return np.mean(out, 0)


def confusion3(y, pred):
    cm = np.zeros((3, 3), np.int64)
    ok = (y >= 0) & (y < 3) & (pred >= 0) & (pred < 3)
    np.add.at(cm, (y[ok], pred[ok]), 1)
    return cm


def _prf(cm):
    out = {}
    for k, name in enumerate(LABELS):
        tp = cm[k, k]
        p = tp / max(cm[:, k].sum(), 1)
        r = tp / max(cm[k].sum(), 1)
        out[name] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                     "f1": round(float(2 * p * r / max(p + r, 1e-9)), 4)}
    return out


def score_paths(split, labeller, keep=False):
    """Run `labeller(half) -> (path, extra)` over a split and collect every number we report."""
    index = split_index(split)
    halves, kept = [], []
    for i in range(len(index)):
        half = load_half(split, i)
        path, extra = labeller(half)
        y = half["y"]
        r = pm.evaluate(y, path)
        halves.append({
            "half": i, "src": half["src"], "frames": int(len(y)),
            "spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"]),
            "n_gt_spells": int(r["n_gt_spells"]), "n_pred_spells": int(r["n_pred_spells"]),
            "gt_mean_spell_s": float(r["gt_mean_spell_duration_s"] or 0),
            "pred_mean_spell_s": float(r["mean_spell_duration_s"] or 0),
            "confusion": confusion3(y, path).tolist(),
            "argmax_acc": extra.get("argmax_acc"),
        })
        if keep:
            kept.append(dict(extra, path=path, y=y, X=half["views"][0][0], src=half["src"], half=i))
        print("%-9s half %2d %-3s spellF1 %.3f macroF1 %.3f  possessions %d (true %d)" % (
            split, i, half["src"], halves[-1]["spell_f1"], halves[-1]["macro_f1"],
            halves[-1]["n_pred_spells"], halves[-1]["n_gt_spells"]), flush=True)

    sources = {}
    for src in SOURCES + ("ALL",):
        hs = [h for h in halves if src == "ALL" or h["src"] == src]
        if not hs:
            continue
        cm = np.sum([h["confusion"] for h in hs], 0)
        acc = [h["argmax_acc"] for h in hs if h["argmax_acc"] is not None]
        sources[src] = {
            "halves": len(hs),
            "spell_f1": round(float(np.mean([h["spell_f1"] for h in hs])), 4),
            "macro_f1": round(float(np.mean([h["macro_f1"] for h in hs])), 4),
            "n_gt_spells": int(sum(h["n_gt_spells"] for h in hs)),
            "n_pred_spells": int(sum(h["n_pred_spells"] for h in hs)),
            "gt_mean_spell_s": round(float(np.mean([h["gt_mean_spell_s"] for h in hs])), 2),
            "pred_mean_spell_s": round(float(np.mean([h["pred_mean_spell_s"] for h in hs])), 2),
            "frame_acc_decoded": round(float(np.trace(cm) / max(cm.sum(), 1)), 4),
            "frame_acc_argmax": round(float(np.mean(acc)), 4) if acc else None,
            "confusion_rows_truth_cols_pred": cm.tolist(),
            "per_class": _prf(cm),
        }
    result = {"split": split, "sources": sources, "halves": halves}
    return (result, kept) if keep else result


def score_nets(nets, split, tta=True, keep=False):
    sh, sa = decoder()

    def labeller(half):
        with torch.no_grad():
            p = half_probs(nets, half, tta)
        path = viterbi(np.log(np.clip(p, 1e-6, 1)), sh, sa)
        extra = {"argmax_acc": float((p.argmax(1) == half["y"]).mean())}
        if keep:
            extra["p"] = p.astype(np.float32)
        return path, extra

    out = score_paths(split, labeller, keep)
    result = out[0] if keep else out
    result.update(tta=bool(tta), n_models=len(nets), decoder={"switch_ha": sh, "switch_alive": sa})
    return out


def score_state(state, split, tta=True, keep=False):
    return score_nets([net_from_state(state)], split, tta, keep)


def nearest_baseline(split, window=50):
    """starter/baseline.py on the external splits: nearest player to a visible ball, then a centred
    majority vote over +/- window frames; D where no ball was seen in the window."""
    def labeller(half):
        X = half["views"][0][0]
        seen = X[:, 0] > 0.5
        home = seen & (X[:, 47] <= 0)            # feature 47 = (nearest home - nearest away) / 10
        away = seen & (X[:, 47] > 0)
        ch = np.concatenate(([0], np.cumsum(home)))
        ca = np.concatenate(([0], np.cumsum(away)))
        t = np.arange(len(X))
        lo, hi = np.maximum(0, t - window), np.minimum(len(X), t + window + 1)
        nh, na = ch[hi] - ch[lo], ca[hi] - ca[lo]
        path = np.where(nh + na == 0, 2, np.where(nh >= na, 0, 1)).astype(np.int8)
        return path, {"argmax_acc": None}

    result = score_paths(split, labeller)
    result.update(reference="nearest player + majority vote", window=window)
    return result


def brief(result):
    return "  ".join("%s %.4f / %.4f" % (s, result["sources"][s]["spell_f1"],
                                         result["sources"][s]["macro_f1"])
                     for s in SOURCES if s in result["sources"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="benchmark", choices=sorted(SPLITS))
    ap.add_argument("--models", nargs="+", help="checkpoints, relative to the repository root")
    ap.add_argument("--reference", choices=("nearest", "zero"))
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out", help="write the full result as JSON")
    args = ap.parse_args()
    if args.reference == "nearest":
        result = nearest_baseline(args.split)
    elif args.reference == "zero":
        result = score_state(netspec.zero_state(), args.split, not args.no_tta)
        result["reference"] = "all-zero network"
    else:
        result = score_nets(load_models(args.models, "cpu"), args.split, not args.no_tta)
        result["models"] = args.models
    print("spellF1 / macroF1   " + brief(result))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="\n") as fh:
            json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
