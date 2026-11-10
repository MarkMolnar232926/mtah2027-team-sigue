"""Turn a practice-pass evaluation into the message the writer reads next (experiment B).

Everything in the message comes from the dev split (external matches). It holds the same numbers a
person would look at first: scores per source, per-class precision/recall, the confusion matrix,
possession counts against the truth, the history so far, and three excerpts of frames around
the longest mistakes of three kinds.
"""
import numpy as np

LABELS = "HAD"
# (column title, feature index) shown in an excerpt
COLUMNS = (("seen", 0), ("since", 5), ("until", 6), ("d1_H", 13), ("d1_A", 30), ("n2_H", 15),
           ("n2_A", 32), ("relv_H", 18), ("relv_A", 35), ("slow_H", 28), ("slow_A", 45),
           ("bI_x", 7), ("bI_y", 8), ("out_x", 9), ("out_y", 10))
KINDS = (("wrong team (truth says one team, the network the other)",
          lambda y, p: (y < 2) & (p < 2) & (y != p)),
         ("false dead ball (truth is live play, the network says D)",
          lambda y, p: (y < 2) & (p == 2)),
         ("missed dead ball (truth is D, the network says live play)",
          lambda y, p: (y == 2) & (p < 2)))
STEP, BEFORE, MAX_ROWS = 10, 50, 26


def _runs(mask):
    """(start, end) of every maximal True run."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _excerpt(kept, kind, test, rng):
    """The longest mistake of one kind in a randomly chosen half that has one."""
    candidates = []
    for h in kept:
        runs = _runs(test(h["y"], h["path"]))
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            candidates.append((h, a, b))
    if not candidates:
        return "Excerpt: %s -- none found.\n" % kind
    h, a, b = candidates[rng.integers(len(candidates))]
    start = max(0, a - BEFORE)
    rows = list(range(start, min(len(h["y"]), b + BEFORE), STEP))[:MAX_ROWS]
    out = ["Excerpt: %s. Source %s, a mistake lasting %.1f s; the mistake starts at the row "
           "marked >." % (kind, h["src"], (b - a) / 25.0),
           "   t(s) truth net  P(H)  P(A)  P(D) | " + " ".join("%6s" % c for c, _ in COLUMNS)]
    for t in rows:
        p = h["p"][t]
        out.append("%s%6.1f   %s    %s  %.3f %.3f %.3f | %s" % (
            ">" if a <= t < a + STEP else " ", (t - start) / 25.0, LABELS[h["y"][t]],
            LABELS[h["path"][t]], p[0], p[1], p[2],
            " ".join("%6.2f" % h["X"][t, i] for _, i in COLUMNS)))
    return "\n".join(out) + "\n"


def _table(result):
    out = ["source  halves  spellF1  macroF1  possessions net/truth  mean length net/truth (s)  "
           "frame acc decoded/argmax"]
    names = {"DFL": "optical, degraded", "SC": "broadcast tracking", "MET": "automatic, real gaps",
             "ALL": "all practice halves"}
    for src in ("DFL", "SC", "MET", "ALL"):
        s = result["sources"].get(src)
        if s:
            out.append("%-6s  %6d  %7.4f  %7.4f  %10d / %-8d  %12.1f / %-10.1f  %.3f / %.3f   (%s)" % (
                src, s["halves"], s["spell_f1"], s["macro_f1"], s["n_pred_spells"],
                s["n_gt_spells"], s["pred_mean_spell_s"], s["gt_mean_spell_s"],
                s["frame_acc_decoded"], s["frame_acc_argmax"], names[src]))
    return "\n".join(out)


def _confusion(result):
    s = result["sources"]["ALL"]
    cm = np.array(s["confusion_rows_truth_cols_pred"], float)
    pct = 100 * cm / max(cm.sum(), 1)
    out = ["Confusion after decoding, all practice halves, % of frames (rows truth, columns network):",
           "            net H   net A   net D"]
    for k in range(3):
        out.append("  truth %s  %6.1f  %6.1f  %6.1f" % (LABELS[k], pct[k, 0], pct[k, 1], pct[k, 2]))
    out.append("Per class after decoding:  " + "   ".join(
        "%s precision %.3f recall %.3f" % (c, s["per_class"][c]["precision"], s["per_class"][c]["recall"])
        for c in LABELS))
    return "\n".join(out)


def apply_summary(report):
    out = ["Your edits: %d statements read, %d applied, %d skipped. The network now has %s nonzero "
           "weights." % (report["statements"], report["applied"], len(report["errors"]),
                         format(report["nonzero_weights"], ","))]
    for e in report["errors"][:12]:
        out.append("  skipped (line %d of the block): %s  -> %s" % (e["line"], e["text"][:90], e["reason"]))
    if len(report["errors"]) > 12:
        out.append("  ... and %d more skipped statements" % (len(report["errors"]) - 12))
    return "\n".join(out)


def message(pass_no, n_passes, report, result, kept, history):
    """history: list of (pass number, ALL spellF1, ALL macroF1) including this pass."""
    rng = np.random.default_rng(1000 + pass_no)
    left = n_passes - pass_no
    parts = [
        "PRACTICE PASS %d of %d." % (pass_no, n_passes),
        apply_summary(report),
        "Scores on the practice matches (four views averaged, decoder applied):\n" + _table(result),
        _confusion(result),
        "History (all practice halves):  " + "   ".join(
            "pass %d: spellF1 %.4f macroF1 %.4f" % h for h in history),
        "In the excerpts below there is one row every %d frames (0.4 s). `truth` is the reference "
        "label, `net` your decoded label, P(.) your per-frame probabilities averaged over the four "
        "views, and the remaining columns are input features of the original view, in the units of "
        "THE INPUT (d1_H = x[13], d1_A = x[30], n2 = x[15] / x[32], relv = x[18] / x[35], slow = "
        "x[28] / x[45], bI = x[7], x[8])." % STEP,
    ]
    parts += [_excerpt(kept, kind, test, rng) for kind, test in KINDS]
    if left > 0:
        parts.append("You have %d practice pass%s left, then one final reply. Reply with a short "
                     "note on what you are changing and why, then one ```edits block. It applies on "
                     "top of your current network (`reset` and `restore N` are available)."
                     % (left, "" if left == 1 else "es"))
    else:
        parts.append("That was the last practice pass. This is your final reply: a short note, then "
                     "one ```edits block (or an empty one to keep the network as it is; `restore N` "
                     "returns to an earlier pass). The network as it stands after this reply is the "
                     "one evaluated on the held-out matches.")
    return "\n\n".join(parts) + "\n"
