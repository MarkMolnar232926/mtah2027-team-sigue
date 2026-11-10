"""Figures for wsp/REPORT.md, written to wsp/fig/.

    python wsp/harness/figures.py results     # benchmark spellF1 of every variant, per source
    python wsp/harness/figures.py anatomy     # the hand-written latches at work on a held-out half
    python wsp/harness/figures.py passes      # experiment B: dev score after each practice pass

Colours follow one rule: colour says how the weights were made (no network / written by the LLM /
trained by SGD / trained then edited by the LLM), and in the timeline it says the label (H / A,
with dead ball in neutral grey). Palette slots 1-3 of the reference palette, validated with the
dataviz validator (CVD and normal-vision separation pass); every bar carries its value.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
EVAL = os.path.join(ROOT, "wsp", "eval")
FIG = os.path.join(ROOT, "wsp", "fig")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2aa"
KIND = {"none": (NEUTRAL, "no network (rule baseline)"),
        "written": (BLUE, "weights written by the LLM, no training"),
        "trained": (ORANGE, "weights trained by SGD"),
        "edited": (AQUA, "trained, then edited by the LLM")}
SOURCES = (("DFL", "DFL (optical, degraded)"), ("SC", "SkillCorner (broadcast)"),
           ("MET", "Metrica game 3"))

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load(name, key=None):
    path = os.path.join(EVAL, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        r = json.load(fh)
    return r[key] if key else r


def rows():
    """(label, kind, result) for every variant that has been scored, in a fixed order."""
    out = [("Nearest player + 4 s vote", "none", _load("ref_nearest_benchmark.json")),
           ("A  written in one reply", "written", _load("expA_scratch_oneshot_benchmark.json")),
           ("B  written, 10 practice passes", "written", _load("expB_scratch_passes10_benchmark.json")),
           ("Trained, one model", "trained", _load("trained_single_benchmark.json")),
           ("C  that model, edited", "edited", _load("expC_edit_trained_benchmark.json", "edited")),
           ("Trained, 3-model ensemble", "trained", _load("trained_ensemble3_benchmark.json"))]
    return [r for r in out if r[2] is not None]


def results():
    data = rows()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.52 * len(data) + 1.9), sharey=True)
    y = np.arange(len(data))[::-1]
    for ax, (src, title) in zip(axes, SOURCES):
        vals = [r["sources"][src]["spell_f1"] for _, _, r in data]
        ax.barh(y, vals, height=0.56, color=[KIND[k][0] for _, k, _ in data], zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.015, yi, "%.3f" % v, va="center", ha="left", color=INK, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75"])
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("spellF1")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for lab, _, _ in data], fontsize=10.5, color=INK)
    used = [k for k in KIND if any(kk == k for _, kk, _ in data)]
    fig.legend(handles=[Patch(color=KIND[k][0], label=KIND[k][1]) for k in used],
               loc="lower center", ncol=len(used), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=2.2)
    fig.suptitle("Held-out benchmark: possessions recovered (spellF1, higher is better)",
                 x=0.012, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    _save(fig, "results.png")


def _strip(ax, row, codes, t):
    """One horizontal strip of H / A / D runs."""
    colors = {0: BLUE, 1: ORANGE, 2: NEUTRAL}
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(codes)]))
    for a, b in zip(starts, ends):
        ax.barh(row, (t[b - 1] - t[a]) + (t[1] - t[0]) - 0.06, left=t[a] + 0.03, height=0.62,
                color=colors[int(codes[a])], linewidth=0, zorder=3)


def anatomy(half_id=18, start_s=None, length_s=100):
    """Truth, the hand-written network and the trained one on a stretch of a held-out half, with
    two of the hand-written recurrent units: the last-controller latch and the dead-ball latch."""
    import torch
    import evaluate
    import netspec
    from model import viterbi

    half = evaluate.load_half("benchmark", half_id)
    y = half["y"]
    hand = netspec.from_checkpoint(os.path.join(ROOT, "wsp", "networks", "expA_scratch_oneshot", "hand.pt"))
    trained = netspec.from_checkpoint(os.path.join(ROOT, "models", "expB_bw3met6k_s0.pt"))
    sh, sa = evaluate.decoder()
    paths = {}
    for name, state in (("hand", hand), ("trained", trained)):
        p = evaluate.half_probs([evaluate.net_from_state(state)], half)
        paths[name] = viterbi(np.log(np.clip(p, 1e-6, 1)), sh, sa)

    n = int(length_s * 25)
    if start_s is None:                          # the stretch with the most label changes in truth
        changes = np.concatenate(([0], np.cumsum(np.diff(y) != 0)))
        cand = np.arange(2000, len(y) - n - 2000, 250)
        start = int(cand[np.argmax(changes[cand + n] - changes[cand])])
    else:
        start = int(start_s * 25)
    a0, b0 = max(0, start - 1000), min(len(y), start + n + 1000)
    X = half["views"][0][0]
    net = evaluate.net_from_state(hand)
    gru1 = torch.nn.GRU(128, 128, num_layers=1, batch_first=True, bidirectional=True)
    gru1.load_state_dict({k[4:]: v for k, v in netspec.to_torch_state(hand).items()
                          if k.startswith("gru.") and "_l0" in k})
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(X[a0:b0], np.float32))[None]
        a = net.inp(x)
        h = a + torch.relu(net.conv(a.transpose(1, 2))).transpose(1, 2)
        g1 = gru1.eval()(h)[0][0].numpy()
    cut = slice(start - a0, start - a0 + n)
    t = np.arange(n) / 25.0
    seen = X[start:start + n, 0] > 0.5

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12.5, 6.4), sharex=True,
                                        gridspec_kw={"height_ratios": [1.5, 1.25, 1.25]})
    rows_ = (("Reference labels", y[start:start + n]),
             ("Written by hand (A)", paths["hand"][start:start + n]),
             ("Trained by SGD", paths["trained"][start:start + n]))
    for r, (_, codes) in enumerate(rows_):
        _strip(ax1, len(rows_) - 1 - r, codes, t)
    ax1.set_yticks(range(len(rows_)))
    ax1.set_yticklabels([lab for lab, _ in rows_][::-1], color=INK)
    ax1.set_ylim(-0.6, len(rows_) - 0.4)
    ax1.legend(handles=[Patch(color=BLUE, label="H  home in possession"),
                        Patch(color=ORANGE, label="A  away in possession"),
                        Patch(color=NEUTRAL, label="D  dead ball")],
               loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9.5,
               handlelength=1.2, columnspacing=2.2)

    ax2.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax2.plot(t, g1[cut, 2], color=INK, linewidth=1.8, zorder=3)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["away", "0", "home"])
    ax2.set_title("Hand-written unit gru1f[2]: last-controller latch (set by sustained control, "
                  "held through ball gaps)", loc="left", fontsize=10, color=INK2)

    ax3.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax3.plot(t, g1[cut, 3], color=INK, linewidth=1.8, zorder=3)
    ax3.set_ylim(-0.1, 1.15)
    ax3.set_yticks([0, 1])
    ax3.set_title("Hand-written unit gru1f[3]: dead-ball latch (set by ball out or stationary, "
                  "reset by live motion)", loc="left", fontsize=10, color=INK2)
    ax3.set_xlabel("seconds")
    for ax in (ax2, ax3):                        # light bands where the tracking does not see the ball
        gaps = np.flatnonzero(np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0]))))
        for g0, g1_ in zip(gaps[::2], gaps[1::2]):
            ax.axvspan(t[g0], t[min(g1_, n - 1)], color="#f0efec", zorder=0, linewidth=0)
    ax3.text(1.0, -0.42, "shaded: ball not seen by the tracking", transform=ax3.transAxes,
             ha="right", va="top", fontsize=9, color=MUTED)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_xlim(0, length_s)
    fig.suptitle("A network nobody trained: %d s of a held-out half (Metrica sample game 3)" % length_s,
                 x=0.012, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "anatomy.png")


def passes(run="expB_scratch_passes10"):
    state = json.load(open(os.path.join(ROOT, "wsp", "runs", run, "state.json")))
    series = {"DFL": [], "SC": [], "MET": []}
    for p in range(1, state["passes"] + 1):
        r = json.load(open(os.path.join(ROOT, "wsp", "runs", run, "turn_%02d" % (p - 1), "dev_eval.json")))
        for s in series:
            series[s].append(r["sources"][s]["spell_f1"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(1, state["passes"] + 1)
    for (s, label), color in zip(SOURCES, (BLUE, ORANGE, AQUA)):
        ax.plot(x, series[s], color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.text(x[-1] + 0.15, series[s][-1], "%s  %.3f" % (label.split(" (")[0], series[s][-1]),
                va="center", color=INK, fontsize=10)
    ax.set_xticks(x)
    ax.set_xlim(0.7, state["passes"] + 2.6)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(length=0)
    ax.set_xlabel("practice pass")
    ax.set_ylabel("spellF1 on the practice matches")
    ax.set_title("Experiment B: a hand-written network improving with feedback", loc="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "passes.png")


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    for which in sys.argv[1:] or ["results"]:
        {"results": results, "anatomy": anatomy, "passes": passes}[which]()
