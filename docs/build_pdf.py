"""Build docs/Sigue.pdf (the technical document) from the numbers in docs/TECHNICAL.md.

    python docs/build_pdf.py        # needs reportlab + matplotlib (not part of the solution env)
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "Sigue.pdf")
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- fonts & palette
FONTS = "C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("Body", os.path.join(FONTS, "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", os.path.join(FONTS, "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", os.path.join(FONTS, "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Mono", os.path.join(FONTS, "consola.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                              boldItalic="Body-Bold")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
RULE = "#d9d8d4"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # validated categorical slots 1-3 (all-pairs safe)
ACCENT = SERIES[0]

S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=20, leading=24,
                            textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Body", fontSize=10.5, leading=14, textColor=INK2,
                          spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=13, leading=16, textColor=INK,
                         spaceBefore=12, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=10.5, leading=13, textColor=INK,
                         spaceBefore=7, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.4, leading=13.2, textColor=INK,
                           alignment=TA_LEFT, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=9.4, leading=13.2,
                             textColor=INK, leftIndent=11, bulletIndent=1, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.4, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.4, leading=10.6,
                            textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Body-Italic", fontSize=8.2, leading=10.8,
                          textColor=INK2, spaceBefore=2, spaceAfter=8),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.2, leading=10.6, textColor=INK,
                           backColor=colors.HexColor("#f3f2ef"), borderPadding=6, spaceBefore=6, spaceAfter=10),
    "kpi": ParagraphStyle("kpi", fontName="Body-Bold", fontSize=17, leading=20, textColor=INK),
    "kpil": ParagraphStyle("kpil", fontName="Body", fontSize=8, leading=10, textColor=INK2),
}


def P(text, st="body"):
    return Paragraph(text, S[st])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def code(t):
    return f'<font face="Mono" size="8.6">{t}</font>'


def table(rows, widths, header=True, bold_last=False):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) or (bold_last and i == len(rows) - 1)
                       else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK)),
          ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor(RULE)),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    t.setStyle(TableStyle(st))
    return t


# ---------------------------------------------------------------- figures
def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(-4.5, 40)
    ax.axis("off")
    plt.rcParams["font.family"] = "Arial"

    def box(x, y, w, h, title, sub, fill="#ffffff", edge=INK2, bold_edge=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc=fill, ec=ACCENT if bold_edge else edge,
                                    lw=1.4 if bold_edge else 0.8))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=7.6,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.6, sub, ha="center", va="top", fontsize=6.2, color=INK2,
                linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                     lw=0.9, color=INK2))

    ax.text(0.5, 39, "INFERENCE  (provided game)", fontsize=6.6, color=MUTED, fontweight="bold",
            va="top")
    yI, hI = 20.5, 15
    box(0.5, yI, 15, hI, "Tactical ATD", "ball + 46 track\nslots, no IDs,\n25 fps")
    box(20, yI, 17, hI, "48 features", "ball, nearest players,\nteam shape; pitch\nflipped: home → +x")
    box(41.5, yI, 19, hI, "BiGRU × 3 seeds", "× 4 views (mirror,\nteam swap), averaged\nin probability",
        bold_edge=True)
    box(65, yI, 15.5, hI, "Viterbi", "penalty 16 per H↔A,\n24 per live↔dead")
    box(84.5, yI, 15, hI, "Sigue.csv", "H / A / D per frame,\nX outside halves")
    for a, b in ((16.0, 19.5), (37.5, 41.0), (61.0, 64.5), (81.0, 84.0)):
        arrow(a, yI + hI / 2, b, yI + hI / 2)

    ax.text(0.5, 17.6, "TRAINING  (external data only)", fontsize=6.6, color=MUTED,
            fontweight="bold", va="top")
    yT, hT = 0.5, 14
    box(0.5, yT, 24, hT, "DFL · 7 matches", "clean optical tracking,\ndegraded ×8 to look\nautomatic",
        fill="#f5f8fd")
    box(27.5, yT, 20, hT, "SkillCorner · 20", "real automatic\nbroadcast tracking ×4", fill="#f5f8fd")
    box(50.5, yT, 19, hT, "Metrica · 3", "real tracking ×8,\nlabels from events", fill="#f5f8fd")
    box(73, yT, 26.5, hT, "Labels + loss", "2 s controlled-possession\nrule; boundary frames\nweighted ×3")
    # the three sources join into the labels / loss box, which trains the network
    for cx in (12.5, 37.5, 60.0):
        ax.plot([cx, cx], [yT - 0.6, -3.2], color=INK2, lw=0.9)
    ax.plot([12.5, 86.25], [-3.2, -3.2], color=INK2, lw=0.9)
    arrow(86.25, -3.2, 86.25, yT - 0.6)
    arrow(80, yT + hT + 0.6, 58, yI - 0.6)
    ax.text(71.5, 18.6, "trains", fontsize=6.2, color=INK2, style="italic")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_results(path):
    stages = ["DFL + SC,\n2 models", "+ 4-view\naveraging", "+ boundary loss,\n3 models",
              "+ Metrica\n(final)"]
    data = {"DFL (degraded)": [0.724, 0.728, 0.752, 0.741],
            "SkillCorner": [0.725, 0.730, 0.746, 0.747],
            "Metrica game 3": [None, None, 0.795, 0.855]}
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(7.0, 2.7), dpi=220)
    w = 0.25
    for k, (name, vals) in enumerate(data.items()):
        xs = [i + (k - 1) * (w + 0.02) for i in range(len(stages))]
        for x, v in zip(xs, vals):
            if v is None:
                ax.text(x, 0.655, "n/a", ha="center", va="bottom", fontsize=6, color=MUTED)
                continue
            ax.bar(x, v - 0.6, w, bottom=0.6, color=SERIES[k], edgecolor="white", linewidth=1.2,
                   label=name if x == xs[-1] or vals.index(v) == 2 else None)
            if x == xs[-1]:
                ax.text(x, v + 0.004, "%.3f" % v, ha="center", va="bottom", fontsize=6.6,
                        color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES]
    ax.legend(handles, list(data), frameon=False, fontsize=7, ncol=3, loc="upper left",
              bbox_to_anchor=(0, 1.13))
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=7, color=INK2)
    ax.set_ylim(0.6, 0.88)
    ax.set_ylabel("spellF1 (held out)", fontsize=7, color=INK2)
    ax.tick_params(axis="y", labelsize=6.6, colors=INK2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color="#ecebe8", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- document
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(18 * mm, 10 * mm, "Team Sigue · More than a Hack 2027 · Phase 1 technical document")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def build():
    fp, fr = os.path.join(FIG, "pipeline.png"), os.path.join(FIG, "results.png")
    fig_pipeline(fp)
    fig_results(fr)
    W = A4[0] - 36 * mm
    st = []

    st += [P("Frame-level ball possession from automatic tracking", "title"),
           P("<b>Team Sigue</b> · More than a Hack 2027 (Metrica Sports, FC Barcelona, MWCapital) · "
             "Phase 1 technical document", "sub")]

    kpis = [("0.855", "spellF1, held-out Metrica match"), ("0.747", "spellF1, held-out SkillCorner"),
            ("0.741", "spellF1, held-out DFL"), ("0", "labels from the provided game")]
    kt = Table([[P(v, "kpi") for v, _ in kpis], [P(l, "kpil") for _, l in kpis]],
               colWidths=[W / 4] * 4, hAlign="LEFT")
    kt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.HexColor(ACCENT)),
                            ("TOPPADDING", (0, 0), (-1, 0), 6), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 1), (-1, 1), 4)]))
    st += [kt, Spacer(1, 8)]

    st += [P("We label every frame <b>H</b>, <b>A</b> or <b>D</b> from the <b>automatic tactical "
             "tracking alone</b>. A bidirectional recurrent network reads identity-free features "
             "of the ball and the players, its output is averaged over three seeds and four "
             "physically equivalent views of the pitch, and a Viterbi decoder that charges a cost "
             "for every change of state turns it into possessions. The network is trained <b>only "
             "on external data</b> — 7 open DFL matches, 20 open SkillCorner matches and Metrica's 3 "
             "public sample matches. No label, statistic or parameter comes from the provided "
             "game's ground truth or video.")]

    # 1. Approach
    st += [P("1. Approach", "h1"),
           P("<b>Why a learned sequence model.</b> Possession is an interval, not a property of a "
             "frame. The ball is unseen on about 41% of graded frames, in long stretches (median "
             "gap ≈ 11 s), precisely during duels and aerial balls. Deciding a frame needs what "
             "happened seconds before <i>and after</i>; a bidirectional GRU sees both directions."),
           P("<b>Why a decoder.</b> spellF1 matches whole possessions, so one flicker splits a "
             "possession in two. Viterbi decoding over {H, A, D} with a penalty per H↔A switch (16) "
             "and per live↔dead switch (24, log-probability units) keeps only the changes the "
             "evidence strongly supports: ≈ +0.08 spellF1 over per-frame argmax on held-out data."),
           KeepTogether([Image(fp, width=W, height=W * 3.2 / 7.2 * 0.93),
                         P("Figure 1. Inference (top) uses only the tactical ATD; the network is "
                           "trained on three external sources (bottom).", "cap")])]
    st += [P("Pipeline", "h2")]
    st += bullets([
        f"<b>Read the ATD</b> ({code('atd_parse.py')}): ball x, y and every track slot's x, y and "
        "team, per video frame (frame_id − 1), in metres around the pitch centre.",
        f"<b>48 identity-free features</b> ({code('features.py')}), with the pitch flipped so the "
        "home team attacks +x: ball visibility, position, speed, acceleration, time since / until "
        "the last / next sighting, position interpolated across gaps, distance to the lines; per "
        "team the nearest and second-nearest player, players within 2 / 5 / 10 m, the relative "
        "velocity of the nearest player and the ball (a player in control moves with it), "
        "centroid, spread, deepest / most advanced player, mean speed and the share of "
        "near-stationary players (a strong dead-ball cue).",
        f"<b>PossessionNet</b> ({code('model.py')}): linear embedding → residual 1-D convolution → "
        "2-layer bidirectional GRU (128 units) → logits over H, A, D, run over each half in "
        "overlapping 4-minute windows.",
        "<b>Ensemble, views, decoding</b>: 3 seeds × 4 views (original, mirrored, home/away "
        "swapped with outputs swapped back, both) averaged in probability, then Viterbi. Frames "
        "outside the two halves are written X."])

    # 2. Assets
    st += [P("2. Assets used and dropped", "h1"),
           table([["asset", "used?", "how / why not"],
                  ["tactical ATD", "<b>yes</b>", "the only input to the model"],
                  ["tactical homography", "no", "the ATD is already in pitch coordinates"],
                  ["Smart Tagging", "no", "exists only for the provided game, so any rule using "
                   "it could not be validated without the ground truth"],
                  ["tactical / broadcast video", "no", "not used as input, not used to check "
                   "outputs"],
                  ["broadcast ATD", "no", "not synchronised with the reference timeline; the "
                   "tactical ATD covers the whole match"],
                  ["public 5-minute ground truth", "check only", "pipeline check, see §6"]],
                 [W * 0.27, W * 0.13, W * 0.60])]

    # 3. Training data
    st += [P("3. Training data", "h1"),
           P("All training and every tuning decision use external data. <b>Nothing from the "
             "provided game</b> — no labels, no ground truth, no video — is used for training, "
             "validation, tuning or model selection. No data was labelled by our team."),
           table([["dataset", "licence", "content", "labels used", "labelled by"],
                  ["DFL open data (Bassek et al., 2025, <i>Scientific Data</i>, "
                   "doi:10.6084/m9.figshare.28196177): 7 matches", "CC BY 4.0",
                   "25 Hz optical tracking, players and ball", "per-frame BallStatus and "
                   "BallPossession", "DFL / Sportec Solutions"],
                  ["SkillCorner Open Data: 20 A-League 2024/25 matches", "MIT",
                   "10 Hz automatic broadcast tracking with detection flags",
                   "phases_of_play (in play + team)", "SkillCorner's automatic pipeline"],
                  ["Metrica Sports sample data (GitHub: metrica-sports / sample-data): 3 matches",
                   "published for community use, source acknowledged",
                   "25 Hz tracking with real ball gaps", "event data → possession (§4)",
                   "Metrica Sports"]],
                 [W * 0.27, W * 0.14, W * 0.21, W * 0.20, W * 0.18]),
           Spacer(1, 6),
           P("≈ 965 k labelled frames from DFL, ≈ 1.45 M from SkillCorner (resampled to 25 Hz) and "
             "≈ 430 k from Metrica."),
           P("<b>Making DFL look like automatic tracking</b> (" + code("degrade.py") + "). Each "
             "half is copied 8 times, each copy with its own noise level (domain randomisation): "
             "the ball is removed in long gaps (25–55% of frames) placed preferentially on hard "
             "frames — ball in the air, two opponents close, dead ball, far from the camera; "
             "airborne balls are projected onto the grass from a simulated high camera; ball and "
             "players get correlated position noise; player tracks are fragmented (2–15% missing) "
             "with occasional ghost duplicates. The ranges come from published facts and from "
             "unlabelled statistics of the ATD (gap lengths, tracked-player counts)."),
           P("<b>SkillCorner and Metrica</b> are real tracking: only lightly jittered (4 and 8 "
             "copies per half), with SkillCorner's extrapolated players randomly hidden. "
             "<b>Augmentation</b> for all: home/away swap and mirroring across the long axis."),
           P("<b>Training</b>: AdamW, one-cycle schedule (peak 2·10<super>−3</super>), 6 000 steps "
             "of 16 random 2-minute crops, cross-entropy over H/A/D. Frames within ±1 s of a label "
             "change count <b>3×</b> in the loss (boundary weighting), which fixed much of the "
             "main failure we saw on held-out data: short possessions merged into their "
             "neighbours. ≈ 12 min per model on an RTX 4060.")]

    # 4. Rules
    st += [P("4. Possession rules implemented", "h1"),
           P("The challenge defines a change of possession as needing <b>controlled</b> "
             "possession; touches, deflections and aerial duels without control do not count, and "
             "a loose ball stays with the last team in possession. The rules are encoded in the "
             "training labels:")]
    st += bullets([
        "<b>Control (2 s rule).</b> Inside a stretch of live play, a team spell shorter than 2 s "
        "that does not start the stretch is not a change of possession and goes to the team that "
        "had the ball before. DFL's per-touch possession (≈ 220 spells per half, median 3.7 s) "
        "becomes ≈ 144 possessions per half (mean 11 s), in line with SkillCorner's independent "
        "phases (≈ 100–140). The 2 s value implements the definition's own proxy (\"keeps the ball "
        "for a few seconds\") and was not fitted to the ground truth.",
        "<b>Loose balls and challenges.</b> Possession stays with the last team until the other "
        "team controls the ball; in Metrica's events, duels and cards never change possession. "
        "At inference the decoder's switch cost requires the evidence for the other team to "
        "persist.",
        "<b>Restarts.</b> Every restart has a clear possessor: the first spell after a dead ball "
        "is never relabelled.",
        "<b>Dead ball.</b> D whenever the source marks the ball dead (DFL BallStatus = 0; outside "
        "every SkillCorner phase; in Metrica from ball-out, foul, goal, shot out of play or "
        "offside until the next restart). Dead-ball ownership is not modelled.",
        "<b>Abstention.</b> Never used on graded frames."])

    # 5. Validation
    st += [P("5. Validation and results", "h1"),
           P("Every recipe change was decided on a fixed <b>external benchmark</b> that no "
             "compared model was trained on: 2 DFL matches (degraded with fixed seeds), 5 "
             "SkillCorner matches and Metrica sample game 3 — the last one in the same EPTS format "
             "as the provided game. Decoder penalties come from leave-one-match-out cross-"
             "validation on DFL; on the benchmark every setting between 16 and 24 lies within "
             "0.0015 spellF1, a broad plateau."),
           KeepTogether([Image(fr, width=W, height=W * 2.7 / 7.0 * 0.95),
                         P("Figure 2. Held-out spellF1 per source as the recipe grew. Metrica "
                           "game 3 entered the benchmark with the third step.", "cap")]),
           P("<b>Adding Metrica</b> trades −0.011 on DFL for +0.06 on the held-out Metrica match, "
             "where the model now finds 283 possessions against 285 true (247 before). We weighted "
             "the Metrica source most because it is the closest to the test (same tracking provider "
             "and file format) and DFL the least (broadcast-grade tracking that we degrade "
             "ourselves). The final models train on all three Metrica matches."),
           P("<b>Tried and not kept</b>:"),
           table([["variant", "DFL", "SkillCorner", "why dropped"],
                  ["larger network (192 units, 2× steps)", "0.733", "0.741", "below the smaller one"],
                  ["boundary weight 5 instead of 3", "0.748", "0.745", "same as 3"],
                  ["5 instead of 3 models", "0.754", "0.739", "noise"],
                  ["2× SkillCorner copies", "0.718", "0.744", "worse on DFL"],
                  ["Kalman-smoothed ball features", "0.744", "0.749", "no gain (velocity error "
                   "2.3 → 1.7 m/s, but not possession)"],
                  ["4 nearest players per team as features", "0.732", "0.755", "mixed"],
                  ["Metrica data only", "0.364", "0.494", "0.860 on Metrica, collapses elsewhere"]],
                 [W * 0.36, W * 0.1, W * 0.13, W * 0.41])]

    # 6. Public GT
    st += [P("6. Use of the public ground truth", "h1"),
           P("Used as the rules intend: to check that the pipeline is correct (frame alignment, "
             "label meaning, file format) — never to train, tune or choose between versions. Every "
             "version was decided before being scored there. For the record, the final file scores "
             "spellF1 0.662 / macroF1 0.890 on the 5 public minutes (earlier versions 0.676 / "
             "0.916 and 0.627 / 0.926). The window holds 17 possessions and the scorer's own "
             "bootstrap puts its noise band at ≈ 0.41, so these numbers mean little beyond \"the "
             "pipeline works\"; choosing a version by them would be tuning on the ground truth.")]

    # 7. Limitations
    st += [P("7. Limitations and what we would do next", "h1")]
    st += bullets([
        "<b>Team assignment is the main remaining error.</b> On the benchmark, perfect dead-ball "
        "timing would add only ≈ 0.05 spellF1; the right team on every live frame would add ≈ 0.2. "
        "Contested phases and very short possessions (< 5 s) are the typical misses.",
        "<b>Part of the Metrica gain is label style.</b> A Metrica-only model matches the final "
        "model on the held-out Metrica match but collapses elsewhere, so the +0.06 partly reflects "
        "Metrica's conventions. The provided game's tracking (46 identity-free, fragmented slots) "
        "differs from the clean sample tracking; mixing all sources keeps the model robust to that.",
        "<b>Label definitions differ between sources</b> (relabelled DFL, SkillCorner phases, "
        "Metrica events); the model learns a blend. Metrica's events count short stoppages as live "
        "slightly more often.",
        "<b>Restarts</b>: dead ball is inferred from player movement and ball position only; some "
        "frames near restarts are labelled live.",
        "<b>No video</b>: a ball detector on the tactical video could fill the long ball gaps. "
        "<b>Smart Tagging</b> could sharpen restarts but could not be validated without the "
        "ground truth.",
        "<b>More open matches</b> in the target format would be the most valuable addition."])

    # 8. Reproduction
    st += [P("8. Reproduction", "h1"),
           P("The repository regenerates the submitted file byte for byte, on GPU or CPU, from "
             "the tactical ATD and the published weights in " + code("models/") + "."),
           Preformatted("python -m venv .venv\n"
                        ".venv/Scripts/pip install torch==2.6.0 --index-url "
                        "https://download.pytorch.org/whl/cu124\n"
                        ".venv/Scripts/pip install -r solution/requirements-solution.txt\n"
                        "# put the match data in data/ (see DATA.md)\n"
                        ".venv/Scripts/python solution/reproduce.py   # -> submission/Sigue.csv, "
                        "validated", S["code"]),
           P("To retrain from scratch, download the three external datasets (links in "
             + code("solution/README.md") + ") and run " + code("python solution/train_all.py")
             + " (≈ 1.5 h on an RTX 4060)."),
           Spacer(1, 6),
           P("<i>Acknowledgements: tracking data courtesy of the DFL (Bassek et al., 2025), "
             "SkillCorner and Metrica Sports.</i>", "cap")]

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=17 * mm, title="Team Sigue — technical "
                            "document", author="Team Sigue",
                            subject="More than a Hack 2027, Phase 1")
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    sub = os.path.join(ROOT, "submission")
    os.makedirs(sub, exist_ok=True)
    shutil.copy(OUT, os.path.join(sub, "Sigue.pdf"))
    print("wrote", OUT, "and submission/Sigue.pdf")


if __name__ == "__main__":
    build()
