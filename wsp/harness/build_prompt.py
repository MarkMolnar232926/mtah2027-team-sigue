"""Assemble the prompt of a from-scratch run (experiments A and B).

Section order follows the prompts in the paper's log release (exp5_write_from_scratch): task, the
problem with its reference source, THE INPUT, THE OUTPUT, THE NETWORK, scoring, the edits language.
The reference source is solution/features.py, inserted verbatim -- our counterpart of the game
source in the paper's MinAtar prompts. The tensor table is generated from netspec.py so the prompt
cannot drift from the network it describes.

    python wsp/harness/build_prompt.py --passes 0 > prompt.txt
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import netspec  # noqa: E402

TASK = """\
Your task is to write, from scratch, the weights of a neural network that labels ball possession in
football matches from automatic tracking data. There is no training: no gradient descent, no data
to fit, no code to run. You have no tools. You decide what every part of the network should compute
and you write the numbers.

{mode}

The network's architecture is fixed and described below; it has {n_params:,} weights and EVERY
WEIGHT STARTS AT 0. You set the weights you need; the rest stay 0.

Reply in this order: first describe your design briefly (which units compute what, and how the
pieces produce the three outputs), then write the weights as one ```edits block in the language
defined at the end. Only the edits block changes the network."""

MODE_ONESHOT = """\
This is a single attempt. You get no feedback and no second reply: the network your reply defines
is evaluated as it stands."""

MODE_PASSES = """\
You get {passes} practice passes. After each reply, the network you have defined so far is run on a
fixed set of practice matches and the next message reports how it did: scores per data source,
per-class precision and recall, a confusion matrix, how many possessions it produced against the
truth, and a few excerpts of frames where it was wrong. You then reply with more edits. Each
reply's edits apply ON TOP OF your current network, so you can change a few weights or rewrite
everything. After the last practice pass you get one final reply; the network as it stands after
that reply is the one evaluated, on different matches that you never see."""

PROBLEM = """\
THE PROBLEM

A football match is recorded at 25 frames per second; a half is about 70,000 frames. For every
frame the network must say one of three things:

  H   the HOME team is in possession and the ball is in play
  A   the AWAY team is in possession and the ball is in play
  D   dead ball: the ball is not in play (out for a throw-in, corner or goal kick, after a foul,
      after a goal, before kick-off), from the moment play stops until it restarts

Possession rules:
  - While the ball is in play, exactly one team is in possession. There is no "contested" or
    "loose" state: when the ball is loose, in the air or being fought over, possession stays with
    the team that last had it.
  - Possession changes only when the other team gains CONTROLLED possession. A touch, a deflection
    or a header that is not followed by control does not change possession. In the reference
    labels, a spell of the other team shorter than 2 seconds in the middle of live play is not a
    change of possession; it belongs to the team that had the ball before.
  - After a dead ball there is always a clear possessor at the restart; the first possession after
    a restart counts whatever its length.
  - Who "owns" a dead ball is not asked. Dead is D.
  - A possession is a maximal run of H or of A. `H H H A A` is two possessions; `H H D D H H` is
    two possessions with a stoppage between them.
  Typically the ball is in play 55-65% of the time and a half has roughly 100-150 possessions.

What the tracking looks like. It is automatic and uncorrected:
  - The ball is NOT SEEN on about 40% of frames, and it is lost in long stretches (many seconds),
    not in flickers: precisely during duels, crowded phases, balls in the air and many dead balls.
    A ball in the air is seen projected onto the grass, so its position can be metres off.
  - Players have no identities. Each team is an unordered set of tracked positions; tracks appear
    and vanish, a player can be missing or briefly duplicated, and positions are noisy.
  - There is no ball height, no event data, no referee signal.

The network does not see raw positions. It sees {n_feat} numbers per frame, computed by the code
below from the ball track and the two sets of player positions. This file is the reference
implementation of the input; `frame_features(ball, home, away)` is called with kalman=False and
nearest=False. Coordinates are metres with the pitch centre at the origin (pitch 105 x 68, so
|x| <= 52.5 and |y| <= 34 inside the pitch). Before anything else the code flips the pitch so that
the HOME team always attacks towards +x.

```python
{features_source}```"""

INPUT = """\
THE INPUT

x[t] is a vector of {n_feat} numbers for frame t. "Ball" in features 7-12 and in all distances means
the ball position with gaps filled by straight-line interpolation between the last and the next
sighting (constant before the first / after the last sighting).

  idx  name         definition                                                         range
  ---  -----------  -----------------------------------------------------------------  ----------
    0  seen         1 if the tracking saw the ball on this frame, else 0                0 / 1
    1  ball_x       ball x / 52.5 if seen, else 0. +x is where HOME attacks            ~[-1, 1]
    2  ball_y       ball y / 34 if seen, else 0                                        ~[-1, 1]
    3  ball_speed   ball speed (m/s) / 10, from sightings 3 frames either side;        [0, 4]
                    0 if either sighting is missing
    4  ball_acc     ln(1 + ball acceleration in m/s^2) / 5; 0 if not computable        [0, 1.2]
    5  since        seconds since the ball was last seen, capped at 10, / 10.          [0, 1]
                    0 when seen now; 1 if it has not been seen for 10 s or ever
    6  until        seconds until the ball is next seen, capped at 10, / 10            [0, 1]
    7  ballI_x      interpolated ball x / 52.5 (equals feature 1 when seen)            ~[-1, 1]
    8  ballI_y      interpolated ball y / 34                                           ~[-1, 1]
    9  out_x        1 if |interpolated ball x| > 52.5 (beyond a goal line)             0 / 1
   10  out_y        1 if |interpolated ball y| > 34 (beyond a touchline)               0 / 1
   11  to_goalline  (52.5 - |interpolated ball x|) / 10; negative when beyond          <= 5.25
   12  to_touchline (34 - |interpolated ball y|) / 10; negative when beyond            <= 3.4

Then one block of 17 features per team: HOME at 13..29, AWAY at 30..46. With k the offset in the
block (HOME index = 13 + k, AWAY index = 30 + k):

    k  name         definition                                                         range
  ---  -----------  -----------------------------------------------------------------  ----------
    0  d1           distance (m) from the ball to the team's nearest player,           [0, 4]
                    capped at 40, / 10
    1  d2           same for the second-nearest player                                 [0, 4]
    2  n2           number of the team's players within 2 m of the ball (raw count)    0, 1, 2..
    3  n5           number within 5 m, / 3                                             0, 1/3, ..
    4  n10          number within 10 m, / 5                                            0, 0.2, ..
    5  relv         |velocity of the nearest player - velocity of the ball| in m/s,    [0, 3]
                    capped at 30, / 10. It is 3.0 WHENEVER THE BALL IS NOT SEEN on
                    this frame (or a velocity is unavailable). Small = moving together
    6  near_speed   speed (m/s) of the nearest player / 5                              [0, 8]
    7  n_seen       number of the team's tracked players / 11                          ~1
    8  cx           team centroid x / 52.5                                             ~[-1, 1]
    9  cy           team centroid y / 34                                               ~[-1, 1]
   10  sx           standard deviation of the players' x / 20                          >= 0
   11  sy           standard deviation of the players' y / 15                          >= 0
   12  front        the team's most advanced player in its own attacking direction:    ~[-1, 1]
                    max(x * side) / 52.5 with side = +1 for HOME, -1 for AWAY
   13  back         the team's deepest player: min(x * side) / 52.5                    ~[-1, 1]
   14  mean_speed   mean speed of the team's players (m/s) / 3                         >= 0
   15  slow         share of the team's players moving slower than 1 m/s               [0, 1]
   16  top_speed    speed of the team's fastest player (m/s) / 8                       [0, 5]

   47  d1_diff      (HOME d1 - AWAY d1) in the units above, i.e. metres / 10.          [-4, 4]
                    Negative = a HOME player is closer to the ball

Examples: x[t][0] is `seen`; x[t][13] is HOME d1; x[t][30] is AWAY d1; x[t][18] is HOME relv;
x[t][35] is AWAY relv; x[t][28] is HOME slow; x[t][45] is AWAY slow; x[t][47] is d1_diff.
A HOME player standing 1.5 m from a visible ball while the nearest AWAY player is 6 m away gives
x[13] = 0.15, x[30] = 0.6, x[15] = 1, x[32] = 0, x[47] = -0.45."""

OUTPUT = """\
THE OUTPUT

Three logits per frame, in the order (H, A, D). They are used as follows, and you should set the
scale of your logits with this in mind:

  1. log-softmax over the three logits gives per-frame log-probabilities.
  2. The network is run on four physically equivalent views of the same half and the four
     probability vectors are averaged: the original; the pitch mirrored across its long axis
     (every y changes sign); the two teams' roles exchanged (the AWAY team is presented as HOME,
     the pitch is flipped again so that this new home attacks +x, and the H and A outputs are
     exchanged back afterwards); and both at once.
  3. Probabilities are clipped to [1e-6, 1], so a per-frame log-probability lies in [-13.8, 0].
  4. A Viterbi decoder picks the label sequence that maximises the sum over frames of the
     log-probability of the chosen label, minus a penalty for every change of label between
     consecutive frames: 16 for H <-> A, and 24 for any change between alive (H or A) and D.
     So a new state is entered only when the accumulated evidence for it over the frames it lasts
     outweighs the penalties for entering and leaving it. With everything at 0 the decoder outputs
     a constant.

Each half is processed in overlapping windows of 6,000 frames (4 minutes): the recurrent state
starts at 0 at the start of every window and only the central 4,000 frames of a window are kept,
so every kept frame has at least 1,000 frames (40 s) of context on each side, except at the very
start and end of a half."""

NETWORK = """\
THE NETWORK

{intro} Input x[t] has {n_feat} entries; every hidden layer has
128 units; T is the number of frames in a window. All vectors below are per frame t.

  a    = GELU(inp.W @ x + inp.b)                               128   GELU(v) = v * Phi(v)
  c[u] = conv.b[u] + sum over i in 0..127, k in 0..8 of
           conv.W[u, i, k] * a[t + k - 4][i]                    128   a is 0 outside the window
  h    = a + relu(c)                                           128

  Layer-1 GRU, forward direction "gru1f": state s, s = 0 before the first frame of the window,
  frames visited in increasing t. With sigma the logistic function and * elementwise:
      r = sigma(Wir @ h + bir + Whr @ s_prev + bhr)
      z = sigma(Wiz @ h + biz + Whz @ s_prev + bhz)
      n = tanh (Win @ h + bin + r * (Whn @ s_prev + bhn))
      s = (1 - z) * n + z * s_prev                              128
  "gru1b" has its own weights and the same equations, but visits frames in DECREASING t (its state
  at t summarises frames t, t+1, ..., and is 0 after the last frame of the window).
  g1 = concat(s of gru1f at t, s of gru1b at t)                 256

  Layer-2 GRU: "gru2f" (increasing t) and "gru2b" (decreasing t), same equations, input g1 instead
  of h. Their Wir, Wiz, Win are 128 x 256: columns 0..127 read gru1f, columns 128..255 read gru1b.
  qf = state of gru2f at t, qb = state of gru2b at t           128 each

  m      = GELU(head1.Wf @ qf + head1.Wb @ qb + head1.Wh @ h + head1.b)     128
  logits = head2.W @ m + head2.b                                 3   order (H, A, D)

Conventions. Every matrix is indexed [unit, input]: row u holds the weights INTO unit u of the
layer being computed, column i reads input i. conv.W[u, i, k] is the weight into unit u from unit i
of `a` at time offset k - 4 (k = 0 is 4 frames in the past, k = 4 is the current frame, k = 8 is
4 frames in the future). Note that z is the gate that KEEPS the old state: z near 1 holds s, z near
0 replaces it by n. GELU(v) is about v for large positive v, about 0 for large negative v, and
GELU(0) = 0. With every weight at 0 all activations and all logits are 0.

The tensors you can set, with their shapes:

{tensor_table}

{closing}"""

INTRO_SCRATCH = """\
The architecture is that of a network we trained with gradient descent for this task; yours has
the same shape and is run by the same code."""
CLOSING_SCRATCH = "{n_params:,} weights in total. EVERY WEIGHT STARTS AT 0."

INTRO_EDIT = """\
This is the architecture of the network you are given. Its {n_params:,} weights were trained with
gradient descent (AdamW, 6,000 steps of 16 random 2-minute crops, cross-entropy over H / A / D with
frames within 1 s of a label change weighted 3x, dropout 0.2 during training) on about 2.8 million
labelled frames from the three tracking sources described under SCORING."""
CLOSING_EDIT = """\
{n_params:,} weights in total, all of them trained. They do not fit in a message; you read and edit
them through the Python environment described below."""

TASK_EDIT = """\
Below is a neural network that was trained with gradient descent to label ball possession in
football matches from automatic tracking data. It works, and it has known weaknesses. Your task is
to make it better BY EDITING ITS WEIGHTS DIRECTLY.

There is no training. You may not run gradient descent, regression, or any other procedure that
fits or searches weights against the labels (no optimisation loops, no least squares, no grid or
random search over weight values). You work the way one debugs a program: read the weights, measure
what parts of the network compute, form a hypothesis about what causes a failure, and change the
weights that implement it. Statistics that describe the network (means of activations over groups
of frames, directions, projections, decompositions of weight matrices) are measurements and are
fine; the edit itself must follow from your understanding of the mechanism, and you must be able to
say what it does and why.

You have a Python environment that holds the network, and a budget of {budget} FORWARD PASSES.
Reading and editing weights is free and unlimited. Data is reachable only by running the network,
and every run costs one pass. The network as it stands when you finish is evaluated on different,
held-out matches that you never see, against the original."""

WEAKNESSES = """\
WHAT IS KNOWN ABOUT THE TRAINED NETWORK

On held-out matches the trained network, with the decoder, reaches a spellF1 of about 0.72 on two
of the three tracking sources and about 0.84 on the third. From our error analysis:

  - Under-splitting. Short possessions after quick turnovers are merged into their neighbours: the
    network produces fewer possessions than the truth has, and possessions shorter than 5 s are
    the ones most often missed. Weighting frames near label changes in the loss reduced this but
    did not remove it.
  - Team assignment is where most of the remaining error is. If the network were given perfect
    dead-ball timing its spellF1 would rise by only about 0.05; if it were given the right team on
    every live frame it would rise by about 0.2. Contested phases and phases with the ball unseen
    are the hard ones.
  - Restarts. Some dead-ball frames just before a restart are labelled as live play.
  - The three sources label slightly differently (one counts short stoppages as live more often),
    and the network has learned a blend.

These are observations, not instructions. You may find something else that matters more."""

TOOLS = """\
THE PYTHON ENVIRONMENT

In each reply you may write ONE fenced block opened with ```python. It is run in a persistent
Python process (variables you define stay available in later blocks) and whatever it prints comes
back as the next message. The process already contains:

{api}

Rules of the environment:
  - Only `numpy`, `math`, `collections`, `itertools`, `functools` and `statistics` can be imported.
    There is no file, network or process access, and names or attributes that start with an
    underscore are refused. A refused block is not run at all.
  - Printed output is cut to about {cap:,} characters, so print summaries, not arrays.
  - A block may run for at most {timeout} minutes. If it runs longer the process is restarted: your
    weight edits up to the previous block are kept, your variables are lost.
  - forward() returns large arrays (about 10 KB per frame in total); keep what you need.
  - The practice halves come from matches of the three sources, labelled the same way as the
    held-out matches. The network saw these matches during training, so its errors on them are
    somewhat fewer than on held-out matches. The DFL halves come in pairs (0 and 1, 2 and 3): the
    same half of play with the same labels, degraded twice with different random noise and gaps.
  - You have {budget} passes in total and at most {max_turns} replies. Plan them: a score() at the
    end to check that an edit helps is usually worth more than one more forward()."""

PROTOCOL_EDIT = """\
HOW TO REPLY

Each reply: a short note on what you are doing and why, then one ```python block. You will receive
its output and can continue.

When you are finished, reply WITHOUT a python block. That final reply must state (1) what you
found, (2) exactly which weights you changed and how, (3) why that change should help on matches
you have not seen, and (4) what you measured. The working network at that moment is the result. If
you conclude that no edit helps, restore the original weights (W[name] = BASE[name] for the tensors
you touched) and say so: an honest "no improvement found" is a valid result.

Begin."""

SCORING = """\
SCORING

The network is evaluated on held-out professional matches from three different tracking sources:
broadcast-grade optical tracking that has been degraded to look automatic (ball hidden on 25-55% of
frames in long gaps, position noise, fragmented player tracks); automatic tracking from broadcast
video (10 Hz, resampled to 25); and automatic tracking with real ball gaps on 30-50% of frames.
For each half the decoded label sequence is compared with reference labels.

  spellF1 (the main score). Every possession (maximal run of H or of A) is one object. Predicted
  possessions are matched one-to-one to true ones by temporal intersection-over-union, and the F1
  of the matching is averaged over IoU thresholds 0.3, 0.4, 0.5, 0.6, 0.7. A constant output
  scores 0. Splitting one true possession into two, merging two into one, or starting one a second
  late all cost; a possession found one second late everywhere scores about 0.65.

  macroF1 (reported beside it). The unweighted mean of the per-frame F1 of H, A and D.

Your network is compared with the same architecture trained by gradient descent on about 2.8
million labelled frames, and with a simple baseline that gives the ball to the team of the nearest
player and smooths over 4 seconds."""

EDITS = """\
HOW TO WRITE THE WEIGHTS

End your reply with one fenced block that opens with ```edits and closes with ```. One statement
per line:

    inp.W[3, 47] = -2.5                  # one weight: into unit 3 of `a`, from input 47
    inp.b[3] = 0.5                       # one bias
    inp.W[4, 13:16] = [1.0, 0.5, -2]     # a list fills a range, left to right
    conv.W[10, 3, :] = 0.111             # a single number fills the whole range
    conv.W[11, 3, 0:9:2] = 0.2           # ranges are a:b (b excluded), a:b:s, or : for a whole axis
    head2.W[0:2, 5:7] = [[1, -1],        # a 2-D list fills a block; a long list may continue
                         [-1, 1]]        #   over the following lines
    gru1f.bhz[0:8] += 2                  # in place: +=  -=  *=
{extra_statements}
Rules:
  - The names are exactly those of the tensor table above. Indices are plain integers (negative
    integers count from the end), no arithmetic and no variables.
  - A value is a decimal number or a list of them; assignment follows numpy rules, so a list must
    match the range it fills and a single number fills any range.
  - Every weight must satisfy |w| <= 1e6.
  - Statements apply in order; a later one overwrites an earlier one. `#` starts a comment.
  - Only the LAST edits block in your reply is read.
  - A statement that cannot be read is skipped and the others still apply.

Now describe your design, then write the edits block."""

EXTRA_PASSES = """\
    reset                                # set every weight back to 0
    restore 3                            # go back to your network as it stood after pass 3
"""


def tensor_table():
    shapes = netspec.alias_shapes()
    rows = [
        ("inp.W", "weights into `a` from the %d inputs" % netspec.N_FEAT),
        ("inp.b", "bias of `a`"),
        ("conv.W", "[unit, unit of a, time offset k]"),
        ("conv.b", "bias of c"),
    ]
    for g, src in (("gru1f", "h"), ("gru1b", "h"), ("gru2f", "g1"), ("gru2b", "g1")):
        rows += [("%s.Wir" % g, "reset gate, from %s" % src),
                 ("%s.Wiz" % g, "update gate, from %s" % src),
                 ("%s.Win" % g, "candidate, from %s" % src),
                 ("%s.Whr" % g, "reset gate, from the previous state"),
                 ("%s.Whz" % g, "update gate, from the previous state"),
                 ("%s.Whn" % g, "candidate, from the previous state"),
                 ("%s.bir  .biz  .bin  .bhr  .bhz  .bhn" % g,
                  "the six biases (%s.bir, %s.biz, ...), 128 each" % (g, g))]
    rows += [("head1.Wf", "weights into m from qf (gru2f)"),
             ("head1.Wb", "weights into m from qb (gru2b)"),
             ("head1.Wh", "weights into m from h"),
             ("head1.b", "bias of m"),
             ("head2.W", "weights into the logits (row 0 = H, 1 = A, 2 = D) from m"),
             ("head2.b", "bias of the logits")]
    lines = []
    for names, what in rows:
        shape = " x ".join(str(d) for d in shapes[names.split()[0]])
        lines.append("  %-41s %s" % (names, what) if " " in names
                     else "  %-10s %-14s %s" % (names, shape, what))
    return "\n".join(lines)


def build(passes=0):
    with open(os.path.join(ROOT, "solution", "features.py"), encoding="utf-8") as fh:
        source = fh.read()
    mode = MODE_PASSES.format(passes=passes) if passes else MODE_ONESHOT
    parts = [
        TASK.format(mode=mode, n_params=netspec.N_PARAMS),
        PROBLEM.format(n_feat=netspec.N_FEAT, features_source=source),
        INPUT.format(n_feat=netspec.N_FEAT),
        OUTPUT,
        NETWORK.format(n_feat=netspec.N_FEAT, tensor_table=tensor_table(), intro=INTRO_SCRATCH,
                       closing=CLOSING_SCRATCH.format(n_params=netspec.N_PARAMS)),
        SCORING,
        EDITS.format(extra_statements=EXTRA_PASSES if passes else ""),
    ]
    return "\n\n\n".join(parts) + "\n"


def build_edit(budget, max_turns, timeout_min, output_cap, api_doc):
    """The first message of experiment C: improve a trained network by editing its weights."""
    with open(os.path.join(ROOT, "solution", "features.py"), encoding="utf-8") as fh:
        source = fh.read()
    scoring = SCORING.split("Your network is compared")[0].rstrip() + (
        "\n\nYour edited network is compared with the original, unedited network on the same "
        "held-out matches, with the same decoder.")
    parts = [
        TASK_EDIT.format(budget=budget),
        PROBLEM.format(n_feat=netspec.N_FEAT, features_source=source),
        INPUT.format(n_feat=netspec.N_FEAT),
        OUTPUT,
        NETWORK.format(n_feat=netspec.N_FEAT, tensor_table=tensor_table(),
                       intro=INTRO_EDIT.format(n_params=netspec.N_PARAMS),
                       closing=CLOSING_EDIT.format(n_params=netspec.N_PARAMS)),
        scoring,
        WEAKNESSES,
        TOOLS.format(api=api_doc, cap=output_cap, timeout=timeout_min, budget=budget,
                     max_turns=max_turns),
        PROTOCOL_EDIT,
    ]
    return "\n\n\n".join(parts) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--passes", type=int, default=0)
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    sys.stdout.write(build(ap.parse_args().passes))
