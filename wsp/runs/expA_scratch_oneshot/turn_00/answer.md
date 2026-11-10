## Design

The network is a hand-built evidence model. The Viterbi decoder does the long temporal integration; the network supplies calibrated per-frame logits plus a few latches. None of this has been run, so every threshold and weight is a reasoned guess. It uses no signed x or y and treats the two teams symmetrically, so the four views should agree.

**Layer `a` (GELU used as hinges, units 0–18)**
- 0, 1: control hinge per team, `4·relu(1 − d1/3.5 m − 0.4·relv/10)`, gated by `seen`. Close and moving with the ball counts as control; a pass flying past a player does not.
- 2–5: closeness to touchline and goal line, 0 at 2 m inside rising to 1 at the line. 6: out flags.
- 7, 8: ball moving fast (ramp from 3 to 7 m/s). 9: ball seen and slow.
- 10–13: length of the current ball gap (`since + until`), in two ramps.
- 14: linear pass-through of player activity (mean speed of both teams).
- 15, 16: kick-off formation (both teams entirely in their own halves, nearly all players tracked).
- 17, 18: weak nearest-player evidence inside ball gaps, fading with gap length.

**Conv (9-frame means then ReLU, units 20–27)**
- 20, 21: sustained control advantage for HOME / AWAY.
- 22: sustained "ball out and not flying", which guards against aerial balls projected outside the pitch.
- 23: sustained stationary visible ball.
- 24: sustained fast ball well inside the pitch (the "live motion" cue).
- 25: ball near or beyond a line and not fast.
- 26, 27: a team's player standing at a stationary ball (the restart taker).

**GRU1**
- Forward and backward units 0, 1: slow (20 s) and fast (0.5 s) averages of activity, stored as deviation from a 1.8 m/s prior. Their difference gives a mostly relative "players are slower than usual" cue that tolerates speed-noise differences between tracking sources.
- Forward 2: last-controller latch, set by sustained control and held through gaps and ball flights.
- Forward 3 / backward 3: dead-ball latches. They are set by "ball out" or "stationary ball", reset by live in-field motion, and leak over about 10 s. The forward one says the last ball event was a dead marker; the backward one says the next ball event is.
- Forward 4: restart-taker trace, set while a player stands at a still ball and decaying over about 3 s after the kick. It lets the first possession after a restart register even if it is short.

**GRU2f** is a pass-through (z ≈ 0) that combines and squashes the GRU1 states. **GRU2b** is unused.

**Head**
- Possession: `p = 0.64·(control difference) + gap evidence + 0.06·latch + restart trace`. The logits are `H = −2·GELU(−p)` and `A = −2·GELU(p)`, so `H − A = 2p` exactly while the better of the two stays near 0. That keeps the dead/alive decision independent of how strong the possession evidence is.
- Dead: `D = clip(d, −3, 1.5)`, where `d = −0.6 + near-line + stationary − 1.2·live-motion + gap length + 2·tanh(activity cue) + latches + kick-off formation`.
- Scale: about 2 s of clear control by the other team is needed to pay the 32 for a there-and-back possession change, and a stoppage needs more than about 1.3 s of saturated dead evidence to pay the 48.

```edits
# ================= layer a =================
# 0,1: control hinge x4 (HOME, AWAY): seen-gated, 1 - d1/3.5m - 0.4*relv/10
inp.W[0, 0] = 6
inp.W[0, 13] = -11.43
inp.W[0, 18] = -1.6
inp.b[0] = -2
inp.W[1, 0] = 6
inp.W[1, 30] = -11.43
inp.W[1, 35] = -1.6
inp.b[1] = -2
# 2,3: touchline closeness x4 (clip pair) ; 4,5: goal-line closeness x4
inp.W[2, 12] = -20
inp.b[2] = 4
inp.W[3, 12] = -20
inp.W[4, 11] = -20
inp.b[4] = 4
inp.W[5, 11] = -20
# 6: out flags x4
inp.W[6, 9:11] = [4, 4]
# 7,8: ball moving fast x4 (ramp 3..7 m/s)
inp.W[7, 3] = 10
inp.b[7] = -3
inp.W[8, 3] = 10
inp.b[8] = -7
# 9: ball seen and slow x4 (seen - v/4)
inp.W[9, 0] = 4
inp.W[9, 3] = -10
# 10,11: gap length ramp 0..5 s ; 12,13: ramp 5..15 s
inp.W[10, 5:7] = [8, 8]
inp.W[11, 5:7] = [8, 8]
inp.b[11] = -4
inp.W[12, 5:7] = [4, 4]
inp.b[12] = -2
inp.W[13, 5:7] = [4, 4]
inp.b[13] = -6
# 14: activity pass-through: 6 + 8u, u = 0.25*(msH+msA) - 0.3
inp.W[14, 27] = 2
inp.W[14, 44] = 2
inp.b[14] = 3.6
# 15,16: kick-off formation x4 (clip pair), needs nearly all players tracked
inp.W[15, 20] = 12
inp.W[15, 37] = 12
inp.W[15, 25] = -20
inp.W[15, 42] = -20
inp.b[15] = -20
inp.W[16, 20] = 12
inp.W[16, 37] = 12
inp.W[16, 25] = -20
inp.W[16, 42] = -20
inp.b[16] = -24
# 17,18: weak nearest-player evidence inside ball gaps (unseen only)
inp.W[17, 0] = -8
inp.W[17, 13] = -20
inp.W[17, 5:7] = [-3, -3]
inp.b[17] = 4
inp.W[18, 0] = -8
inp.W[18, 30] = -20
inp.W[18, 5:7] = [-3, -3]
inp.b[18] = 4

# ================= conv (9-frame means + relu) =================
# 20,21: sustained control advantage HOME / AWAY
conv.W[20, 0, :] = 0.027778
conv.W[20, 1, :] = -0.027778
conv.b[20] = -0.1
conv.W[21, 0, :] = -0.027778
conv.W[21, 1, :] = 0.027778
conv.b[21] = -0.1
# 22: sustained out and not flying
conv.W[22, 6, :] = 0.055556
conv.W[22, 7, :] = -0.027778
conv.W[22, 8, :] = 0.027778
conv.b[22] = -1
# 23: sustained stationary visible ball
conv.W[23, 9, :] = 0.041667
conv.b[23] = -0.5
# 24: sustained fast ball well inside the pitch
conv.W[24, 7, :] = 0.034722
conv.W[24, 8, :] = -0.034722
conv.W[24, 2, :] = -0.034722
conv.W[24, 3, :] = 0.034722
conv.W[24, 4, :] = -0.034722
conv.W[24, 5, :] = 0.034722
conv.W[24, 6, :] = -0.034722
conv.b[24] = -0.25
# 25: near/over a line and not fast
conv.W[25, 2, :] = 0.027778
conv.W[25, 3, :] = -0.027778
conv.W[25, 4, :] = 0.027778
conv.W[25, 5, :] = -0.027778
conv.W[25, 7, :] = -0.027778
conv.W[25, 8, :] = 0.027778
# 26,27: team's player at a stationary ball (restart taker) HOME / AWAY
conv.W[26, 0, :] = 0.027778
conv.W[26, 1, :] = -0.027778
conv.W[26, 9, :] = 0.027778
conv.b[26] = -1.0
conv.W[27, 0, :] = -0.027778
conv.W[27, 1, :] = 0.027778
conv.W[27, 9, :] = 0.027778
conv.b[27] = -1.0

# ================= GRU1 forward =================
# 0: slow EMA of tanh(u) (20 s) ; 1: fast EMA (0.5 s)
gru1f.Win[0, 14] = 0.125
gru1f.bin[0] = -0.75
gru1f.biz[0] = 6.21
gru1f.Win[1, 14] = 0.125
gru1f.bin[1] = -0.75
gru1f.biz[1] = 2.4
# 2: last-controller latch
gru1f.Wiz[2, 20:22] = [-24, -24]
gru1f.biz[2] = 9
gru1f.Win[2, 20:22] = [6, -6]
# 3: dead latch (set by out / stationary, reset by live motion, ~10 s leak)
gru1f.Wiz[3, 22:25] = [-20, -8, -20]
gru1f.biz[3] = 5.5
gru1f.Win[3, 22:24] = [3, 0.6]
# 4: restart-taker trace (3 s decay)
gru1f.Wiz[4, 26:28] = [-30, -30]
gru1f.biz[4] = 4.3
gru1f.Win[4, 26:28] = [8, -8]

# ================= GRU1 backward =================
gru1b.Win[0, 14] = 0.125
gru1b.bin[0] = -0.75
gru1b.biz[0] = 6.21
gru1b.Win[1, 14] = 0.125
gru1b.bin[1] = -0.75
gru1b.biz[1] = 2.4
# 3: dead latch looking into the future
gru1b.Wiz[3, 22:25] = [-20, -8, -20]
gru1b.biz[3] = 5.5
gru1b.Win[3, 22:24] = [0.6, 1.2]

# ================= GRU2 forward: pass-through (z ~ 0) =================
gru2f.biz[0:5] = -20
# 0: activity cue tanh(8*(0.85*slow - fast))
gru2f.Win[0, 0] = 3.4
gru2f.Win[0, 128] = 3.4
gru2f.Win[0, 1] = -4
gru2f.Win[0, 129] = -4
# 1: controller latch ; 2: forward dead latch ; 3: backward dead latch ; 4: restart trace
gru2f.Win[1, 2] = 2
gru2f.Win[2, 3] = 1.5
gru2f.Win[3, 131] = 1.5
gru2f.Win[4, 4] = 0.5

# ================= head =================
# m0 = GELU(p), m1 = GELU(-p)
head1.Wh[0, 0:2] = [0.16, -0.16]
head1.Wh[0, 17:19] = [0.06, -0.06]
head1.Wf[0, 1] = 0.06
head1.Wf[0, 4] = 0.5
head1.Wh[1, 0:2] = [-0.16, 0.16]
head1.Wh[1, 17:19] = [-0.06, 0.06]
head1.Wf[1, 1] = -0.06
head1.Wf[1, 4] = -0.5
# m2 = GELU(d + 3), m3 = GELU(d - 1.5)
head1.Wh[2, 10:14] = [0.125, -0.125, 0.1, -0.1]
head1.Wh[2, 15:17] = [0.5, -0.5]
head1.Wh[2, 23:26] = [0.6, -1.2, 1.0]
head1.Wf[2, 0] = 2.0
head1.Wf[2, 2:4] = [1.0, 0.6]
head1.b[2] = 2.4
head1.Wh[3, 10:14] = [0.125, -0.125, 0.1, -0.1]
head1.Wh[3, 15:17] = [0.5, -0.5]
head1.Wh[3, 23:26] = [0.6, -1.2, 1.0]
head1.Wf[3, 0] = 2.0
head1.Wf[3, 2:4] = [1.0, 0.6]
head1.b[3] = -2.1
# logits: H = -2*GELU(-p), A = -2*GELU(p), D = clip(d, -3, 1.5)
head2.W[0, 1] = -2
head2.W[1, 0] = -2
head2.W[2, 2:4] = [1, -1]
head2.b[2] = -3
```