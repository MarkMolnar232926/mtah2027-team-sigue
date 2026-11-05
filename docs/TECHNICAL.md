# Team Sigue — Frame-level ball possession from automatic tracking

*More than a Hack 2027, Phase 1. Technical document.*

## 1. Summary

We label every frame `H`, `A` or `D` from the **automatic tactical tracking (ATD) alone**, with a
bidirectional recurrent network, averaged over four symmetric views of the pitch, followed by a
Viterbi decoder that charges a cost for every change of state. The network is trained **only on
external data** — 7 open DFL Bundesliga matches, 20 open SkillCorner A-League matches and Metrica
Sports' 3 public sample matches — and the DFL tracking is deliberately *degraded* to look like
uncorrected automatic tracking. No label, statistic or parameter comes from the provided game's
ground truth or video.

Fixed external benchmark — 2 held-out DFL matches (degraded with fixed seeds), 5 held-out
SkillCorner matches (real automatic tracking) and Metrica sample game 3 (Metrica tracking in the
same EPTS format as the provided game); no model below was trained on them:

| recipe | DFL | SkillCorner | Metrica g3 |
|---|---|---|---|
| DFL + SkillCorner training, 2-model ensemble | 0.724 | 0.725 | – |
| + four-view averaging | 0.728 | 0.730 | – |
| + boundary-weighted loss, 3-model ensemble | 0.752 | 0.746 | 0.795 |
| **+ Metrica games 1–2 in training, 6 000 steps (final recipe)** | **0.741** | **0.747** | **0.855** |

(spellF1.) Adding Metrica trades −0.011 on DFL for +0.06 on Metrica: on the held-out Metrica match
the model finds 283 possessions against 285 true (247 before), fixing most of the under-splitting.
We weighted the Metrica source most because it is the closest to the test data (same tracking
provider and file format); DFL is the furthest (broadcast-grade optical tracking that we degrade
artificially). The final models are trained on all three Metrica matches.

Also tried and **not** kept: a larger network (0.733 / 0.741 on DFL / SkillCorner), a stronger
boundary weight (0.748 / 0.745), 5 instead of 3 models (0.754 / 0.739), twice as many SkillCorner
copies (0.718 / 0.744), Kalman-smoothed ball features (0.744 / 0.749), per-player features of the 4
nearest players of each team (0.732 / 0.755), other decoder penalties (flat within ±0.0015 between
16 and 24).

Earlier evaluations: DFL 7-fold leave-one-match-out 0.698; DFL-only model on all 20 SkillCorner
matches 0.648.

## 2. Approach

**Why a learned sequence model.** Possession is an interval, not a frame property: the ball is
unseen on ~41% of graded frames and goes missing in long stretches (median gap ≈ 11 s), precisely
during duels and aerial balls. Deciding a frame needs what happened seconds before *and after*. A
bidirectional GRU sees both directions; a per-frame classifier does not.

**Why a decoder.** The ranking metric matches whole possessions, so one spurious flicker splits a
possession in two. Viterbi decoding over {H, A, D} with a penalty per H↔A switch (16) and per
alive↔dead switch (24, in log-probability units) keeps only state changes the evidence strongly
supports. On held-out DFL this adds ≈ +0.08 spellF1 over per-frame argmax.

### Pipeline

1. **Read the ATD** (`solution/atd_parse.py`): ball x, y and every track slot's x, y and team, per
   video frame (`frame_id − 1`), converted to metres with the pitch centre as origin.
2. **Identity-free features** (`solution/features.py`, 48 per frame). The pitch is flipped so the
   home team attacks +x (direction estimated from the teams' mean positions in each half).
   - ball: visible flag, position, speed, acceleration, time since last / until next sighting,
     position interpolated across gaps, out-of-bounds flags, distance to touch- and goal-lines;
   - per team: distance of the nearest and second-nearest player to the ball, number of players
     within 2 / 5 / 10 m, relative velocity between the nearest player and the ball (a player
     controlling the ball moves with it), nearest player's speed, number of tracked players,
     centroid, spread, most advanced / deepest player, mean speed, share of near-stationary
     players (a strong dead-ball cue), top speed;
   - nearest-home minus nearest-away distance.
3. **PossessionNet** (`solution/model.py`): linear embedding → residual 1-D convolution → 2-layer
   bidirectional GRU (128 units) → per-frame logits over H, A, D. Inference runs over each half in
   overlapping 4-minute windows stitched at their centres.
4. **Ensemble, views and decoding**: three networks (seeds 0, 1, 2), each run on four views of the
   same half — original, mirrored across the long axis, home/away roles swapped (H and A outputs
   swapped back), and both — averaged in probability space, then Viterbi. The four views are
   physically equivalent, so averaging removes view-specific noise. Frames outside the two halves
   are written `X` (never graded).

## 3. Assets used and dropped

| asset | used? | how / why not |
|---|---|---|
| tactical ATD | **yes** | the only input to the model |
| tactical homography | no | ATD is already on pitch coordinates |
| Smart Tagging | no | it exists only for the provided game, so any rule using it could not be validated without the ground truth. Kept out deliberately |
| tactical / broadcast video | no | not used as input and not used to check outputs |
| broadcast ATD | no | unsynchronised with the reference timeline; tactical ATD covers the full match |
| public 5-minute ground truth | pipeline check only | see §6 |

## 4. Training data

All training and every tuning decision use external data. **Nothing from the provided game** —
no labels, no ground truth, no video — is used for training, validation, tuning or model
selection. No data was labelled by our team.

| dataset | licence | content | labels used | labelled by |
|---|---|---|---|---|
| DFL open dataset (Bassek et al., 2025, *Scientific Data*, [doi:10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177)): 7 Bundesliga / 2. Bundesliga matches | CC BY 4.0 | 25 Hz optical tracking of players and ball | per-frame `BallStatus` (alive/dead) and `BallPossession` (team) | the data provider (DFL / Sportec Solutions) |
| SkillCorner Open Data: 20 A-League 2024/25 matches | MIT (credit: SkillCorner) | 10 Hz automatic broadcast tracking, with `is_detected` flags | `phases_of_play` intervals (ball in play + team in possession) | SkillCorner's automatic pipeline |
| Metrica Sports sample data ([github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)): 3 anonymised matches | published by Metrica Sports for community use, source acknowledged (no formal licence file) | 25 Hz tracking with real ball gaps (≈ 30–50% of frames) | event data (passes, carries, recoveries, set pieces, ball out, fouls…) turned into possession, see §5 | Metrica Sports |

≈ 965 k labelled frames from DFL, ≈ 1.45 M (after resampling to 25 Hz) from SkillCorner and
≈ 430 k from Metrica.

**Making DFL look like automatic tracking** (`solution/degrade.py`). Each match half is copied 8
times; every copy draws its own noise level from a broad range (domain randomisation), so the model
learns robustness to a family of trackers rather than to one setting:
- ball lost in **long gaps** (25–55% of frames), placed preferentially on hard frames — ball in the
  air, two opponents close to the ball, dead ball, far from the camera — then closed/opened so
  short gaps and short sightings disappear, as an interpolating tracker would produce;
- airborne balls **projected onto the grass** from a simulated high side camera;
- temporally correlated position noise on ball and players, rare spurious ball detections;
- player tracks **fragmented** (2–15% missing in stretches), occasional ghost duplicates;
- no identities are used at any point.

The *ranges* were set from published facts (DATA.md) and from unlabelled statistics of the ATD
(ball-gap lengths, tracked-player counts) — no ground truth, no video.

**SkillCorner** is already real automatic tracking: only ball detections flagged `is_detected` are
kept, a random share of extrapolated players is hidden, and positions are lightly jittered. It is
resampled 10 → 25 Hz (positions interpolated, labels nearest-sample).

**Metrica** (`solution/metrica_parse.py`) is real tracking as well and is used like SkillCorner
(8 lightly jittered copies per half). Its possession labels come from Metrica's *event* data:
the ball is live from a set piece (or goal-kick pass) until ball-out, a foul, a goal, a shot out of
play, an offside or the end of the half; while live, possession belongs to the team of the last
controlled action (pass, carry, shot, recovery, set piece, ball lost); duels and cards do not change
it. The same 2 s control rule as for DFL is then applied. This gives 115–152 possessions per half.

**Augmentation**: home/away swap (labels H↔A swapped) and mirroring across the long axis.

**Training**: AdamW, one-cycle learning rate (peak 2·10⁻³), 6 000 steps of 16 random 2-minute
crops, cross-entropy over H/A/D, ~12 min per model on an RTX 4060.

**Boundary-weighted loss.** Error analysis on held-out external matches showed the main failure
was *under-splitting*: short possessions after quick turnovers merged into their neighbours. Frames
within ±1 s (25 frames) of a label change therefore count 3× in the loss, which teaches the network
to place changes of possession sharply. This was the largest single gain (+0.02 DFL, +0.014
SkillCorner on the benchmark); weight 5 gave the same result as 3.

**Validation and tuning**: decoder penalties chosen by grid search on held-out matches — 7-fold
leave-one-match-out on DFL, re-checked on the external benchmark for the final recipe. The optimum is
a broad plateau (every setting between 16 and 24 is within 0.0015 spellF1), so the choice is not
fragile. Every recipe change above was decided on the external benchmark only.

## 5. Possession rules implemented

The provided definition (CHALLENGE.md) says a change of possession needs **controlled**
possession; touches, deflections and aerial duels without control do not count, and a loose ball
stays with the last team in possession.

- **Training labels encode this rule.** DFL's per-frame possession switches on every touch
  (≈ 220 team spells per half, median 3.7 s). We relabel it: inside a stretch of live play, a team
  spell shorter than **2 s** that does not start the stretch is not a change of possession and goes
  to the team that had the ball before. 2 s implements the definition's own proxy ("a player keeps
  the ball for a few seconds"); it gives ≈ 144 possessions per half with a mean of 11 s, in line
  with the football-analytics literature and with SkillCorner's independent phase labels
  (≈ 100–140 per half). It was not fitted to the ground truth.
- **Restarts**: every restart has a clear possessor; the first spell after a dead ball is never
  relabelled.
- **Dead ball**: `D` whenever the source marks the ball dead (DFL `BallStatus = 0`; outside every
  SkillCorner phase; in Metrica from ball-out, foul, goal, shot out of play or offside until the
  next restart). Dead-ball ownership is not modelled, as the challenge asks.
- **Loose / contested balls**: handled by the decoder's switch cost — possession changes only when
  the evidence for the other team persists.
- **Abstention**: never used on graded frames.

## 6. Use of the public ground truth

Used exactly as the rules intend: to check that the pipeline is correct (frame alignment, label
meaning, file format), never to train, tune or choose between versions. Every version was decided
before being scored there. For the record, the final file scores spellF1 0.662 / macroF1 0.890 on
the 5 public minutes (earlier versions: 0.676 / 0.916 and 0.627 / 0.926). The window holds only 17 possessions and
the scorer's own bootstrap puts its noise band at ≈ 0.41, so these numbers say little beyond "the
pipeline works" — and choosing the earlier version because it scores higher there would be tuning
on the ground truth. The final version was chosen on the external benchmark (§1) alone.

## 7. Limitations and what we would do next

- **Where the remaining error is.** On the benchmark, giving the model perfect dead-ball timing
  would raise spellF1 by only ≈ 0.05; giving it the right team on every live frame would raise it by
  ≈ 0.2. Team assignment in contested phases, and very short possessions (< 5 s), are the main
  misses.
- **Part of the Metrica gain is label style.** A model trained on the Metrica sample matches alone
  scores as well on the held-out Metrica match (0.860) as the final model, but collapses elsewhere
  (DFL 0.36, SkillCorner 0.49). The +0.06 on Metrica therefore partly reflects Metrica's labelling
  conventions; we expect a smaller gain on the provided game, whose tracking (46 identity-free,
  fragmented track slots) differs from the clean sample tracking. Mixing all sources keeps the model
  robust to that difference.
- **Kalman smoothing** of the ball (`solution/kalman.py`, noise parameters set on degraded vs clean
  DFL tracking, no labels) gave cleaner ball velocities (median error 2.3 → 1.7 m/s) but no
  possession gain on the benchmark, so it is not used in the final model.
- **Restarts**: some dead-ball frames close to a restart are labelled live; the dead-ball signal is
  inferred from player movement and ball position only.
- **Label definitions differ between sources** (DFL relabelled vs SkillCorner phases vs Metrica
  events vs the organisers' corrected-tracking ownership); the model learns a blend of them.
  Metrica's event labels count short stoppages as live slightly more often than the others.
- **SkillCorner labels are at 10 Hz**, so their transition timing is accurate to ±2 frames.
- **No video**: a ball detector run on the tactical video could fill the long ball gaps; we did not
  use the video at all.
- **Smart Tagging** could sharpen restarts, but could not be validated without the ground truth.

## 8. Reproduction

```bash
python -m venv .venv
.venv/Scripts/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # or the CPU wheel
.venv/Scripts/pip install -r solution/requirements-solution.txt
# put the match data in data/ (see DATA.md)
.venv/Scripts/python solution/reproduce.py          # -> submission/Sigue.csv, validated
```

`reproduce.py` reads only the tactical ATD and the published weights in `models/` and regenerates
the submitted file byte for byte. To retrain from scratch, download the external data (see
`solution/README.md`) and run `python solution/train_all.py`.
