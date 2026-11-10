# Journal — weight-space programming on the possession model

Chronological log of what was done, why, and what came out. Numbers are spellF1 unless stated.
Dates are 2026.

## Nov 10 — setup

**Idea.** *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab;
[post](https://rikiyat.github.io/blog/weight-space-programming/),
[logs](https://github.com/RikiyaT/lm_understand-logs)) reports that a frontier LLM can write or edit
the weights of a neural network directly as text, guided by a description of the task, with no
gradient descent. We try it on our task: per-frame ball possession (H / A / D) from automatic
tracking.

**Decisions.**

1. *Same architecture as the trained model.* The network to write is `PossessionNet(48, 128)` from
   `solution/model.py`, 698,115 weights, all starting at 0. A hand-written checkpoint has the same
   format as `models/*.pt`, so it runs through `eval_fold.py` and `predict.py` unchanged, and the
   comparison "SGD on 2.8M labelled frames vs written by hand" is like for like: same features,
   same decoder, same four-view averaging. This mirrors the paper's Exp. 5, where the hand-written
   MinAtar policies have the shape of the PPO baselines.
2. *The writer is a fresh, tool-free model call.* Like the paper's `codex exec` with every tool
   disabled, each turn is `claude -p --tools "" --strict-mcp-config --safe-mode` run from an empty
   folder with a one-line system prompt. The session reports `tools: []` and `mcp_servers: []`;
   `run.json` records that as `audit.tool_free`. Model `claude-fable-5-2`.
3. *Three experiments*, following the paper's: **A** one reply, no feedback (Exp. 5); **B** the
   same with 10 practice passes of feedback (the idea of Exp. 6); **C** edit a trained network
   through a Python kernel with a budget of forward passes (Exp. 4 / 8).
4. *Two splits, external data only.* `benchmark` is the fixed held-out benchmark of
   `docs/TECHNICAL.md` (DFL 5–6, SkillCorner 15–19, Metrica game 3); writers never see it.
   `dev` (DFL 0, SkillCorner 0–2, Metrica game 1) is what feedback comes from in B and C. The
   provided match and its public ground truth appear in no prompt and no feedback.
5. *No empirical statistics in the from-scratch prompt*, except two pieces of general football
   knowledge (ball in play 55–65% of the time, roughly 100–150 possessions per half). Feature
   ranges are the analytic ones from `features.py`.

**Environment.** This clone had no data. External data and the match were linked (hardlinks and a
junction, nothing copied) from the team's other working copy with `wsp/setup_local_data.ps1`;
parsed with the existing `solution/*_parse.py`. Python 3.13 venv, `torch 2.6.0+cpu` and the pins of
`solution/requirements-solution.txt`. CPU only (i7-1260P).

**A snag: inference speed.** `model.predict` runs a half as ~17 windows one after another. On this
CPU the recurrent layers cost about 1 ms per frame, which would make one benchmark run of one model
take over two hours. The windows are independent, so `evaluate.predict_batched` sends all windows of
equal length, from all four views, through the network as one batch: same windows, same stitching,
log-probabilities equal to `predict` within 2·10⁻⁵ (unit-tested), about 12× faster. A benchmark run
of one model now takes ~12 minutes. `eval_fold.py` and `predict.py` are untouched and still work on
any checkpoint produced here; they are just slow without a GPU.

**Harness** (`wsp/harness/`): `netspec.py` (network + tensor aliases), `edits.py` (the edits
language), `llm.py` (one tool-free turn, logged), `build_prompt.py`, `evaluate.py`, `feedback.py`,
`guard.py` + `kernel.py` (analysis kernel for C), `run_scratch.py`, `run_edit.py`. 17 unit tests in
`wsp/tests/`. The only change to existing code is that `eval_fold.benchmark_halves()` takes the
match indices as parameters, defaulting to the benchmark.

## Nov 10 — reference rows on the benchmark

Scored with `wsp/harness/evaluate.py` (four views averaged, decoder penalties from
`models/decoder.json`), spellF1 / macroF1:

| | DFL | SkillCorner | Metrica g3 |
|---|---|---|---|
| nearest player + 4 s majority vote (`starter/baseline.py` logic) | 0.202 / 0.533 | 0.318 / 0.702 | 0.416 / 0.812 |
| trained by SGD, one model (`models/expB_bw3met6k_s0.pt`) | 0.724 / 0.912 | 0.726 / 0.913 | 0.842 / 0.934 |

The baseline invents about twice as many possessions as there are (5,734 against 2,843 over the 20
benchmark halves). The trained single model is a little below the published 3-model ensemble
(0.741 / 0.747 / 0.855), as expected.

## Nov 10 — experiment A: one reply, no feedback

`wsp/runs/expA_scratch_oneshot/turn_00/`. One fresh conversation, the 24,154-character prompt
(`prompt.txt`), one reply (`answer.md`), effort `xhigh`.

- 1,216 s wall time; 114,658 output tokens, of which 110,189 were reasoning. The reasoning itself is
  not exposed by the CLI (only an opaque signature, which the logs omit); the reply is 8,051
  characters: a design note and the edits block.
- `audit.tool_free: true` (the session listed no tools and no MCP servers; the reply has no tool call).
- 140 edit statements, 140 applied, 0 errors. **396 of 698,115 weights are nonzero.**

What it wrote (its own description, `answer.md`): hinge units for "a player is close to the ball and
moving with it", closeness to the lines, ball speed and gap length; 9-frame averages in the conv
layer; in the first GRU a *last-controller latch* (possession stays with the last team through gaps
and ball flights), forward and backward *dead-ball latches* set by "ball out" or "ball stationary"
and reset by live motion, slow and fast running averages of player speed whose difference says
"players are slower than usual", and a restart-taker trace; a head that outputs `H − A = 2p` and a
clipped dead-ball score, with the scale chosen against the decoder's penalties ("about 2 s of clear
control by the other team is needed to pay the 32 for a there-and-back possession change").

**Result on the held-out benchmark** (spellF1 / macroF1):

| | DFL | SkillCorner | Metrica g3 |
|---|---|---|---|
| nearest-player baseline | 0.202 / 0.533 | 0.318 / 0.702 | 0.416 / 0.812 |
| **A: written by hand, one shot, 396 weights** | **0.378 / 0.741** | **0.471 / 0.802** | **0.546 / 0.809** |
| trained by SGD, same architecture, one model | 0.724 / 0.912 | 0.726 / 0.913 | 0.842 / 0.934 |

So a network that has never seen a frame of data, written in one reply, clearly beats the baseline
on the ranking metric on all three sources (+0.13 to +0.18) and closes roughly a third of the gap
to the trained network. Where it falls short: it produces too few possessions (2,285 against 2,843
true; the trained model 2,623) and over-calls dead ball (D recall 0.94 but precision 0.79; H and A
recall about 0.70). On Metrica its D precision is only 0.67: it reads long ball gaps as stoppages.

**Usage note.** The reply cost about four times the output tokens I had estimated. Asked how to
budget B and C, the team chose to run C next, at effort `high`.
