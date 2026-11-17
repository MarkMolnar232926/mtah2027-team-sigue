# Team Sigue — More than a Hack 2027, Phase 1: Ball Possession

Frame-level ball possession (`H` / `A` / `D`) for the BAR-ATH match, computed from the **automatic
tactical tracking (ATD) alone**. A bidirectional GRU trained **only on external open data** (DFL,
SkillCorner, Metrica sample games), averaged over 3 seeds × 4 symmetric pitch views, followed by a
Viterbi decoder that charges a cost for every change of state. No label, statistic or parameter
comes from the provided game's ground truth or video.

- **Technical document:** [docs/TECHNICAL.md](docs/TECHNICAL.md) (also submitted as `Sigue.pdf`)
- **Solution code and retraining:** [solution/README.md](solution/README.md)

## Regenerate the submission

Python 3.12. The match data is not in this repository; unpack the competition bundle into `data/`
(see [DATA.md](DATA.md)). Only `data/tactical/BAR-ATH_tactical_atd.{xml,txt}` is read.

```bash
python -m venv .venv
# GPU (CUDA 12.4):
.venv/Scripts/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# or CPU:  .venv/Scripts/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r solution/requirements-solution.txt
.venv/Scripts/python solution/reproduce.py          # -> submission/Sigue.csv, validated
```

On Linux/macOS use `.venv/bin/` instead of `.venv/Scripts/`. `reproduce.py` parses the ATD, runs the
three published models in `models/` and validates the output with the official
`evaluation/validate_submission.py`. It regenerates the submitted file byte for byte, on GPU or CPU.

To retrain from scratch, download the three external datasets (links in
[solution/README.md](solution/README.md)) and run `python solution/train_all.py` (≈ 1.5 h on an
RTX 4060).

## Repository layout

```
solution/                 our pipeline: parsing, features, model, training, prediction
  reproduce.py            one command: ATD -> submission/Sigue.csv
  train_all.py            one command: external data -> models/final_met_s*.pt
  logs/                   training-set build logs (external data only)
models/
  final_met_s{0,1,2}.pt   the three models used for the submission
  decoder.json            Viterbi switch penalties (chosen by cross-validation on DFL)
  experiments/            weights of the variants compared in TECHNICAL.md §1 (not used)
docs/                     technical document and figures
evaluation/               the organisers' scorer and validator, unchanged
starter/, games.py        the organisers' starter kit, unchanged
data/, external/          match data and external datasets — gitignored, never committed
```

The competition description from the organisers is kept unchanged in [CHALLENGE.md](CHALLENGE.md),
[RULES.md](RULES.md), [SUBMISSION.md](SUBMISSION.md), [EVALUATION.md](EVALUATION.md),
[DATA.md](DATA.md), [FAQ.md](FAQ.md) and [OVERVIEW.md](OVERVIEW.md).

## Data and licences

The match data is licensed for the competition only and is **not** in this repository. External
training data: DFL open data (Bassek et al., 2025, CC BY 4.0), SkillCorner Open Data (MIT) and
Metrica Sports sample data. Code and documentation: MIT, see [LICENSE](LICENSE).
