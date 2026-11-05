# Team Sigue solution

Method, data and rule compliance: [docs/TECHNICAL.md](../docs/TECHNICAL.md).

## Regenerate the submission

```bash
python -m venv .venv
.venv/Scripts/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/pip install -r solution/requirements-solution.txt
# unpack the match data into data/ (see ../DATA.md)
.venv/Scripts/python solution/reproduce.py         # -> submission/Sigue.csv
```

Python 3.12. Only `data/tactical/BAR-ATH_tactical_atd.{xml,txt}` is read. CPU works too (slower);
the output is the same file.

## Retrain from external data

1. **DFL open data** (CC BY 4.0) — Bassek et al. (2025), *An integrated dataset of spatiotemporal
   and event data in elite soccer*, Scientific Data, data: https://doi.org/10.6084/m9.figshare.28196177. Put the `DFL_02_01_matchinformation_*.xml`,
   `DFL_03_02_events_raw_*.xml` and `DFL_04_03_positions_raw_observed_*.xml` files of the 7 matches
   in `external/raw/`.
2. **SkillCorner Open Data** (MIT) — `git clone https://github.com/SkillCorner/opendata
   external/skillcorner` (needs git-lfs).
3. **Metrica Sports sample data** (published for community use; source acknowledged) —
   download `data/Sample_Game_{1,2,3}/*` from https://github.com/metrica-sports/sample-data into
   `external/metrica/`.
4. `python solution/train_all.py`

## Files

| file | role |
|---|---|
| `dfl_parse.py`, `skillcorner_parse.py`, `metrica_parse.py` | external data → per-half arrays (+ labels from Metrica events) |
| `dfl_labels.py` | DFL per-touch possession → controlled possession (2 s rule) |
| `degrade.py` | makes clean tracking look like automatic tracking |
| `features.py` | 48 identity-free per-frame features |
| `build_train.py` | augmented training set |
| `model.py` | bidirectional GRU + Viterbi decoder |
| `train.py` | training, cross-validation, decoder tuning |
| `eval_external.py`, `eval_fold.py` | scoring on held-out external matches (SkillCorner; fixed DFL + SkillCorner benchmark) |
| `kalman.py`, `kalman_tune.py` | optional Kalman-smoothed ball features (tested, not used in the final model) |
| `atd_parse.py`, `predict.py` | competition ATD → submission |
| `reproduce.py`, `train_all.py` | one-command entry points |

Acknowledgements: tracking data courtesy of the DFL (Bassek et al., 2025), SkillCorner and Metrica
Sports (sample data).
