# MLB Analysis Toolkit

This project fetches MLB data, builds daily game panels, and trains/predicts team and player outcomes.

The main entry point is `workflow.py`, which runs the full daily pipeline.

## Requirements

- Python 3.10+
- `pandas`
- `scikit-learn`
- `certifi`
- `pytest` (for tests)

## Quick start

```bash
cd /Users/elkingarcia/Documents/python/mlb-analysis-toolkit

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install pandas scikit-learn certifi pytest
```

## Run the main workflow

```bash
python workflow.py
```

This performs the daily MLB workflow, including season detection, data fetches, alignment, training, and prediction steps.

### Useful options

```bash
python workflow.py --as-of 2026-08-18
python workflow.py --only predict
python workflow.py --only train
python workflow.py --skip-fetch
python workflow.py --skip-train
python workflow.py --skip-predict
```

## Train and predict directly

The project also exposes direct model commands in `train_predict.py`:

```bash
python train_predict.py train --target team_win
python train_predict.py train --all

python train_predict.py predict --as-of 2026-08-09 --target team_win
python train_predict.py predict --as-of 2026-08-09 --all
```

## Notes

- The workflow is designed for a morning cron run and can skip certain steps based on the detected season phase.
- Data is stored under `data/` and model outputs are written to `data/models/` and `data/predictions/`.
- If you are running tests:

```bash
pytest
```
