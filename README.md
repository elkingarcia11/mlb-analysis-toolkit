# MLB Analysis Toolkit

This project fetches MLB data, builds daily game panels, and trains/predicts team and player outcomes.

The main entry point is `workflow.py`, which runs the full daily pipeline.

## Requirements

- Python 3.10+
- Network access for MLB Stats API, ESPN scoreboard, and park-factor data
- Google Cloud Storage access when using the `--data-uri` workflow mode
- No API key is required for the supported data sources

## Quick start

```bash
cd /Users/elkingarcia/Documents/python/mlb-analysis-toolkit

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

# For development and tests, install the test dependency too:
pip install -r requirements-dev.txt
```

Runtime dependencies are listed in `requirements.txt`; `requirements-dev.txt`
includes those dependencies plus `pytest`.

## Run the main workflow

```bash
python workflow.py
```

This performs the daily MLB workflow, including season detection, data fetches, alignment, training, and prediction steps.

### Useful options

```bash
python workflow.py --as-of 2026-08-18
python workflow.py --data-uri gs://mlb-analysis-toolkit --gcs-credentials gcs-sa.json
python workflow.py --only predict
python workflow.py --only train
python workflow.py --only backfill
python workflow.py --skip-fetch
python workflow.py --skip-train
python workflow.py --skip-predict
python workflow.py --skip-backfill
python workflow.py --force-stats
python workflow.py --force-train
python workflow.py --include-today-boxscores
python workflow.py --target team_win --target player_hits
```

`--as-of` uses an ISO date (`YYYY-MM-DD`). `--data-dir` changes the default
`data/` location. Use `--season-phase regular`, `--season-phase postseason`,
or `--season-phase offseason` to override automatic season detection. The
workflow skips today's boxscores by default because a morning run happens
before games finish; `--include-today-boxscores` is intended for later replays.

The workflow runs results settlement, prior-day backfill, today's stats,
schedule, ESPN DraftKings odds, panel alignment, model training, and
prediction as applicable to the detected season phase. Existing snapshots and
up-to-date models are reused unless their force flags are supplied.

### GCS-backed data

Use the GCS mode to keep raw snapshots, caches, panels, models, and predictions
in a bucket instead of the local `data/` directory:

```bash
python workflow.py \
	--data-uri gs://mlb-analysis-toolkit \
	--migrate-local-data \
	--gcs-credentials gcs-sa.json
```

Run `--migrate-local-data` once, before the first GCS-backed workflow, to copy
the existing local `data/` directory into the bucket. Omit it on subsequent
runs so the bucket remains the source of truth. Use `--data-dir` if the local
source directory is not the default `data/`.

The URI may include a prefix, such as `gs://mlb-analysis-toolkit/data`. The
workflow downloads that prefix into a temporary workspace, runs the normal
pipeline, and uploads the resulting files back to the same prefix. Existing
local `--data-dir` behavior is unchanged. Instead of `--gcs-credentials`, set
`GOOGLE_APPLICATION_CREDENTIALS` to the service-account JSON path. Never commit
that credential file; `gcs-sa.json` is ignored by Git.

## Train and predict directly

The project also exposes direct model commands in `train_predict.py`:

```bash
python train_predict.py train --target team_win
python train_predict.py train --all

python train_predict.py predict --as-of 2026-08-09 --target team_win
python train_predict.py predict --as-of 2026-08-09 --all
```

Available targets are `team_win`, `team_total_runs`, `team_run_diff`,
`player_hits`, `player_home_runs`, `player_strikeouts`, and
`pitcher_strikeouts`.

## Notes

- The workflow is designed for a morning cron run and can skip certain steps based on the detected season phase.
- Raw daily snapshots are stored under `data/raw/YYYY-MM-DD/` locally, or the
	equivalent prefix in GCS.
- Aligned labeled panels are stored under `data/panels/YYYY-MM-DD/` locally, or
	the equivalent prefix in GCS.
- Saved models are written to `data/models/`; predictions are written to
	`data/predictions/YYYY-MM-DD/`, locally or in the configured GCS prefix.
- Runtime logs can be redirected to `logs/`; that directory is ignored by git.
- If you are running tests:

```bash
pytest
```
