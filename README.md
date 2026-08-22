# MLB Analysis Toolkit

This project fetches MLB data, builds daily game panels, and trains/predicts team and player outcomes.

The main entry point is `workflow.py`, which runs the full daily pipeline.

## Requirements

- Python 3.10+
- Network access for MLB Stats API, ESPN scoreboard, and park-factor data
- Google Cloud Storage access when using the default GCS-backed workflow mode
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

GCS is the default source of truth. The command automatically:

1. Downloads **all existing data** from `gs://mlb-analysis-toolkit` into a temporary workspace
2. Merges any newly-fetched local `data/` files on top
3. Fetches new stats/schedule/odds for today
4. Backfills prior days (schedule, panels, boxscore labels) — covering old raw days already in GCS
5. Retrains every model using the full merged history
6. Predicts today's matchups
7. Uploads everything (new raw days, panels, models, predictions) back to `gs://mlb-analysis-toolkit`

Credentials are auto-detected from `gcs-sa.json` in the project root, or
`GOOGLE_APPLICATION_CREDENTIALS`. No extra flags are needed.

```bash
python workflow.py --local-only          # old behavior: local data/ only
python workflow.py --gcs-credentials other-sa.json
python workflow.py --data-uri gs://my-bucket/my-prefix
python workflow.py --target player_hits   # restrict to one target
```

### Useful options

```bash
python workflow.py --as-of 2026-08-18
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
up-to-date models are reused unless their force flags are supplied. In GCS
mode (the default), retraining is forced so the merged history is always used.

### GCS-backed data

By default `python workflow.py` reads and writes `gs://mlb-analysis-toolkit`.
The bucket may be changed with `--data-uri gs://bucket[/prefix]`. The workflow
downloads the whole prefix into a temporary workspace, merges local `data/`
files on top, runs the normal pipeline, and uploads the entire workspace back
to the same prefix, so nothing is ever lost.

The first run may also explicitly upload the local `data/` directory:

```bash
python workflow.py \
	--data-uri gs://mlb-analysis-toolkit \
	--migrate-local-data \
	--gcs-credentials gcs-sa.json
```

On subsequent runs `--migrate-local-data` is unnecessary (the local merge
happens automatically). Instead of `--gcs-credentials`, set
`GOOGLE_APPLICATION_CREDENTIALS` to the service-account JSON path. Never commit
that credential file; `gcs-sa.json` is ignored by Git.

Use `--local-only` to keep everything in the local `data/` directory with no
GCS interactions.

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
