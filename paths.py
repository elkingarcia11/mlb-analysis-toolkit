"""
Shared data-layout helpers.

Canonical layout:
  data/raw/YYYY-MM-DD/teams.csv
  data/raw/YYYY-MM-DD/players.csv
  data/raw/YYYY-MM-DD/games.csv
  data/panels/YYYY-MM-DD/team_game.csv
  data/panels/YYYY-MM-DD/player_game.csv

Odds stamps the as-of day folder. Results settles every raw day folder
strictly before as-of. Legacy flat folders (teams.csv directly under
--data-dir) still resolve for smoke tests / overrides.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path


def raw_day_dir(as_of: date, data_dir: Path = Path("data")) -> Path:
    """Dated raw snapshot folder: data/raw/YYYY-MM-DD."""
    return data_dir / "raw" / as_of.isoformat()


def panels_day_dir(as_of: date, data_dir: Path = Path("data")) -> Path:
    """Dated ML panel folder: data/panels/YYYY-MM-DD."""
    return data_dir / "panels" / as_of.isoformat()


def resolve_day_dir(data_dir: Path, as_of: date) -> Path:
    """
    Folder that holds teams.csv / players.csv for as_of.

    If data_dir already contains those CSVs (legacy/smoke), use it as-is;
    otherwise use data/raw/YYYY-MM-DD.
    """
    if (data_dir / "teams.csv").exists() or (data_dir / "players.csv").exists():
        return data_dir
    return raw_day_dir(as_of, data_dir)


def resolve_day_csvs(
    *,
    data_dir: Path,
    as_of: date,
    teams_csv: Path | None = None,
    players_csv: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve (teams.csv, players.csv) for a single calendar day."""
    day = resolve_day_dir(data_dir, as_of)
    return (
        teams_csv or (day / "teams.csv"),
        players_csv or (day / "players.csv"),
    )


def iter_raw_day_dirs(data_dir: Path) -> list[tuple[date, Path]]:
    """
    Yield (date, path) for every data/raw/YYYY-MM-DD folder.

    If raw/ is missing but data_dir itself has CSVs, yield a single
    synthetic entry keyed by today (caller should still filter on
    date_fetched inside the file).
    """
    raw = data_dir / "raw"
    out: list[tuple[date, Path]] = []
    if raw.is_dir():
        for path in sorted(raw.iterdir()):
            if not path.is_dir():
                continue
            try:
                day = date.fromisoformat(path.name)
            except ValueError:
                continue
            out.append((day, path))
        return out

    if (data_dir / "teams.csv").exists() or (data_dir / "players.csv").exists():
        out.append((date.today(), data_dir))
    return out
