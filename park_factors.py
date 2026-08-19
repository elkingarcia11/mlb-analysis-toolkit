"""Fetch and cache three-year rolling Baseball Savant park factors."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from http_utils import default_ssl_context
PARK_FACTORS_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
USER_AGENT = "Mozilla/5.0 (compatible; mlb-analysis-toolkit/1.0)"
FACTOR_COLUMNS = (
    "index_runs",
    "index_woba",
    "index_obp",
    "index_so",
    "index_bb",
    "index_hits",
    "index_1b",
    "index_2b",
    "index_3b",
    "index_hr",
)


def default_cache_path(season: int, data_dir: Path = Path("data")) -> Path:
    return data_dir / "cache" / f"park_factors_{season}.json"


def fetch_park_factors(season: int, *, rolling: int = 3) -> dict[str, dict[str, str]]:
    params = {
        "batSide": "",
        "condition": "All",
        "parks": "mlb",
        "rolling": rolling,
        "stat": "index_wOBA",
        "type": "year",
        "year": season,
    }
    url = f"{PARK_FACTORS_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60, context=default_ssl_context()) as response:
        html = response.read().decode("utf-8", "replace")

    match = re.search(r"var data = (\[.*?\]);\s*var ", html, flags=re.DOTALL)
    if not match:
        # Current page places the next script statement after the array.
        match = re.search(r"var data = (\[.*?\]);\s*</script>", html, flags=re.DOTALL)
    if not match:
        raise RuntimeError("Could not locate Baseball Savant park-factor data")

    rows: list[dict[str, Any]] = json.loads(match.group(1))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        venue_id = str(row.get("venue_id") or "").strip()
        if not venue_id:
            continue
        factor = {
            "venue_name": str(row.get("venue_name") or ""),
            "year_range": str(row.get("year_range") or ""),
        }
        for col in FACTOR_COLUMNS:
            factor[col] = str(row.get(col) or "")
        out[venue_id] = factor
    return out


def load_park_factors(
    season: int,
    *,
    data_dir: Path = Path("data"),
    refresh: bool = False,
) -> dict[str, dict[str, str]]:
    path = default_cache_path(season, data_dir)
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): dict(v) for k, v in payload.items()}

    factors = fetch_park_factors(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(factors, indent=2, sort_keys=True), encoding="utf-8")
    return factors
