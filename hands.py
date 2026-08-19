"""
Fetch/cache MLB player bat side and pitch hand from Stats API.

Writes data/cache/people_hands.csv and reuses it across daily runs.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from http_utils import default_ssl_context
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
USER_AGENT = "mlb-analysis-toolkit/1.0 (+hands)"
MAX_RETRIES = 3
BATCH_SIZE = 50
REQUEST_PAUSE_S = 0.1

HAND_COLUMNS = ("player_id", "bat_side", "pitch_hand", "full_name")


def default_cache_path(data_dir: Path = Path("data")) -> Path:
    return data_dir / "cache" / "people_hands.csv"


def _http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=45, context=default_ssl_context()) as resp:
                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last_err = err
            time.sleep(min(2 ** attempt, 6))
    raise RuntimeError(f"GET failed after {MAX_RETRIES} tries: {url}") from last_err


def load_hands_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = str(row.get("player_id") or "").strip()
            if not pid:
                continue
            out[pid] = {
                "bat_side": str(row.get("bat_side") or "").strip().upper(),
                "pitch_hand": str(row.get("pitch_hand") or "").strip().upper(),
                "full_name": str(row.get("full_name") or ""),
            }
    return out


def save_hands_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(HAND_COLUMNS))
        writer.writeheader()
        for pid in sorted(cache, key=lambda x: int(x) if x.isdigit() else x):
            row = cache[pid]
            writer.writerow(
                {
                    "player_id": pid,
                    "bat_side": row.get("bat_side", ""),
                    "pitch_hand": row.get("pitch_hand", ""),
                    "full_name": row.get("full_name", ""),
                }
            )


def fetch_people_hands(player_ids: Iterable[str]) -> dict[str, dict[str, str]]:
    """Batch-fetch batSide / pitchHand for the given player ids."""
    ids = sorted({str(p).strip() for p in player_ids if str(p).strip()})
    out: dict[str, dict[str, str]] = {}
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        payload = _http_get_json(PEOPLE_URL, {"personIds": ",".join(batch)})
        for person in payload.get("people") or []:
            pid = str(person.get("id") or "").strip()
            if not pid:
                continue
            bat = person.get("batSide") or {}
            hand = person.get("pitchHand") or {}
            out[pid] = {
                "bat_side": str(bat.get("code") or "").strip().upper(),
                "pitch_hand": str(hand.get("code") or "").strip().upper(),
                "full_name": str(person.get("fullName") or ""),
            }
        time.sleep(REQUEST_PAUSE_S)
    return out


def ensure_hands(
    player_ids: Iterable[str],
    *,
    cache_path: Path | None = None,
    data_dir: Path = Path("data"),
) -> dict[str, dict[str, str]]:
    """
    Return hands map for all requested ids, fetching only cache misses.
    """
    path = cache_path or default_cache_path(data_dir)
    cache = load_hands_cache(path)
    wanted = {str(p).strip() for p in player_ids if str(p).strip()}
    missing = [pid for pid in wanted if pid not in cache]
    if missing:
        try:
            fetched = fetch_people_hands(missing)
            cache.update(fetched)
        except Exception:
            # Offline / bad ids: leave blanks rather than aborting align.
            fetched = {}
        for pid in missing:
            cache.setdefault(pid, {"bat_side": "", "pitch_hand": "", "full_name": ""})
        save_hands_cache(path, cache)
    return {pid: cache[pid] for pid in wanted if pid in cache}


def platoon_advantage(bat_side: str, pitch_hand: str) -> str:
    """1 if traditional platoon edge (opposite hands), 0 if same, blank if unknown/switch."""
    bats = (bat_side or "").strip().upper()
    throws = (pitch_hand or "").strip().upper()
    if bats not in {"L", "R"} or throws not in {"L", "R"}:
        return ""
    return "1" if bats != throws else "0"
