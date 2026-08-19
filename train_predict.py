"""
Train / predict ML targets from aligned team_game and player_game panels.

Models (sklearn HistGradientBoosting):
  team_win              — P(win) from team_game.label_win
  team_total_runs       — predicted combined runs (home row view uses label_total_runs)
  team_run_diff         — predicted run differential
  player_hits           — predicted batter hits
  player_home_runs      — predicted batter HRs
  player_strikeouts     — predicted batter strikeouts
  pitcher_strikeouts    — predicted pitcher Ks (rows with pitching IP or probable)

Usage:
  python train_predict.py train --target team_win
  python train_predict.py train --all
  python train_predict.py predict --as-of 2026-08-09 --target team_win
  python train_predict.py predict --as-of 2026-08-09 --all
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from paths import panels_day_dir

MODEL_DIR_DEFAULT = Path("data/models")
PRED_DIR_DEFAULT = Path("data/predictions")

FEATURE_PREFIXES_TEAM = ("team_", "opp_", "ctx_", "is_home")
FEATURE_PREFIXES_PLAYER = (
    "player_",
    "opp_pitcher_",
    "park_factor_",
    "batter_vs_hand_",
    "batter_weekday_",
    "is_home",
    "is_probable_pitcher",
)
FEATURE_EXACT_PLAYER = {
    "is_home",
    "is_probable_pitcher",
    "venue_id",
    "is_day_game",
    "game_hour_utc",
    "batter_bats_R",
    "batter_bats_L",
    "batter_bats_S",
    "opp_throws_R",
    "opp_throws_L",
    "platoon_advantage",
}

MIN_LABELED_ROWS = 12


def _is_feature_col(
    name: str,
    prefixes: tuple[str, ...],
    *,
    exact: set[str] | None = None,
) -> bool:
    if exact and name in exact:
        return True
    if name in {"is_home", "is_probable_pitcher"}:
        return True
    return any(name.startswith(p) for p in prefixes if p.endswith("_"))


def _team_home_only(df: pd.DataFrame) -> pd.DataFrame:
    if "is_home" in df.columns:
        return df.loc[df["is_home"].astype(str) == "1"].copy()
    return df


def _team_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _player_batter_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Training rows with box-score metadata: keep batters who appeared with AB > 0.
    Unlabeled rows (today's slate): keep all so predict covers every matchup.
    """
    if "label_appeared" not in df.columns or "label_at_bats" not in df.columns:
        return df

    label_col = "label_hits" if "label_hits" in df.columns else ""
    if label_col:
        has_label = df[label_col].astype(str).str.strip().ne("")
    else:
        has_label = pd.Series(False, index=df.index)

    appeared = df["label_appeared"].astype(str) == "1"
    ab = pd.to_numeric(df["label_at_bats"], errors="coerce")

    # Upcoming games — no outcome yet.
    predict_rows = ~has_label
    # Completed games — only train/score rows with a real plate appearance.
    labeled_batter = has_label & appeared & ab.fillna(0).gt(0)
    # Legacy panels stamped before label_appeared existed.
    labeled_legacy = has_label & ~appeared & df["label_at_bats"].astype(str).str.strip().eq("")

    return df.loc[predict_rows | labeled_batter | labeled_legacy].copy()


def _player_pitcher_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "label_innings_pitched" in df.columns:
        ip = df["label_innings_pitched"].astype(str).str.strip()
        mask = ip.ne("") & ip.ne("nan")
        if mask.any():
            return df.loc[mask].copy()
    if "is_probable_pitcher" in df.columns:
        return df.loc[df["is_probable_pitcher"].astype(str) == "1"].copy()
    return df.iloc[0:0].copy()


TargetSpec = dict[str, Any]

TARGETS: dict[str, TargetSpec] = {
    "team_win": {
        "panel": "team_game",
        "label": "label_win",
        "task": "classification",
        "filter": _team_frame,
        "feature_prefixes": FEATURE_PREFIXES_TEAM,
    },
    "team_total_runs": {
        "panel": "team_game",
        "label": "label_total_runs",
        "task": "regression",
        "filter": _team_home_only,
        "feature_prefixes": FEATURE_PREFIXES_TEAM,
    },
    "team_run_diff": {
        "panel": "team_game",
        "label": "label_run_diff",
        "task": "regression",
        "filter": _team_frame,
        "feature_prefixes": FEATURE_PREFIXES_TEAM,
    },
    "player_hits": {
        "panel": "player_game",
        "label": "label_hits",
        "task": "regression",
        "filter": _player_batter_frame,
        "feature_prefixes": FEATURE_PREFIXES_PLAYER,
        "feature_exact": FEATURE_EXACT_PLAYER,
    },
    "player_home_runs": {
        "panel": "player_game",
        "label": "label_home_runs",
        "task": "regression",
        "filter": _player_batter_frame,
        "feature_prefixes": FEATURE_PREFIXES_PLAYER,
        "feature_exact": FEATURE_EXACT_PLAYER,
    },
    "player_strikeouts": {
        "panel": "player_game",
        "label": "label_strikeouts",
        "task": "regression",
        "filter": _player_batter_frame,
        "feature_prefixes": FEATURE_PREFIXES_PLAYER,
        "feature_exact": FEATURE_EXACT_PLAYER,
    },
    "pitcher_strikeouts": {
        "panel": "player_game",
        "label": "label_pitcher_strikeouts",
        "task": "regression",
        "filter": _player_pitcher_frame,
        "feature_prefixes": FEATURE_PREFIXES_PLAYER,
        "feature_exact": FEATURE_EXACT_PLAYER,
    },
}

def list_panel_days(data_dir: Path) -> list[date]:
    root = data_dir / "panels"
    if not root.is_dir():
        return []
    out: list[date] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        try:
            out.append(date.fromisoformat(path.name))
        except ValueError:
            continue
    return out


def load_panels(
    *,
    data_dir: Path,
    panel: str,
    as_of: date | None = None,
    through: date | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    days = list_panel_days(data_dir)
    for day in days:
        if as_of is not None and day != as_of:
            continue
        if through is not None and day > through:
            continue
        path = panels_day_dir(day, data_dir) / f"{panel}.csv"
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=False))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def feature_matrix(
    df: pd.DataFrame,
    prefixes: tuple[str, ...],
    feature_cols: list[str] | None = None,
    *,
    exact: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns if _is_feature_col(c, prefixes, exact=exact)
        ]
    X = df.reindex(columns=feature_cols, fill_value="")
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, feature_cols


def labeled_xy(
    df: pd.DataFrame,
    spec: TargetSpec,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    filtered = spec["filter"](df)
    label = spec["label"]
    if label not in filtered.columns:
        return pd.DataFrame(), pd.Series(dtype=float), []
    y_raw = filtered[label].astype(str).str.strip()
    mask = y_raw.ne("") & y_raw.ne("nan")
    filtered = filtered.loc[mask].copy()
    y = pd.to_numeric(filtered[label], errors="coerce")
    mask2 = y.notna()
    filtered = filtered.loc[mask2].copy()
    y = y.loc[mask2]
    X, cols = feature_matrix(
        filtered,
        spec["feature_prefixes"],
        exact=spec.get("feature_exact"),
    )
    usable = [col for col in cols if X[col].notna().any() and X[col].nunique(dropna=True) > 1]
    X = X.loc[:, usable]
    cols = usable
    return X, y, cols


def train_target(
    target: str,
    *,
    data_dir: Path,
    model_dir: Path,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    if target not in TARGETS:
        raise KeyError(f"unknown target {target!r}; choose from {sorted(TARGETS)}")
    spec = TARGETS[target]
    df = load_panels(data_dir=data_dir, panel=spec["panel"])
    X, y, feature_cols = labeled_xy(df, spec)
    if len(X) < MIN_LABELED_ROWS:
        raise RuntimeError(
            f"{target}: need >={MIN_LABELED_ROWS} labeled rows, got {len(X)}. "
            "Accumulate more raw days / boxscores first."
        )

    stratify = y if spec["task"] == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if spec["task"] == "classification":
        model: Any = HistGradientBoostingClassifier(max_depth=4, max_iter=150)
        model.fit(X_train, y_train.astype(int))
        proba = model.predict_proba(X_test)
        pred = model.predict(X_test)
        metrics = {
            "task": "classification",
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "accuracy": float(accuracy_score(y_test.astype(int), pred)),
        }
        try:
            metrics["log_loss"] = float(log_loss(y_test.astype(int), proba, labels=model.classes_))
        except ValueError:
            metrics["log_loss"] = None
    else:
        model = HistGradientBoostingRegressor(max_depth=4, max_iter=150)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "task": "regression",
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "target": target,
        "panel": spec["panel"],
        "label": spec["label"],
        "task": spec["task"],
        "feature_cols": feature_cols,
        "model": model,
        "metrics": metrics,
    }
    path = model_dir / f"{target}.pkl"
    with path.open("wb") as fh:
        pickle.dump(artifact, fh)
    meta_path = model_dir / f"{target}.json"
    meta_path.write_text(json.dumps({"target": target, "metrics": metrics, "n_features": len(feature_cols)}, indent=2))
    return {"target": target, "path": str(path), "metrics": metrics, "n_features": len(feature_cols)}


def predict_target(
    target: str,
    *,
    as_of: date,
    data_dir: Path,
    model_dir: Path,
    pred_dir: Path,
) -> dict[str, Any]:
    if target not in TARGETS:
        raise KeyError(f"unknown target {target!r}")
    spec = TARGETS[target]
    model_path = model_dir / f"{target}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path} (run train first)")
    with model_path.open("rb") as fh:
        artifact = pickle.load(fh)
    model = artifact["model"]
    feature_cols: list[str] = list(artifact["feature_cols"])

    df = load_panels(data_dir=data_dir, panel=spec["panel"], as_of=as_of)
    if df.empty:
        raise FileNotFoundError(f"no {spec['panel']} panel for {as_of.isoformat()}")
    filtered = spec["filter"](df)
    X, _ = feature_matrix(filtered, spec["feature_prefixes"], feature_cols=feature_cols)

    if spec["task"] == "classification":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # probability of positive class (win=1) when present
            classes = list(getattr(model, "classes_", [0, 1]))
            if 1 in classes:
                idx = classes.index(1)
                score = proba[:, idx]
            else:
                score = proba[:, -1]
            pred = model.predict(X)
        else:
            pred = model.predict(X)
            score = pred
        out = filtered.copy()
        out["prediction"] = pred
        out["prediction_proba"] = score
    else:
        pred = model.predict(X)
        out = filtered.copy()
        out["prediction"] = pred

    id_cols = [
        c
        for c in (
            "game_date",
            "gamePk",
            "team_id",
            "team_abbr",
            "player_id",
            "player_name",
            "is_home",
            "is_probable_pitcher",
            spec["label"],
        )
        if c in out.columns
    ]
    keep = id_cols + [c for c in out.columns if c.startswith("prediction")]
    result = out[keep].copy()

    dest = pred_dir / as_of.isoformat()
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{target}.csv"
    result.to_csv(path, index=False)
    return {
        "target": target,
        "as_of": as_of.isoformat(),
        "rows": int(len(result)),
        "path": str(path),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", type=Path, default=Path("data"))
        p.add_argument("--model-dir", type=Path, default=MODEL_DIR_DEFAULT)
        p.add_argument(
            "--target",
            choices=sorted(TARGETS),
            action="append",
            default=None,
            help="Target name (repeatable). Default with --all or required.",
        )
        p.add_argument("--all", action="store_true", help="All known targets")

    train_p = sub.add_parser("train", help="Fit models from labeled panels")
    add_shared(train_p)

    pred_p = sub.add_parser("predict", help="Score a day's panel with saved models")
    add_shared(pred_p)
    pred_p.add_argument("--as-of", type=date.fromisoformat, required=True)
    pred_p.add_argument("--pred-dir", type=Path, default=PRED_DIR_DEFAULT)
    return parser.parse_args(argv)


def _resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(TARGETS)
    if not args.target:
        raise SystemExit("pass --target NAME and/or --all")
    return list(dict.fromkeys(args.target))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    targets = _resolve_targets(args)
    if args.command == "train":
        for target in targets:
            try:
                summary = train_target(
                    target,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                )
            except (RuntimeError, FileNotFoundError, KeyError) as err:
                print(f"{target}: SKIP — {err}", file=sys.stderr)
                continue
            m = summary["metrics"]
            if m["task"] == "classification":
                print(
                    f"{target}: trained n={m['n_train']}+{m['n_test']} "
                    f"acc={m['accuracy']:.3f} -> {summary['path']}"
                )
            else:
                print(
                    f"{target}: trained n={m['n_train']}+{m['n_test']} "
                    f"mae={m['mae']:.3f} r2={m['r2']:.3f} -> {summary['path']}"
                )
        return 0

    if args.command == "predict":
        for target in targets:
            try:
                summary = predict_target(
                    target,
                    as_of=args.as_of,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    pred_dir=args.pred_dir,
                )
            except (FileNotFoundError, KeyError, RuntimeError) as err:
                print(f"{target}: SKIP — {err}", file=sys.stderr)
                continue
            print(f"{target}: wrote {summary['rows']} rows -> {summary['path']}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
