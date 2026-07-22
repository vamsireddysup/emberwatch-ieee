"""Dataset contracts and bounded, chronology-preserving sampling for EmberWatch ML."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURE_DIR = REPO_ROOT / "data" / "processed"

FEATURE_COLUMNS = (
    "ambient_temp_c",
    "delta_c",
    "slope_1min",
    "slope_5min",
    "slope_15min",
    "variance_30min",
)
LABELS = ("Normal", "Warming", "Anomaly")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}

REQUIRED_COLUMNS = {
    "timestamp",
    *FEATURE_COLUMNS,
    "label",
    "event_id",
    "split",
}


def discover_feature_files(
    feature_dir: Path = DEFAULT_FEATURE_DIR, stations: Iterable[str] | None = None
) -> list[Path]:
    paths = sorted(feature_dir.glob("features_v2_*.csv"))
    if stations:
        wanted = set(stations)
        paths = [path for path in paths if path.stem.removeprefix("features_v2_") in wanted]
    if not paths:
        raise FileNotFoundError(f"No features_v2_*.csv files found in {feature_dir}")
    return paths


def _context_sample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Keep complete fault neighborhoods and add contiguous normal background windows."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df

    n = len(df)
    keep = np.zeros(n, dtype=bool)
    event_ids = [value for value in df["event_id"].unique() if value > 0]
    event_budget = int(max_rows * 0.70)
    per_event = max(12, event_budget // max(1, len(event_ids)))
    labels = df["label"].to_numpy()
    ids = df["event_id"].to_numpy()
    for event_id in event_ids:
        rows = np.flatnonzero(ids == event_id)
        positive = rows[labels[rows] != "Normal"]
        center = int(positive[0] if len(positive) else rows[len(rows) // 2])
        start = max(int(rows[0]), center - per_event // 3)
        end = min(int(rows[-1]) + 1, start + per_event)
        start = max(int(rows[0]), end - per_event)
        keep[start:end] = True

    budget = max(0, max_rows - int(keep.sum()))
    if budget:
        rng = np.random.default_rng(seed)
        window = 24  # two hours of contiguous nominal behavior
        candidates = np.flatnonzero(~keep)
        starts = candidates[::window]
        rng.shuffle(starts)
        for start in starts:
            if budget <= 0:
                break
            end = min(n, start + min(window, budget))
            available = ~keep[start:end]
            keep[start:end] |= available
            budget -= int(available.sum())

    sampled = df.loc[keep].copy()
    if len(sampled) > max_rows:
        sampled = sampled.iloc[:max_rows].copy()
    return sampled


def load_split(
    split: str,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    stations: Iterable[str] | None = None,
    max_rows_per_station: int = 0,
    seed: int = 20260722,
) -> pd.DataFrame:
    """Load one time split without mixing station boundaries or fitting any transforms."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train, val, or test, got {split!r}")

    frames = []
    usecols = list(REQUIRED_COLUMNS)
    optional = {"fault_type", "source"}
    for path_index, path in enumerate(discover_feature_files(feature_dir, stations)):
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = REQUIRED_COLUMNS - header
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        station = path.stem.removeprefix("features_v2_")
        df = pd.read_csv(
            path,
            usecols=usecols + sorted(optional & header),
            parse_dates=["timestamp"],
        )
        df = df.loc[df["split"] == split].copy()
        df = df.dropna(subset=list(FEATURE_COLUMNS) + ["timestamp", "label"])
        unknown = set(df["label"].unique()) - set(LABELS)
        if unknown:
            raise ValueError(f"{path} has unsupported labels: {sorted(unknown)}")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = _context_sample(df, max_rows_per_station, seed + path_index)
        df["station"] = station
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["label_index"] = out["label"].map(LABEL_TO_INDEX).astype(np.int8)
    station_change = out["station"].ne(out["station"].shift())
    gap = out["timestamp"].diff().gt(pd.Timedelta(minutes=10))
    out["sequence_reset"] = (station_change | gap).fillna(True)
    return out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
