"""Deterministic physics-informed thermal/fault dataset generator.

This is a documented replacement for the untracked generator that created the existing
local synthetic-v2 artifacts. It does not claim to reproduce those files byte-for-byte.
By default it writes under artifacts/ and refuses to overwrite files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AMBIENT_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "synthetic_v2"
CALIBRATION_PATH = REPO_ROOT / "data" / "processed" / "ett_calibration.json"

FAULT_TYPES = ("overload", "loose_connection", "cooling_degradation", "thermal_runaway")
SEVERITIES = ("minor", "major", "critical")


@dataclass(frozen=True)
class Event:
    event_id: int
    station: str
    fault_type: str
    severity: str
    start_index: int
    duration_samples: int


def load_noise_std(path: Path = CALIBRATION_PATH) -> float:
    if not path.exists():
        return 0.30
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload.get("sensor_noise_std", 0.30))


def load_ambient(path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "ambient_temp_c"], parse_dates=["timestamp"])
    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]
    if df.empty:
        raise ValueError(f"No ambient samples in {path} between {start} and {end}")
    regular = df.resample("5min").mean().interpolate(limit=12).ffill().bfill()
    return regular.reset_index()


def simulate_nominal(ambient: np.ndarray, rng: np.random.Generator, noise_std: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(ambient)
    steps = np.arange(n)
    daily_phase = 2 * np.pi * (steps % 288) / 288
    weekly_phase = 2 * np.pi * (steps % (288 * 7)) / (288 * 7)
    stochastic = np.empty(n, dtype=np.float32)
    stochastic[0] = rng.normal(0, 0.04)
    for index in range(1, n):
        stochastic[index] = 0.97 * stochastic[index - 1] + rng.normal(0, 0.025)
    load = 0.58 + 0.16 * np.sin(daily_phase - 0.7) + 0.06 * np.sin(weekly_phase) + stochastic
    load = np.clip(load, 0.20, 1.05).astype(np.float32)

    target = ambient + 7.0 + 21.0 * np.square(load)
    asset = np.empty(n, dtype=np.float32)
    asset[0] = target[0]
    alpha = 1.0 - np.exp(-5.0 / 95.0)  # nominal 95-minute thermal time constant
    for index in range(1, n):
        asset[index] = asset[index - 1] + alpha * (target[index] - asset[index - 1])
    asset += rng.normal(0, noise_std, n).astype(np.float32)
    return asset, load


def schedule_events(n_rows: int, station: str, events_per_year: int, rng: np.random.Generator, first_id: int) -> list[Event]:
    samples_per_year = 288 * 365
    count = max(1, round(events_per_year * n_rows / samples_per_year))
    margin = 288
    candidates = np.arange(margin, max(margin + 1, n_rows - margin), 288)
    rng.shuffle(candidates)
    selected = []
    min_spacing = 288 * 3
    for candidate in candidates:
        if all(abs(candidate - previous) >= min_spacing for previous in selected):
            selected.append(int(candidate))
        if len(selected) >= count:
            break
    selected.sort()
    events = []
    duration_ranges = {
        "overload": (24, 96),
        "loose_connection": (72, 420),
        "cooling_degradation": (288, 1152),
        "thermal_runaway": (24, 144),
    }
    severity_probabilities = [0.25, 0.45, 0.30]
    for offset, start in enumerate(selected):
        fault = str(rng.choice(FAULT_TYPES))
        severity = str(rng.choice(SEVERITIES, p=severity_probabilities))
        low, high = duration_ranges[fault]
        duration = int(rng.integers(low, high + 1))
        events.append(Event(first_id + offset, station, fault, severity, start, min(duration, n_rows - start)))
    return events


def fault_excess(event: Event, rng: np.random.Generator) -> np.ndarray:
    n = event.duration_samples
    progress = np.linspace(0.0, 1.0, n, dtype=np.float32)
    amplitude = {"minor": 4.0, "major": 10.0, "critical": 20.0}[event.severity]
    if event.fault_type == "overload":
        shape = 1.0 - np.exp(-6.0 * progress)
    elif event.fault_type == "loose_connection":
        shape = np.power(progress, 1.25) * (1.0 + 0.12 * np.sin(18 * np.pi * progress))
    elif event.fault_type == "cooling_degradation":
        shape = 1.0 - np.exp(-3.0 * progress)
    else:
        shape = np.power(progress, 2.4)
    noise = rng.normal(0.0, 0.03 * amplitude, n).astype(np.float32)
    return np.maximum(0.0, amplitude * shape + noise)


def add_faults(asset: np.ndarray, events: list[Event], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    output = asset.copy()
    labels = np.full(len(asset), "Normal", dtype=object)
    event_ids = np.zeros(len(asset), dtype=np.int32)
    fault_types = np.full(len(asset), "none", dtype=object)
    catalog = []
    for event in events:
        end = event.start_index + event.duration_samples
        excess = fault_excess(event, rng)
        output[event.start_index:end] += excess
        event_ids[event.start_index:end] = event.event_id
        fault_types[event.start_index:end] = event.fault_type
        event_labels = np.where(excess >= 6.0, "Anomaly", np.where(excess >= 2.0, "Warming", "Normal"))
        labels[event.start_index:end] = event_labels
        catalog.append(
            {
                "event_id": event.event_id,
                "station": event.station,
                "fault_type": event.fault_type,
                "severity": event.severity,
                "start_index": event.start_index,
                "end_index": end - 1,
                "first_warming_index": event.start_index + int(np.argmax(excess >= 2.0)) if np.any(excess >= 2.0) else None,
                "first_anomaly_index": event.start_index + int(np.argmax(excess >= 6.0)) if np.any(excess >= 6.0) else None,
                "peak_excess_c": round(float(excess.max()), 3),
            }
        )
    return output, labels, event_ids, fault_types, catalog


def engineer_features(raw: pd.DataFrame, station: str) -> pd.DataFrame:
    asset = raw["device_temp_c"]
    output = pd.DataFrame(
        {
            "timestamp": raw["timestamp"],
            "asset_temp_c": asset,
            "ambient_temp_c": raw["ambient_temp_c"],
            "delta_c": asset - raw["ambient_temp_c"],
            "slope_1min": asset.diff() / 5.0,
            "slope_5min": asset.diff() / 5.0,
            "slope_15min": asset.diff(3) / 15.0,
            "variance_30min": asset.rolling(6, min_periods=2).var(),
            "label": raw["label"],
            "source": f"synthetic_v2_reproducible_{station}",
            "event_id": raw["event_id"],
            "fault_type": raw["fault_type"],
            "split": raw["split"],
        }
    )
    return output


def generate_station(ambient_path: Path, station: str, start: str, end: str, events_per_year: int, seed: int, first_id: int):
    rng = np.random.default_rng(seed)
    ambient_frame = load_ambient(ambient_path, start, end)
    ambient = ambient_frame["ambient_temp_c"].to_numpy(dtype=np.float32)
    nominal, load = simulate_nominal(ambient, rng, load_noise_std())
    events = schedule_events(len(ambient), station, events_per_year, rng, first_id)
    asset, labels, event_ids, fault_types, catalog = add_faults(nominal, events, rng)
    year = ambient_frame["timestamp"].dt.year
    split = np.where(year <= 2021, "train", np.where(year == 2022, "val", "test"))
    raw = pd.DataFrame(
        {
            "timestamp": ambient_frame["timestamp"],
            "device_temp_c": np.round(asset, 3),
            "ambient_temp_c": ambient,
            "load_factor": np.round(load, 4),
            "fault_type": fault_types,
            "event_id": event_ids,
            "label": labels,
            "split": split,
        }
    )
    for row in catalog:
        for key in ("start", "end", "first_warming", "first_anomaly"):
            index = row.pop(f"{key}_index")
            row[f"{key}_ts"] = ambient_frame.iloc[index]["timestamp"] if index is not None else None
    return raw, engineer_features(raw, station), catalog


def _write_csv(frame: pd.DataFrame, path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ambient-dir", type=Path, default=DEFAULT_AMBIENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stations", nargs="*", default=["hood_river", "hillsboro", "troutdale", "klamath", "gerber"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--events-per-year", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    catalog_rows = []
    next_id = 1
    for station_index, station in enumerate(args.stations):
        ambient_path = args.ambient_dir / f"ambient_{station}.csv"
        raw, features, catalog = generate_station(
            ambient_path,
            station,
            args.start,
            args.end,
            args.events_per_year,
            args.seed + station_index,
            next_id,
        )
        _write_csv(raw, args.output_dir / f"device_thermal_{station}.csv", args.force)
        _write_csv(features, args.output_dir / f"features_v2_{station}.csv", args.force)
        catalog_rows.extend(catalog)
        next_id += len(catalog)
        print(f"{station}: {len(raw)} rows, {len(catalog)} events")
    _write_csv(pd.DataFrame(catalog_rows), args.output_dir / "events_catalog.csv", args.force)
    manifest = {
        "generator_version": 1,
        "seed": args.seed,
        "start": args.start,
        "end": args.end,
        "events_per_year": args.events_per_year,
        "stations": args.stations,
        "warning": "Synthetic engineering data; not field performance evidence.",
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --force explicitly")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reproducible dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
