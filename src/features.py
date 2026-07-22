"""
Step 2 feature engineering for EmberWatch.

Input: data/processed/asset_nab_with_hood_river.csv (from src/ingest.py), which only has
4 point-in-time anomaly labels out of 22,695 rows -- NAB's combined_labels.json marks single
anomalous timestamps, not the lead-up to them. That's not enough signal to train a 3-class
(Normal/Warming/Anomaly) classifier, so this script first expands each point anomaly into a
window before computing features:
    [T - 60min, T)        -> Warming
    [T, T + 30min]         -> Anomaly
    everything else        -> Normal
If two expanded windows were to overlap, Anomaly wins over Warming. In this dataset the 4
anomalies are days to weeks apart, so no overlap actually occurs, but the logic handles it
either way.

A handful of rows in the input share the same timestamp (duplicate readings); these are
averaged together before feature computation, same fix applied to the ASOS stations in
src/ingest.py.

Feature definitions:
  slope_Xmin  - rate of change of asset_temp_c, in degrees C per minute, computed against
                the most recent prior observation that is at least X minutes older than the
                current row (found via merge_asof on real timestamps), divided by the actual
                elapsed minutes between those two rows -- not by X and not by row count. On
                this dataset's ~5-minute native grid, slope_1min and slope_5min both resolve
                to a 1-row lookback (since any prior row is already >=5min old, which
                satisfies ">=1min old"); slope_15min looks back further, to roughly the row
                3 samples back.
  variance_30min - rolling variance of asset_temp_c over the trailing 30 minutes of real time
                (pandas time-based rolling window, so it is robust to any timestamp gaps).

Output: data/processed/features_nab.csv
Output columns:
    timestamp       - from the input file
    asset_temp_c     - from the input file
    ambient_temp_c   - from the input file
    delta_c          - from the input file
    slope_1min       - degrees C / minute, see above
    slope_5min       - degrees C / minute, see above
    slope_15min      - degrees C / minute, see above
    variance_30min   - rolling variance of asset_temp_c over trailing 30 real minutes
    label            - 'Normal', 'Warming', or 'Anomaly' after window expansion (see above)
    source           - passed through from the input file (fixed 'real_winter')

Usage:
    python src/features.py                       # v1 NAB default, output unchanged
    python src/features.py --input data/synthetic_v2/device_thermal_<station>.csv \
                           --output data/processed/features_v2_<station>.csv \
                           --source-name synthetic_v2_<station>

v2 mode (auto-detected by the device_temp_c column): the input's own physics-derived
labels are passed through untouched (no NAB window expansion), device_temp_c is renamed
asset_temp_c so downstream consumers see one schema, delta_c is computed, and
event_id/fault_type/split are appended after the v1 columns.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "data" / "processed" / "asset_nab_with_hood_river.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "features_nab.csv"

WARMING_LOOKBACK = pd.Timedelta(minutes=60)
ANOMALY_LOOKAHEAD = pd.Timedelta(minutes=30)


def load_asset_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.groupby("timestamp", as_index=False).agg(
        {
            "asset_temp_c": "mean",
            "ambient_temp_c": "mean",
            "delta_c": "mean",
            "label": "first",
            "source": "first",
        }
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def expand_anomaly_windows(df: pd.DataFrame) -> pd.Series:
    seed_timestamps = df.loc[df["label"] == "anomaly", "timestamp"]

    anomaly_mask = pd.Series(False, index=df.index)
    warming_mask = pd.Series(False, index=df.index)
    for t in seed_timestamps:
        anomaly_mask |= (df["timestamp"] >= t) & (df["timestamp"] <= t + ANOMALY_LOOKAHEAD)
        warming_mask |= (df["timestamp"] >= t - WARMING_LOOKBACK) & (df["timestamp"] < t)

    labels = np.where(anomaly_mask, "Anomaly", np.where(warming_mask, "Warming", "Normal"))
    return pd.Series(labels, index=df.index)


def compute_slope(df: pd.DataFrame, column: str, minutes: int) -> pd.Series:
    lookup = pd.DataFrame(
        {
            "orig_idx": df.index,
            "orig_timestamp": df["timestamp"],
            "lookup_time": df["timestamp"] - pd.Timedelta(minutes=minutes),
        }
    ).sort_values("lookup_time")

    source = df[["timestamp", column]].rename(columns={column: "lag_value"}).sort_values("timestamp")

    matched = pd.merge_asof(
        lookup, source, left_on="lookup_time", right_on="timestamp", direction="backward"
    )
    matched = matched.set_index("orig_idx").reindex(df.index)

    elapsed_minutes = (matched["orig_timestamp"] - matched["timestamp"]).dt.total_seconds() / 60
    rate = (df[column].to_numpy() - matched["lag_value"].to_numpy()) / elapsed_minutes.to_numpy()
    return pd.Series(rate, index=df.index)


def compute_variance(df: pd.DataFrame, column: str, minutes: int) -> pd.Series:
    series = df.set_index("timestamp")[column].rolling(f"{minutes}min").var()
    return series.reset_index(drop=True).set_axis(df.index)


def main():
    parser = argparse.ArgumentParser(description="EmberWatch feature engineering")
    parser.add_argument("--input", type=Path, default=INPUT_PATH,
                        help="Input CSV: v1 asset channel (default) or a v2 device_thermal_<station>.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--source-name", default=None,
                        help="Value for the source column (v2 inputs have no source column of their own)")
    args = parser.parse_args()

    print(f"Loading {args.input}")
    # v2 synthetic files carry device_temp_c plus their own physics-derived labels,
    # event_id/fault_type/split; v1 needs dedup + NAB point-label window expansion.
    peek = pd.read_csv(args.input, nrows=1)
    is_v2 = "device_temp_c" in peek.columns

    if is_v2:
        df = pd.read_csv(args.input, parse_dates=["timestamp"])
        df = df.rename(columns={"device_temp_c": "asset_temp_c"})
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["delta_c"] = df["asset_temp_c"] - df["ambient_temp_c"]
        df["source"] = args.source_name if args.source_name else "synthetic_v2"
        print(f"  v2 input: {len(df)} rows, labels passed through unchanged")
    else:
        df = load_asset_data(args.input)
        print(f"  rows after deduping timestamps: {len(df)}")
        print("Expanding anomaly points into Warming/Anomaly windows...")
        df["label"] = expand_anomaly_windows(df)

    print("Computing slopes and variance...")
    df["slope_1min"] = compute_slope(df, "asset_temp_c", 1)
    df["slope_5min"] = compute_slope(df, "asset_temp_c", 5)
    df["slope_15min"] = compute_slope(df, "asset_temp_c", 15)
    df["variance_30min"] = compute_variance(df, "asset_temp_c", 30)

    columns = [
        "timestamp",
        "asset_temp_c",
        "ambient_temp_c",
        "delta_c",
        "slope_1min",
        "slope_5min",
        "slope_15min",
        "variance_30min",
        "label",
        "source",
    ]
    if is_v2:
        columns += ["event_id", "fault_type", "split"]
    out = df[columns]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nWrote {args.output}")
    print(f"  rows: {len(out)}")
    print(f"  date range: {out['timestamp'].min()} to {out['timestamp'].max()}")
    print("  label distribution:")
    counts = out["label"].value_counts()
    for label, count in counts.items():
        print(f"    {label}: {count} ({count / len(out):.2%})")
    print(f"  null counts:\n{out.isna().sum()}")


if __name__ == "__main__":
    main()
