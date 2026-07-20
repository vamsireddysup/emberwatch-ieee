"""
Step 3 baseline detectors for EmberWatch.

Input: data/processed/features_nab.csv (from src/features.py).
Ground truth "alert" = label is Warming or Anomaly (as opposed to Normal) -- this is the
same binary framing the on-device model actually acts on: wake the radio or stay quiet.

Two detectors, both calibrated the same way: compute a metric's mean and standard
deviation over Normal-only rows, then set the alert threshold at mean + 3*std. This is a
standard, simple anomaly-detection baseline (fixed z-score cutoff) and keeps both
detectors comparable -- neither one is hand-tuned differently from the other.

  Fixed threshold detector:
    metric = delta_c (asset_temp_c - ambient_temp_c)
    two-sided: alert if delta_c is more than 3 std from the Normal-only mean in either
    direction. NAB's real machine sensor and Hood River's real ambient weather are two
    independent signals with no designed physical relationship, and in this dataset
    delta_c actually falls during the labeled failure windows rather than rising, so a
    one-sided (rise-only) threshold catches nothing (recall 0) -- two-sided is what
    actually gives this detector a chance on this data.

  Moving average detector:
    metric = asset_temp_c minus its own trailing 30-minute rolling mean (deviation from
    local trend, using real timestamps via pandas' time-based rolling window)
    two-sided: alert if the deviation is more than 3 std from the Normal-only mean
    deviation, in either direction, for the same reason as above.

Both detectors are scored through the shared harness in src/evaluate.py (score_predictions),
which prints accuracy/precision/recall/false_positive_rate and records each result as a row
in data/processed/evaluation_results.csv, so they show up alongside any other model (e.g.
Amogh's ESN) scored through that same harness.

Rows with a null slope/variance (the first few rows of the series, before enough history
exists) are dropped before evaluation, since those are feature-computation artifacts, not
real detector failures.

Usage:
    python src/baselines.py
"""
from pathlib import Path

import pandas as pd

from evaluate import score_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features_nab.csv"

Z_SCORE = 3


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.dropna(subset=["slope_1min", "slope_5min", "slope_15min", "variance_30min"])
    return df.sort_values("timestamp").reset_index(drop=True)


def fixed_threshold_detector(df: pd.DataFrame) -> tuple:
    normal = df.loc[df["label"] == "Normal", "delta_c"]
    mean, std = normal.mean(), normal.std()
    lower, upper = mean - Z_SCORE * std, mean + Z_SCORE * std
    pred = (df["delta_c"] < lower) | (df["delta_c"] > upper)
    return pred, (lower, upper)


def moving_average_detector(df: pd.DataFrame) -> tuple:
    rolling_mean = df.set_index("timestamp")["asset_temp_c"].rolling("30min").mean()
    deviation = df["asset_temp_c"].to_numpy() - rolling_mean.to_numpy()

    normal_deviation = deviation[df["label"].to_numpy() == "Normal"]
    mean, std = normal_deviation.mean(), normal_deviation.std()
    lower, upper = mean - Z_SCORE * std, mean + Z_SCORE * std
    pred = (deviation < lower) | (deviation > upper)
    return pred, (lower, upper)


def main():
    print(f"Loading {FEATURES_PATH}")
    df = load_features(FEATURES_PATH)
    print(f"  rows: {len(df)}")

    y_true = df["label"].isin(["Warming", "Anomaly"])
    print(f"  alert rows (Warming or Anomaly): {y_true.sum()} / {len(df)}")

    print("\nFixed threshold detector (delta_c, two-sided)")
    pred_fixed, (lower_fixed, upper_fixed) = fixed_threshold_detector(df)
    print(f"  bounds: [{lower_fixed:.4f}, {upper_fixed:.4f}] degrees C")
    score_predictions("fixed_threshold", y_true, pred_fixed)

    print("\nMoving average detector (asset_temp_c deviation from 30min rolling mean, two-sided)")
    pred_ma, (lower_ma, upper_ma) = moving_average_detector(df)
    print(f"  bounds: [{lower_ma:.4f}, {upper_ma:.4f}] degrees C")
    score_predictions("moving_average", y_true, pred_ma)


if __name__ == "__main__":
    main()
