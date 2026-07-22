"""
Dataset v2 calibration for EmberWatch.

Extracts statistical signatures from real transformer temperature data (the ETT dataset,
data/raw/merged_ETT_dataset.csv) so the synthetic device channel produced by
src/synthesize_thermal.py is grounded in measured reality instead of invented numbers.
Only the ETTm1 rows are used (15-minute resolution, one real transformer, July 2016 -
June 2018). OT is the transformer's measured oil temperature; HUFL is its high-useful-load
series.

Writes data/processed/ett_calibration.json with:
    sensor_noise_std        - std (deg C) of OT residual after a centered 1h rolling mean;
                              proxy for sensor/measurement noise on a real thermal channel
    diurnal_amplitude_mean  - mean of daily OT (max - min), deg C
    diurnal_amplitude_std   - std of the same
    ot_autocorr             - {"1h","6h","24h"}: OT autocorrelation at those lags;
                              realism validation targets for the synthetic output
    daily_load_shapes       - list of 96-step (15-min) min-max-normalized daily HUFL
                              curves sampled from real complete days; the simulator draws
                              from these so synthetic load has real intra-day structure

Usage:
    python src/calibrate_ett.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ETT_PATH = REPO_ROOT / "data" / "raw" / "merged_ETT_dataset.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "ett_calibration.json"

STEPS_PER_DAY = 96  # 15-min resolution
N_LOAD_SHAPES = 100
SHAPE_SEED = 42


def main():
    print(f"Loading {ETT_PATH} (ETTm1 rows only)")
    df = pd.read_csv(ETT_PATH, parse_dates=["date"])
    df = df[df["source"] == "ETTm1.csv"].sort_values("date").reset_index(drop=True)
    print(f"  rows: {len(df)}, range: {df['date'].min()} to {df['date'].max()}")

    ot = df.set_index("date")["OT"]

    # Sensor noise proxy: high-frequency residual after a centered 1h (5-sample) mean.
    residual = ot - ot.rolling(5, center=True).mean()
    sensor_noise_std = float(residual.std())

    daily_range = ot.resample("1D").agg(["min", "max", "count"])
    complete = daily_range[daily_range["count"] == STEPS_PER_DAY]
    amplitude = complete["max"] - complete["min"]

    autocorr = {
        "1h": float(ot.autocorr(lag=4)),
        "6h": float(ot.autocorr(lag=24)),
        "24h": float(ot.autocorr(lag=96)),
    }

    # Library of real normalized daily load-curve shapes.
    hufl = df.set_index("date")["HUFL"]
    shapes = []
    for _, day in hufl.groupby(hufl.index.date):
        if len(day) != STEPS_PER_DAY:
            continue
        values = day.to_numpy(dtype=float)
        span = values.max() - values.min()
        if span < 1e-6 or np.isnan(span):
            continue
        shapes.append((values - values.min()) / span)
    rng = np.random.default_rng(SHAPE_SEED)
    picked = rng.choice(len(shapes), size=min(N_LOAD_SHAPES, len(shapes)), replace=False)
    load_shapes = [shapes[i].round(4).tolist() for i in picked]

    calibration = {
        "source": "ETTm1.csv (real transformer oil temperature + load, 15-min, 2016-2018)",
        "sensor_noise_std": round(sensor_noise_std, 4),
        "diurnal_amplitude_mean": round(float(amplitude.mean()), 4),
        "diurnal_amplitude_std": round(float(amplitude.std()), 4),
        "ot_autocorr": {k: round(v, 4) for k, v in autocorr.items()},
        "daily_load_shapes": load_shapes,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(calibration))

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  sensor_noise_std:       {calibration['sensor_noise_std']} degC")
    print(f"  diurnal_amplitude_mean: {calibration['diurnal_amplitude_mean']} degC")
    print(f"  diurnal_amplitude_std:  {calibration['diurnal_amplitude_std']} degC")
    print(f"  ot_autocorr:            {calibration['ot_autocorr']}")
    print(f"  daily_load_shapes:      {len(load_shapes)} real daily curves")


if __name__ == "__main__":
    main()
