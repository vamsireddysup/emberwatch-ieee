"""
Step 1 data ingestion for EmberWatch.

Ensures the three NAB files (machine_temperature_system_failure.csv,
ambient_temperature_system_failure.csv, combined_labels.json) exist in data/raw/,
re-downloading from the numenta/NAB GitHub repo if missing.

Processes five independent ambient weather stations -- each written to its own output
file, none merged together:
    Hood River (ASOS, Ken_Jernstedt_Airfield.csv)      -> data/processed/ambient_hood_river.csv
    Hillsboro  (ASOS, Portland-Hillsboro_Airport.csv)  -> data/processed/ambient_hillsboro.csv
    Troutdale  (ASOS, Portland-Troutdale_Airport.csv)  -> data/processed/ambient_troutdale.csv
    Klamath    (WRCC RAWS export, Lower_Klamath_CA_20_23.xls)   -> data/processed/ambient_klamath.csv
    Gerber     (WRCC RAWS export, Gerber_Reservoir_OR_20_23.xls) -> data/processed/ambient_gerber.csv

ASOS files: 10 comment lines, then a header row, then a units row (not data -- dropped),
then data. Timestamps are ISO8601 with explicit UTC offset (parsed and converted to UTC).
Temperature column is air_temp_set_1 (already Celsius).

WRCC RAWS exports (despite the .xls extension, these are tab-delimited text, not real
Excel): 1 station-name line + 3 header lines, then data. Timestamps are YYMMDDhhmm in
LST (fixed Pacific Standard Time, no DST -- localized as Etc/GMT+8 then converted to
UTC). Air temperature is column index 4 (Deg C) in both files -- verified by inspection,
consistent position despite differing trailing columns between stations. -9999 is WRCC's
missing-value sentinel and is treated as NaN.

Every station series is resampled to 5-minute intervals with forward fill, per station,
independently.

Also builds the asset channel: NAB's real machine_temperature_system_failure.csv paired
with Hood River ambient temp only (the primary real-data pairing), written to
    data/processed/asset_nab_with_hood_river.csv
Ambient values are matched to each NAB timestamp by nearest 5-minute-grid neighbor.
label is 'anomaly' for timestamps listed under machine_temperature_system_failure.csv in
NAB's combined_labels.json, 'normal' otherwise. source is fixed to 'real_winter' (NAB's
window falls in Dec-Feb, the only season with real labeled sensor data in this project).

Output columns:
  ambient_<station>.csv:
    timestamp        - 5-minute-resolution UTC timestamp
    ambient_temp_c    - forward-filled station air temperature, Celsius
    station           - station name string (e.g. 'hood_river')

  asset_nab_with_hood_river.csv:
    timestamp        - NAB's native ~5-minute timestamp (naive, as published by NAB)
    asset_temp_c      - NAB machine_temperature_system_failure.csv value (raw NAB units)
    ambient_temp_c    - nearest Hood River ambient reading, Celsius
    delta_c           - asset_temp_c - ambient_temp_c
    label             - 'normal' or 'anomaly' per NAB combined_labels.json
    source            - fixed 'real_winter'

Usage:
    python src/ingest.py
"""
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
NAB_FILES = {
    "machine_temperature_system_failure.csv": f"{NAB_BASE}/data/realKnownCause/machine_temperature_system_failure.csv",
    "ambient_temperature_system_failure.csv": f"{NAB_BASE}/data/realKnownCause/ambient_temperature_system_failure.csv",
}
NAB_LABELS_URL = f"{NAB_BASE}/labels/combined_labels.json"
NAB_LABELS_KEY = "realKnownCause/machine_temperature_system_failure.csv"

ASOS_STATIONS = {
    "hood_river": "Ken_Jernstedt_Airfield.csv",
    "hillsboro": "Portland-Hillsboro_Airport.csv",
    "troutdale": "Portland-Troutdale_Airport.csv",
}
WRCC_STATIONS = {
    "klamath": "Lower_Klamath_CA_20_23.xls",
    "gerber": "Gerber_Reservoir_OR_20_23.xls",
}
WRCC_TEMP_COL_INDEX = 4  # Av Air Temp (Deg C), verified positionally for both files
WRCC_MISSING_SENTINEL = -9999


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already have {dest.name}, skipping download")
        return
    print(f"  downloading {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  saved {dest}")


def fetch_nab_data() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in NAB_FILES.items():
        download_file(url, RAW_DIR / filename)
    download_file(NAB_LABELS_URL, RAW_DIR / "combined_labels.json")


def process_asos_station(path: Path, station: str) -> pd.Series:
    df = pd.read_csv(path, skiprows=10)
    df = df.iloc[1:].reset_index(drop=True)  # drop the units row
    df = df[["Date_Time", "air_temp_set_1"]].rename(
        columns={"Date_Time": "timestamp", "air_temp_set_1": "ambient_temp_c"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ambient_temp_c"] = pd.to_numeric(df["ambient_temp_c"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    series = df.set_index("timestamp")["ambient_temp_c"].resample("5min").ffill()
    return series


def process_wrcc_station(path: Path, station: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t", skiprows=4, header=None)
    raw_timestamp = df[0].astype(str)
    temp = pd.to_numeric(df[WRCC_TEMP_COL_INDEX], errors="coerce")
    temp = temp.replace(WRCC_MISSING_SENTINEL, pd.NA)

    timestamp = pd.to_datetime(raw_timestamp, format="%y%m%d%H%M")
    timestamp = timestamp.dt.tz_localize("Etc/GMT+8").dt.tz_convert("UTC")

    out = pd.DataFrame({"timestamp": timestamp, "ambient_temp_c": temp})
    out = out.dropna(subset=["timestamp", "ambient_temp_c"]).sort_values("timestamp")

    series = out.set_index("timestamp")["ambient_temp_c"].resample("5min").ffill()
    return series


def write_ambient_output(series: pd.Series, station: str) -> Path:
    df = series.rename("ambient_temp_c").reset_index()
    df["station"] = station
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"ambient_{station}.csv"
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()})")
    return out_path


def process_ambient_stations() -> pd.Series:
    """Processes all 5 stations, writes each output file, and returns the Hood River
    series (needed separately below to build the asset channel)."""
    hood_river_series = None

    for station, filename in ASOS_STATIONS.items():
        path = RAW_DIR / filename
        if not path.exists():
            print(f"  ERROR: {path} not found, skipping {station}", file=sys.stderr)
            continue
        print(f"Processing ASOS station: {station}")
        series = process_asos_station(path, station)
        write_ambient_output(series, station)
        if station == "hood_river":
            hood_river_series = series

    for station, filename in WRCC_STATIONS.items():
        path = RAW_DIR / filename
        if not path.exists():
            print(f"  ERROR: {path} not found, skipping {station}", file=sys.stderr)
            continue
        print(f"Processing WRCC station: {station}")
        series = process_wrcc_station(path, station)
        write_ambient_output(series, station)

    if hood_river_series is None:
        print("ERROR: Hood River station failed to process; cannot build asset channel.", file=sys.stderr)
        sys.exit(1)

    return hood_river_series


def load_nab_anomaly_timestamps() -> set:
    import json

    labels_path = RAW_DIR / "combined_labels.json"
    with open(labels_path) as f:
        labels = json.load(f)
    timestamps = labels.get(NAB_LABELS_KEY, [])
    return {pd.Timestamp(t) for t in timestamps}


def build_asset_channel(hood_river_series: pd.Series) -> None:
    print("Building asset channel: NAB machine temp + Hood River ambient")
    nab_df = pd.read_csv(RAW_DIR / "machine_temperature_system_failure.csv")
    nab_df = nab_df.rename(columns={"value": "asset_temp_c"})
    nab_df["timestamp"] = pd.to_datetime(nab_df["timestamp"])
    nab_df = nab_df.sort_values("timestamp")

    # Hood River series is UTC-tz-aware; NAB timestamps are naive as published.
    # Drop tz info for this nearest-match merge rather than assume a specific NAB timezone.
    hood_river_naive = hood_river_series.copy()
    hood_river_naive.index = hood_river_naive.index.tz_localize(None)
    ambient_df = hood_river_naive.rename("ambient_temp_c").reset_index()

    merged = pd.merge_asof(
        nab_df, ambient_df, on="timestamp", direction="nearest", tolerance=pd.Timedelta("10min")
    )
    merged["delta_c"] = merged["asset_temp_c"] - merged["ambient_temp_c"]

    anomaly_timestamps = load_nab_anomaly_timestamps()
    merged["label"] = merged["timestamp"].isin(anomaly_timestamps).map({True: "anomaly", False: "normal"})
    merged["source"] = "real_winter"

    out_path = PROCESSED_DIR / "asset_nab_with_hood_river.csv"
    merged.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(merged)} rows)")
    print(f"  label counts: {merged['label'].value_counts().to_dict()}")
    print(f"  null ambient_temp_c: {merged['ambient_temp_c'].isna().sum()}")


def main():
    print("Fetching NAB datasets...")
    fetch_nab_data()

    print("\nProcessing ambient stations...")
    hood_river_series = process_ambient_stations()

    print()
    build_asset_channel(hood_river_series)


if __name__ == "__main__":
    main()
