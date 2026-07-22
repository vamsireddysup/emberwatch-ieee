"""
Dataset v2 generator for EmberWatch: physics-grounded synthetic device thermal data.

Why this exists: no public dataset pairs real equipment temperature with real ambient
weather and labeled faults. This script simulates a transformer-like device thermally
driven by REAL ambient data from our 5 Oregon/California stations (loaded via the
existing loaders in src/ingest.py), with every tunable grounded in real measurements
from the ETT transformer dataset (via data/processed/ett_calibration.json, produced by
src/calibrate_ett.py).

Physics: lumped-parameter RC thermal model (Newton's law of cooling with internal
heating), the standard first-order model for transformer thermal behavior:

    T_dev(t+dt) = T_dev(t) + (dt / (tau * tau_mult)) *
                  (T_amb + K_LOAD * k_mult * load^2 + K_SOLAR * solar + H_fault - T_dev(t))

Constants (engineering ranges for small distribution transformers, documented inline):
    tau     = 180 min thermal time constant
    K_LOAD  = 28  -> full load (1.0) gives ~28 degC steady-state rise over ambient
    K_SOLAR = 5   -> full sun adds up to ~5 degC (enclosure solar gain)

Inputs per station:
    ambient  - real station data, 5-min grid, forward-fill capped at 1h; longer gaps
               split the simulation into independent segments (state resets; no
               fabricated bridging). Segments shorter than 24h are dropped.
    load     - real daily load-curve shapes sampled from ETT (calibration file),
               scaled into [0.45, 0.75], weekend factor 0.85, small AR(1) jitter
    solar    - Klamath/Gerber: the real measured Solar Rad. column from the same .xls;
               ASOS stations: clear-sky curve (45N seasonal) x slow AR(1) cloudiness

Fault injection (Poisson-scheduled, ~1 event per --mean-gap-days per station,
non-overlapping; each modifies the physics, not the output):
    loose_connection    - contact heating ramps 0->8-25 degC over 0.5-3 days, then repair
    overload            - load x1.5-2.5 for 2-12 h
    cooling_degradation - tau and K_LOAD effectively worsen over 1.5-4 days (blocked
                          ventilation), then repair
    thermal_runaway     - slow precursor (0->2-4 degC over 1-3 days) then accelerating
                          exponential burst (doubling ~45 min, capped +60 degC), 2-5 h,
                          then trip. The fire-risk case.

Labeling by counterfactual twin: each segment is simulated twice with identical inputs
-- a healthy twin (no faults) and a faulted twin. excess = T_faulted - T_healthy is the
ground-truth abnormal heating. Labels use an alarm-style hysteresis latch:
    Anomaly  - latched when excess >= 6 degC, OR excess >= 2 degC and rising
               >= 0.3 degC/min; released when excess < 2 degC
    Warming  - excess >= 2 degC and not falling (and not latched Anomaly)
    Normal   - otherwise
So every event has a physically real lead-up (Normal -> Warming -> Anomaly) whose length
depends on the fault type, instead of fixed-width label windows.

Sensor model (observed device channel only; ground truth stays clean internally):
    Gaussian noise (std from ETT calibration, ~0.29 degC) + 0.1 degC quantization
    (NTC + ADC), rare stuck-sensor episodes (~4/yr, 15-45 min, holds last value) and
    dropouts (~6/yr, 5-15 min, NaN) -- both labeled per ground truth, since the
    underlying state doesn't change just because the sensor glitched.

Outputs (nothing in v1 is touched):
    data/synthetic_v2/device_thermal_<station>.csv with columns:
        timestamp       - 5-min UTC
        device_temp_c   - sensor-observed simulated device temperature
        ambient_temp_c  - real station ambient
        load_factor     - ground-truth load 0-1 (aux/debug only; NOT a model input --
                          the real node has only two thermistors)
        fault_type      - none | loose_connection | overload | cooling_degradation |
                          thermal_runaway
        event_id        - 0 when none; unique within station; joins events_catalog.csv
        label           - Normal | Warming | Anomaly
        split           - train (2020-21) | val (2022) | test (2023), chronological
    data/synthetic_v2/events_catalog.csv (upserted per station): event_id, station,
        fault_type, severity (minor <8 / major 8-20 / critical >=20 peak excess),
        start_ts, first_warming_ts, first_anomaly_ts, end_ts, peak_excess_c

Also prints a realism-validation table comparing the synthetic device channel (Normal
rows) against the real ETT oil-temperature targets from the calibration file.

Usage:
    python src/synthesize_thermal.py [--station all|hood_river|hillsboro|troutdale|klamath|gerber]
                                     [--start 2020-01-01] [--end 2023-12-31]
                                     [--seed 42] [--mean-gap-days 24]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ingest import (
    ASOS_STATIONS,
    RAW_DIR,
    WRCC_MISSING_SENTINEL,
    WRCC_STATIONS,
    process_asos_station,
    process_wrcc_station,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = REPO_ROOT / "data" / "processed" / "ett_calibration.json"
OUTPUT_DIR = REPO_ROOT / "data" / "synthetic_v2"
CATALOG_PATH = OUTPUT_DIR / "events_catalog.csv"

STEP_MIN = 5
STEPS_PER_DAY = 288
FFILL_LIMIT = 12          # 1 hour at 5-min steps
MIN_SEGMENT_STEPS = 72    # 6 h; shorter fragments between data gaps are discarded

TAU_MIN = 180.0
K_LOAD = 28.0
K_SOLAR = 5.0
QUANT_STEP = 0.1
# Fraction of the ETT-measured residual applied as sensor noise: the measured residual
# includes the real transformer's own micro-dynamics, which our simulation already
# produces (load jitter, ambient fluctuation), so adding the full value double-counts.
# 0.7 was calibrated so the synthetic total residual lands on the ETT target.
SENSOR_NOISE_FRACTION = 0.7

WARMING_EXCESS = 2.0
ANOMALY_EXCESS = 6.0
ANOMALY_RATE_PER_MIN = 0.3

FAULT_TYPES = ["loose_connection", "overload", "cooling_degradation", "thermal_runaway"]
FAULT_PROBS = [0.35, 0.30, 0.20, 0.15]
EVENT_BUFFER_STEPS = 2 * STEPS_PER_DAY  # quiet time enforced after each event

WRCC_SOLAR_COL_INDEX = 11  # 'Solar Rad.' W/m2, last column in both RAWS exports


def station_rng(seed: int, station: str) -> np.random.Generator:
    return np.random.default_rng([seed, int.from_bytes(station.encode(), "little") % (2**31)])


def load_ambient(station: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if station in ASOS_STATIONS:
        series = process_asos_station(RAW_DIR / ASOS_STATIONS[station], station, ffill_limit=FFILL_LIMIT)
    else:
        series = process_wrcc_station(RAW_DIR / WRCC_STATIONS[station], station, ffill_limit=FFILL_LIMIT)
    return series.loc[start:end]


def load_real_solar(station: str, index: pd.DatetimeIndex) -> np.ndarray:
    """Klamath/Gerber: real measured solar radiation from the same WRCC .xls, 0-1."""
    df = pd.read_csv(RAW_DIR / WRCC_STATIONS[station], sep="\t", skiprows=4, header=None)
    timestamp = pd.to_datetime(df[0].astype(str), format="%y%m%d%H%M")
    timestamp = timestamp.dt.tz_localize("Etc/GMT+8").dt.tz_convert("UTC")
    solar = pd.to_numeric(df[WRCC_SOLAR_COL_INDEX], errors="coerce").replace(WRCC_MISSING_SENTINEL, np.nan)
    series = pd.DataFrame({"t": timestamp, "s": solar}).dropna().groupby("t")["s"].mean()
    series = series.resample("5min").ffill(limit=FFILL_LIMIT)
    aligned = series.reindex(index).ffill().fillna(0.0)
    return np.clip(aligned.to_numpy() / 1000.0, 0.0, 1.0)


def synthetic_solar(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """ASOS stations: clear-sky curve for ~45N with a slow AR(1) daily cloudiness walk."""
    hours_local = (index.hour + index.minute / 60.0 - 8.0) % 24  # fixed UTC-8
    elevation = np.sin(np.pi * (hours_local - 6.0) / 12.0)
    elevation = np.where((hours_local > 6) & (hours_local < 18), np.clip(elevation, 0, 1), 0.0)
    doy = index.dayofyear.to_numpy()
    seasonal = 0.55 + 0.45 * np.cos(2 * np.pi * (doy - 172) / 365.0)

    days = index.normalize()
    unique_days, day_inverse = np.unique(days, return_inverse=True)
    cloud = np.empty(len(unique_days))
    c = 0.65
    for i in range(len(unique_days)):
        c = np.clip(0.65 + 0.8 * (c - 0.65) + rng.normal(0, 0.15), 0.15, 1.0)
        cloud[i] = c
    return elevation * seasonal * cloud[day_inverse]


def build_load(index: pd.DatetimeIndex, rng: np.random.Generator, shapes: list) -> np.ndarray:
    """Real ETT daily load shapes, scaled to [0.35, 0.80], weekend factor, AR(1) jitter."""
    shape_bank = [np.interp(np.linspace(0, 95, STEPS_PER_DAY), np.arange(96), np.asarray(s)) for s in shapes]
    days = index.normalize()
    unique_days, day_inverse = np.unique(days, return_inverse=True)

    load = np.empty(len(index))
    step_of_day = ((index.hour * 60 + index.minute) // STEP_MIN).to_numpy()
    for i, day in enumerate(unique_days):
        mask = day_inverse == i
        shape = shape_bank[rng.integers(len(shape_bank))]
        weekend = 0.85 if pd.Timestamp(day).dayofweek >= 5 else 1.0
        load[mask] = (0.45 + 0.30 * shape[step_of_day[mask]]) * weekend

    jitter = np.empty(len(index))
    j = 0.0
    noise = rng.normal(0, 0.01, len(index))
    for i in range(len(index)):
        j = 0.9 * j + noise[i]
        jitter[i] = j
    return np.clip(load + jitter, 0.05, 1.3)


def make_envelope(fault_type: str, rng: np.random.Generator) -> dict:
    """Per-step physics-modifier arrays for one fault event."""
    if fault_type == "loose_connection":
        ramp = int(rng.uniform(0.5, 3.0) * STEPS_PER_DAY)
        peak = rng.uniform(8, 25)
        h = peak * (np.arange(1, ramp + 1) / ramp) ** 1.5
        return {"h_add": h}
    if fault_type == "overload":
        dur = int(rng.uniform(2, 12) * 60 / STEP_MIN)
        mult = rng.uniform(1.5, 2.5)
        return {"load_mult": np.full(dur, mult)}
    if fault_type == "cooling_degradation":
        dur = int(rng.uniform(1.5, 4.0) * STEPS_PER_DAY)
        ramp = np.arange(1, dur + 1) / dur
        return {
            "tau_mult": 1 + ramp * (rng.uniform(1.5, 2.5) - 1),
            "k_mult": 1 + ramp * (rng.uniform(1.3, 1.8) - 1),
        }
    # thermal_runaway: slow precursor, then accelerating exponential burst, then trip.
    pre = int(rng.uniform(1, 3) * STEPS_PER_DAY)
    pre_peak = rng.uniform(2, 4)
    burst = int(rng.uniform(2, 5) * 60 / STEP_MIN)
    h_pre = pre_peak * np.arange(1, pre + 1) / pre
    h_burst = np.clip(pre_peak * 2 ** (np.arange(1, burst + 1) * STEP_MIN / 45.0), 0, 60.0)
    return {"h_add": np.concatenate([h_pre, h_burst])}


def schedule_events(n_steps: int, rng: np.random.Generator, mean_gap_days: float, first_event_id: int):
    """Poisson-gap scheduling of non-overlapping fault events across one segment.
    Returns per-step arrays (h_add, load_mult, tau_mult, k_mult, event_id, fault_idx)
    and the list of scheduled events."""
    h_add = np.zeros(n_steps)
    load_mult = np.ones(n_steps)
    tau_mult = np.ones(n_steps)
    k_mult = np.ones(n_steps)
    event_id = np.zeros(n_steps, dtype=int)
    fault_idx = np.full(n_steps, -1, dtype=int)

    events = []
    eid = first_event_id
    mean_gap_steps = mean_gap_days * STEPS_PER_DAY
    cursor = int(rng.exponential(mean_gap_steps))
    while cursor < n_steps:
        f_idx = rng.choice(len(FAULT_TYPES), p=FAULT_PROBS)
        fault_type = FAULT_TYPES[f_idx]
        env = make_envelope(fault_type, rng)
        length = len(next(iter(env.values())))
        end = min(cursor + length, n_steps)
        span = slice(cursor, end)
        cut = end - cursor

        if "h_add" in env:
            h_add[span] = env["h_add"][:cut]
        if "load_mult" in env:
            load_mult[span] = env["load_mult"][:cut]
        if "tau_mult" in env:
            tau_mult[span] = env["tau_mult"][:cut]
            k_mult[span] = env["k_mult"][:cut]
        event_id[span] = eid
        fault_idx[span] = f_idx
        events.append({"event_id": eid, "fault_type": fault_type, "start_idx": cursor, "env_end_idx": end})
        eid += 1
        cursor = end + EVENT_BUFFER_STEPS + int(rng.exponential(mean_gap_steps))
    return h_add, load_mult, tau_mult, k_mult, event_id, fault_idx, events


def integrate_twins(ambient, load, solar, h_add, load_mult, tau_mult, k_mult):
    """Sequential RC integration of the healthy and faulted twins over one segment."""
    n = len(ambient)
    t_healthy = np.empty(n)
    t_faulted = np.empty(n)
    t_h = ambient[0] + K_LOAD * load[0] ** 2 + K_SOLAR * solar[0]
    t_f = t_h
    for i in range(n):
        eq_h = ambient[i] + K_LOAD * load[i] ** 2 + K_SOLAR * solar[i]
        t_h += (STEP_MIN / TAU_MIN) * (eq_h - t_h)
        eq_f = (
            ambient[i]
            + K_LOAD * k_mult[i] * (load[i] * load_mult[i]) ** 2
            + K_SOLAR * solar[i]
            + h_add[i]
        )
        t_f += (STEP_MIN / (TAU_MIN * tau_mult[i])) * (eq_f - t_f)
        t_healthy[i] = t_h
        t_faulted[i] = t_f
    return t_healthy, t_faulted


def label_from_excess(excess: np.ndarray) -> np.ndarray:
    """Alarm-style hysteresis labeling from ground-truth excess heating."""
    labels = np.full(len(excess), "Normal", dtype=object)
    rate = np.diff(excess, prepend=excess[0]) / STEP_MIN
    latched = False
    for i in range(len(excess)):
        # Rate trigger requires excess to already be meaningful, so a fast-but-tiny blip
        # can't latch a one-step Anomaly.
        if not latched and (
            excess[i] >= ANOMALY_EXCESS
            or (rate[i] >= ANOMALY_RATE_PER_MIN and excess[i] >= WARMING_EXCESS)
        ):
            latched = True
        elif latched and excess[i] < WARMING_EXCESS:
            latched = False
        if latched:
            labels[i] = "Anomaly"
        elif excess[i] >= WARMING_EXCESS and rate[i] >= 0:
            labels[i] = "Warming"
    return labels


def apply_sensor_model(true_temp: np.ndarray, n_years: float, rng: np.random.Generator) -> np.ndarray:
    obs = true_temp + rng.normal(0, SENSOR_NOISE_STD, len(true_temp))
    for _ in range(int(round(4 * n_years))):  # stuck-sensor episodes
        start = rng.integers(0, max(1, len(obs) - 1))
        dur = int(rng.uniform(15, 45) / STEP_MIN)
        obs[start : start + dur] = obs[start]
    for _ in range(int(round(6 * n_years))):  # dropouts
        start = rng.integers(0, max(1, len(obs) - 1))
        dur = int(rng.uniform(5, 15) / STEP_MIN)
        obs[start : start + dur] = np.nan
    return np.round(obs / QUANT_STEP) * QUANT_STEP


def extend_event_tails(event_id, fault_idx, excess, events):
    """Attribute post-envelope cooldown rows (excess still >= WARMING_EXCESS) to their event."""
    cap = 2 * STEPS_PER_DAY
    for ev in events:
        i = ev["env_end_idx"]
        end = i
        while end < len(excess) and end - i < cap and excess[end] >= WARMING_EXCESS and event_id[end] == 0:
            end += 1
        if end > i:
            span = slice(i, end)
            event_id[span] = ev["event_id"]
            fault_idx[span] = FAULT_TYPES.index(ev["fault_type"])
        ev["end_idx"] = end
    return event_id, fault_idx, events


def synthesize_station(station: str, start: pd.Timestamp, end: pd.Timestamp, seed: int, mean_gap_days: float, shapes: list):
    rng = station_rng(seed, station)
    print(f"\n=== {station} ===")
    ambient = load_ambient(station, start, end)
    valid = ambient.notna()
    print(f"  ambient rows on 5-min grid: {len(ambient)}, valid: {valid.sum()}")

    solar_full = (
        load_real_solar(station, ambient.index)
        if station in WRCC_STATIONS
        else synthetic_solar(ambient.index, rng)
    )
    load_full = build_load(ambient.index, rng, shapes)

    # Contiguous valid-ambient runs become independent simulation segments.
    run_id = (valid != valid.shift()).cumsum()
    frames, all_events = [], []
    next_eid, dropped = 1, 0
    for _, run in ambient.groupby(run_id):
        if not run.notna().all():
            continue
        if len(run) < MIN_SEGMENT_STEPS:
            dropped += len(run)
            continue
        pos = ambient.index.get_indexer(run.index)
        amb = run.to_numpy(dtype=float)
        load = load_full[pos]
        solar = solar_full[pos]

        h_add, load_mult, tau_mult, k_mult, event_id, fault_idx, events = schedule_events(
            len(run), rng, mean_gap_days, next_eid
        )
        t_healthy, t_faulted = integrate_twins(amb, load, solar, h_add, load_mult, tau_mult, k_mult)
        excess = t_faulted - t_healthy
        labels = label_from_excess(excess)
        event_id, fault_idx, events = extend_event_tails(event_id, fault_idx, excess, events)

        n_years = len(run) / (STEPS_PER_DAY * 365.0)
        observed = apply_sensor_model(t_faulted, n_years, rng)

        for ev in events:
            seg_excess = excess
            ev_mask = event_id == ev["event_id"]
            ev_labels = labels[ev_mask]
            ev_times = run.index[ev_mask]
            warming_ts = ev_times[ev_labels == "Warming"]
            anomaly_ts = ev_times[ev_labels == "Anomaly"]
            all_events.append(
                {
                    "event_id": ev["event_id"],
                    "station": station,
                    "fault_type": ev["fault_type"],
                    "severity": None,  # filled below from peak excess
                    "start_ts": run.index[ev["start_idx"]],
                    "first_warming_ts": warming_ts.min() if len(warming_ts) else pd.NaT,
                    "first_anomaly_ts": anomaly_ts.min() if len(anomaly_ts) else pd.NaT,
                    "end_ts": run.index[min(ev["end_idx"], len(run) - 1)],
                    "peak_excess_c": round(float(seg_excess[ev_mask].max()), 2) if ev_mask.any() else 0.0,
                }
            )
        next_eid += len(events)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": run.index,
                    "device_temp_c": observed,
                    "ambient_temp_c": amb,
                    "load_factor": np.round(load, 4),
                    "fault_type": np.where(fault_idx >= 0, np.array(FAULT_TYPES, dtype=object)[fault_idx], "none"),
                    "event_id": event_id,
                    "label": labels,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    year = pd.DatetimeIndex(df["timestamp"]).year
    df["split"] = np.select([year <= 2021, year == 2022], ["train", "val"], default="test")

    for ev in all_events:
        peak = ev["peak_excess_c"]
        ev["severity"] = "critical" if peak >= 20 else ("major" if peak >= 8 else "minor")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"device_thermal_{station}.csv"
    df.to_csv(out_path, index=False)

    counts = df["label"].value_counts()
    non_normal = 1 - counts.get("Normal", 0) / len(df)
    print(f"  wrote {out_path} ({len(df)} rows, dropped {dropped} rows in short segments)")
    print(f"  events: {len(all_events)}; labels: {counts.to_dict()} (non-Normal {non_normal:.2%})")
    print(f"  splits: {df['split'].value_counts().to_dict()}")
    return df, all_events


def validate(df: pd.DataFrame, calibration: dict, station: str) -> None:
    """Compare synthetic Normal-operation statistics against real ETT targets."""
    normal = df[(df["label"] == "Normal") & df["device_temp_c"].notna()]
    series = normal.set_index("timestamp")["device_temp_c"]
    residual = series - series.rolling(13, center=True).mean()
    daily = series.resample("1D").agg(["min", "max", "count"])
    complete = daily[daily["count"] >= STEPS_PER_DAY * 0.95]
    amplitude = complete["max"] - complete["min"]
    full = df.set_index("timestamp")["device_temp_c"]

    print(f"  realism vs real ETT oil temperature ({station}):")
    print(f"    {'metric':<28}{'synthetic':>12}{'ETT target':>12}")
    print(f"    {'noise std (degC)':<28}{residual.std():>12.3f}{calibration['sensor_noise_std']:>12.3f}")
    print(
        f"    {'diurnal amplitude (degC)':<28}{amplitude.mean():>12.2f}"
        f"{calibration['diurnal_amplitude_mean']:>12.2f}"
    )
    for name, lag in [("autocorr 1h", 12), ("autocorr 6h", 72), ("autocorr 24h", 288)]:
        print(f"    {name:<28}{full.autocorr(lag=lag):>12.4f}{calibration['ot_autocorr'][name.split()[1]]:>12.4f}")


def upsert_catalog(events: list) -> None:
    new = pd.DataFrame(events)
    if CATALOG_PATH.exists():
        existing = pd.read_csv(CATALOG_PATH)
        existing = existing[~existing["station"].isin(new["station"].unique())]
        if len(existing):
            new = pd.concat([existing, new], ignore_index=True)
    new.to_csv(CATALOG_PATH, index=False)
    print(f"\nCatalog: {CATALOG_PATH} ({len(new)} events total)")


def main():
    parser = argparse.ArgumentParser(description="EmberWatch dataset v2 thermal synthesizer")
    parser.add_argument("--station", default="all", choices=["all"] + list(ASOS_STATIONS) + list(WRCC_STATIONS))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mean-gap-days", type=float, default=15.0)
    args = parser.parse_args()

    calibration = json.loads(CALIBRATION_PATH.read_text())
    global SENSOR_NOISE_STD
    SENSOR_NOISE_STD = SENSOR_NOISE_FRACTION * calibration["sensor_noise_std"]

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)

    stations = list(ASOS_STATIONS) + list(WRCC_STATIONS) if args.station == "all" else [args.station]
    all_events = []
    for station in stations:
        df, events = synthesize_station(
            station, start, end, args.seed, args.mean_gap_days, calibration["daily_load_shapes"]
        )
        validate(df, calibration, station)
        all_events.extend(events)
    upsert_catalog(all_events)


if __name__ == "__main__":
    main()
