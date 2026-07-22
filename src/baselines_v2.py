"""Leakage-safe multi-station statistical baselines for synthetic-v2 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import binary_alert_metrics, event_metrics
from .ml_data import DEFAULT_FEATURE_DIR, load_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "reports" / "baseline_metrics.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "generated" / "BASELINE_RESULTS.md"
MULTIVARIATE_COLUMNS = ("delta_c", "slope_15min", "variance_30min")


def fit_station_parameters(train) -> dict:
    parameters = {}
    for station, frame in train.groupby("station", sort=False):
        normal = frame.loc[frame["label"] == "Normal"]
        delta_mean = float(normal["delta_c"].mean())
        delta_std = max(float(normal["delta_c"].std()), 1e-6)
        means = normal.loc[:, MULTIVARIATE_COLUMNS].mean().to_numpy(dtype=float)
        stds = normal.loc[:, MULTIVARIATE_COLUMNS].std().to_numpy(dtype=float)
        stds[stds < 1e-6] = 1.0
        scores = ((normal.loc[:, MULTIVARIATE_COLUMNS].to_numpy() - means) / stds).max(axis=1)
        parameters[station] = {
            "delta_upper": delta_mean + 3.0 * delta_std,
            "means": means.tolist(),
            "stds": stds.tolist(),
            "multivariate_threshold": float(np.quantile(scores, 0.997)),
        }
    return parameters


def predict(test, parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    fixed = np.zeros(len(test), dtype=bool)
    multivariate = np.zeros(len(test), dtype=bool)
    for station, index in test.groupby("station", sort=False).groups.items():
        rows = np.asarray(list(index), dtype=int)
        frame = test.loc[rows]
        params = parameters[station]
        fixed[rows] = frame["delta_c"].to_numpy() > params["delta_upper"]
        values = frame.loc[:, MULTIVARIATE_COLUMNS].to_numpy(dtype=float)
        score = ((values - np.asarray(params["means"])) / np.asarray(params["stds"])).max(axis=1)
        multivariate[rows] = score > params["multivariate_threshold"]
    return fixed, multivariate


def radio_metrics(prediction: np.ndarray, resets: np.ndarray, heartbeat_samples: int = 72) -> dict:
    previous = False
    since_tx = heartbeat_samples
    transmissions = 0
    for alert, reset in zip(prediction, resets):
        if reset:
            previous = False
            since_tx = heartbeat_samples
        changed = bool(alert) != previous
        if changed or since_tx >= heartbeat_samples:
            transmissions += 1
            since_tx = 0
        else:
            since_tx += 1
        previous = bool(alert)
    return {
        "transmissions": transmissions,
        "periodic_transmissions": len(prediction),
        "transmissions_avoided_fraction": 1.0 - transmissions / len(prediction),
    }


def evaluate_model(frame, prediction) -> dict:
    truth = frame["label"].isin(["Warming", "Anomaly"]).to_numpy()
    return {
        "rows": len(frame),
        "alert": binary_alert_metrics(truth, prediction),
        "events": event_metrics(frame, prediction),
        "radio": radio_metrics(
            np.asarray(prediction, dtype=bool), frame["sequence_reset"].to_numpy(dtype=bool)
        ),
    }


def write_markdown(results: dict) -> None:
    lines = [
        "# Generated Baseline Results",
        "",
        "Training-only station calibration; evaluation on the bounded synthetic-v2 test split.",
        "",
        "| Model | Precision | Recall | FPR | Event recall | Transmissions avoided |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results["models"].items():
        lines.append(
            f"| {name} | {metrics['alert']['precision']:.4f} | {metrics['alert']['recall']:.4f} | "
            f"{metrics['alert']['false_positive_rate']:.4f} | {metrics['events']['event_recall']:.4f} | "
            f"{metrics['radio']['transmissions_avoided_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "These are synthetic software metrics, not field-performance claims. The fixed detector is one-sided because the target is excess heating.",
            "",
        ]
    )
    DEFAULT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_MARKDOWN.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--stations", nargs="*", default=None)
    parser.add_argument("--max-rows-per-station", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    kwargs = dict(
        feature_dir=args.feature_dir,
        stations=args.stations,
        max_rows_per_station=args.max_rows_per_station,
        seed=args.seed,
    )
    train = load_split("train", **kwargs)
    test = load_split("test", **kwargs)
    parameters = fit_station_parameters(train)
    fixed, multivariate = predict(test, parameters)
    results = {
        "train_rows": len(train),
        "test_rows": len(test),
        "max_rows_per_station": args.max_rows_per_station,
        "parameters": parameters,
        "models": {
            "fixed_delta_3sigma": evaluate_model(test, fixed),
            "multivariate_statistical": evaluate_model(test, multivariate),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(results)
    for name, metrics in results["models"].items():
        alert = metrics["alert"]
        print(f"{name}: precision={alert['precision']:.4f}, recall={alert['recall']:.4f}, FPR={alert['false_positive_rate']:.4f}")
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
