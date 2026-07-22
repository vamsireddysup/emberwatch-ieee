"""Leave-one-station-out ESN experiment for geographic generalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .esn import ESNConfig, EchoStateClassifier
from .metrics import apply_alert_policy, binary_alert_metrics, event_metrics, tune_alert_threshold
from .ml_data import DEFAULT_FEATURE_DIR, FEATURE_COLUMNS, discover_feature_files, feature_matrix, load_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "reports" / "loso_metrics.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "generated" / "LOSO_RESULTS.md"


def evaluate(model, frame, threshold: float) -> dict:
    probabilities = model.predict_proba(
        feature_matrix(frame), frame["sequence_reset"].to_numpy(dtype=bool)
    )
    resets = frame["sequence_reset"].to_numpy(dtype=bool)
    states, transmit = apply_alert_policy(probabilities, threshold, resets=resets)
    truth = frame["label_index"].to_numpy() > 0
    return {
        "rows": len(frame),
        "alert": binary_alert_metrics(truth, states > 0),
        "events": event_metrics(frame, states > 0),
        "transmissions_avoided_fraction": float(1.0 - transmit.mean()),
    }


def write_markdown(results: dict) -> None:
    lines = [
        "# Leave-One-Station-Out Results",
        "",
        "Each row tests 2023 data from a station excluded from all model fitting and threshold selection.",
        "",
        "| Held-out station | Precision | Recall | FPR | Event recall | Median lead min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for station, result in results["stations"].items():
        alert = result["test"]["alert"]
        events = result["test"]["events"]
        lead = "n/a" if events["median_lead_minutes"] is None else f"{events['median_lead_minutes']:.1f}"
        lines.append(
            f"| {station} | {alert['precision']:.4f} | {alert['recall']:.4f} | "
            f"{alert['false_positive_rate']:.4f} | {events['event_recall']:.4f} | {lead} |"
        )
    lines.extend(
        [
            "",
            "This is synthetic geographic generalization evidence, not field validation.",
            "",
        ]
    )
    DEFAULT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_MARKDOWN.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--max-rows-per-station", type=int, default=30000)
    parser.add_argument("--reservoir-size", type=int, default=32)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    stations = [path.stem.removeprefix("features_v2_") for path in discover_feature_files(args.feature_dir)]
    results = {
        "max_rows_per_station": args.max_rows_per_station,
        "reservoir_size": args.reservoir_size,
        "stations": {},
    }
    for held_out_index, held_out in enumerate(stations):
        train_stations = [station for station in stations if station != held_out]
        common = dict(
            feature_dir=args.feature_dir,
            max_rows_per_station=args.max_rows_per_station,
            seed=args.seed + held_out_index,
        )
        train = load_split("train", stations=train_stations, **common)
        validation = load_split("val", stations=train_stations, **common)
        test = load_split("test", stations=[held_out], **common)
        config = ESNConfig(
            input_size=len(FEATURE_COLUMNS),
            reservoir_size=args.reservoir_size,
            seed=args.seed + held_out_index,
        )
        model = EchoStateClassifier(config).fit(
            feature_matrix(train),
            train["label_index"].to_numpy(),
            train["sequence_reset"].to_numpy(dtype=bool),
        )
        validation_probabilities = model.predict_proba(
            feature_matrix(validation), validation["sequence_reset"].to_numpy(dtype=bool)
        )
        threshold, _ = tune_alert_threshold(
            validation["label_index"].to_numpy(), validation_probabilities, args.target_fpr
        )
        results["stations"][held_out] = {
            "training_stations": train_stations,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "threshold": threshold,
            "test": evaluate(model, test, threshold),
        }
        alert = results["stations"][held_out]["test"]["alert"]
        print(f"held out {held_out}: precision={alert['precision']:.4f}, recall={alert['recall']:.4f}, FPR={alert['false_positive_rate']:.4f}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(results)
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
