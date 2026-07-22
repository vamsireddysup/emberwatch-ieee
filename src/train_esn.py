"""Train, validate, evaluate, save, and export the EmberWatch ESN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .esn import ESNConfig, EchoStateClassifier
from .export_c import export_model
from .metrics import (
    apply_alert_policy,
    binary_alert_metrics,
    classification_metrics,
    event_metrics,
    tune_alert_threshold,
)
from .ml_data import DEFAULT_FEATURE_DIR, FEATURE_COLUMNS, feature_matrix, load_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts"
DEFAULT_RESULTS_MD = REPO_ROOT / "docs" / "generated" / "ML_RESULTS.md"


def _evaluate_split(model: EchoStateClassifier, df, threshold: float) -> dict:
    features = feature_matrix(df)
    resets = df["sequence_reset"].to_numpy(dtype=bool)
    truth = df["label_index"].to_numpy(dtype=np.int8)
    probabilities = model.predict_proba(features, resets)
    raw_classes = probabilities.argmax(axis=1)
    operational_states, transmit = apply_alert_policy(probabilities, threshold, resets=resets)
    result = {
        "rows": len(df),
        "class_distribution": df["label"].value_counts().to_dict(),
        "three_class": classification_metrics(truth, raw_classes),
        "raw_alert": binary_alert_metrics(truth > 0, probabilities[:, 1:].sum(axis=1) >= threshold),
        "operational_alert": binary_alert_metrics(truth > 0, operational_states > 0),
        "events": event_metrics(df, operational_states > 0),
        "radio": {
            "transmissions": int(transmit.sum()),
            "periodic_transmissions": len(transmit),
            "transmissions_avoided_fraction": float(1.0 - transmit.mean()),
        },
    }
    return result


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_markdown(results: dict, path: Path) -> None:
    val = results["validation"]
    test = results["test"]
    lines = [
        "# Generated ML Results",
        "",
        f"Run profile: `{results['profile']}`. Seed: `{results['config']['seed']}`.",
        "",
        "These values are software validation on the local synthetic-v2 data. They are not field-performance claims.",
        "",
        "| Metric | Validation | Test |",
        "|---|---:|---:|",
        f"| Rows | {val['rows']} | {test['rows']} |",
        f"| Three-class macro F1 | {_fmt(val['three_class']['macro_f1'])} | {_fmt(test['three_class']['macro_f1'])} |",
        f"| Alert precision | {_fmt(val['operational_alert']['precision'])} | {_fmt(test['operational_alert']['precision'])} |",
        f"| Alert recall | {_fmt(val['operational_alert']['recall'])} | {_fmt(test['operational_alert']['recall'])} |",
        f"| Alert false-positive rate | {_fmt(val['operational_alert']['false_positive_rate'])} | {_fmt(test['operational_alert']['false_positive_rate'])} |",
        f"| Event recall | {_fmt(val['events']['event_recall'])} | {_fmt(test['events']['event_recall'])} |",
        f"| Median lead minutes | {_fmt(val['events']['median_lead_minutes'])} | {_fmt(test['events']['median_lead_minutes'])} |",
        f"| False alert episodes/device-day | {_fmt(val['events']['false_alert_episodes_per_day'])} | {_fmt(test['events']['false_alert_episodes_per_day'])} |",
        f"| Periodic transmissions avoided | {_fmt(val['radio']['transmissions_avoided_fraction'])} | {_fmt(test['radio']['transmissions_avoided_fraction'])} |",
        "",
        f"Validation-selected alert threshold: `{results['alert_threshold']:.3f}`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        results["command"],
        "```",
        "",
        "Inspect `artifacts/reports/esn_metrics.json` for confusion matrices and per-class values.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--stations", nargs="*", default=None)
    parser.add_argument("--max-rows-per-station", type=int, default=80000)
    parser.add_argument("--reservoir-size", type=int, default=48)
    parser.add_argument("--spectral-radius", type=float, default=0.85)
    parser.add_argument("--leak-rate", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()

    load_args = dict(
        feature_dir=args.feature_dir,
        stations=args.stations,
        max_rows_per_station=args.max_rows_per_station,
        seed=args.seed,
    )
    print("Loading train/validation/test splits...")
    train = load_split("train", **load_args)
    validation = load_split("val", **load_args)
    test = load_split("test", **load_args)
    print(f"  rows: train={len(train)}, val={len(validation)}, test={len(test)}")

    config = ESNConfig(
        input_size=len(FEATURE_COLUMNS),
        reservoir_size=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        leak_rate=args.leak_rate,
        ridge=args.ridge,
        seed=args.seed,
    )
    model = EchoStateClassifier(config)
    model.metadata.update(
        {
            "feature_columns": list(FEATURE_COLUMNS),
            "sample_cadence_minutes": 5,
            "stations": args.stations or "all",
            "max_rows_per_station": args.max_rows_per_station,
        }
    )
    print("Training weighted ESN readout...")
    model.fit(
        feature_matrix(train),
        train["label_index"].to_numpy(),
        train["sequence_reset"].to_numpy(dtype=bool),
    )

    val_probabilities = model.predict_proba(
        feature_matrix(validation), validation["sequence_reset"].to_numpy(dtype=bool)
    )
    threshold, threshold_metrics = tune_alert_threshold(
        validation["label_index"].to_numpy(), val_probabilities, args.target_fpr
    )
    model.alert_threshold = threshold
    model.metadata["validation_threshold_metrics"] = threshold_metrics
    print(f"Selected validation alert threshold: {threshold:.3f}")

    results = {
        "profile": "bounded" if args.max_rows_per_station else "full",
        "config": config.__dict__,
        "feature_columns": list(FEATURE_COLUMNS),
        "alert_threshold": threshold,
        "train_rows": len(train),
        "validation": _evaluate_split(model, validation, threshold),
        "test": _evaluate_split(model, test, threshold),
        "command": (
            "./venv/bin/python -m src.train_esn "
            f"--max-rows-per-station {args.max_rows_per_station} --reservoir-size {args.reservoir_size}"
        ),
    }

    model_path = args.artifact_dir / "models" / "esn_model.npz"
    report_path = args.artifact_dir / "reports" / "esn_metrics.json"
    header_path = REPO_ROOT / "firmware" / "generated" / "emberwatch_model.h"
    model.save(model_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    export_model(model, header_path)
    write_markdown(results, DEFAULT_RESULTS_MD)

    test_alert = results["test"]["operational_alert"]
    test_events = results["test"]["events"]
    print(f"Test alert precision={test_alert['precision']:.4f}, recall={test_alert['recall']:.4f}, FPR={test_alert['false_positive_rate']:.4f}")
    print(f"Test event recall={test_events['event_recall']:.4f}, median lead={test_events['median_lead_minutes']}")
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"Exported C header: {header_path}")


if __name__ == "__main__":
    main()
