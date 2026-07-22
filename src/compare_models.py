"""One apples-to-apples comparison table across every EmberWatch detector.

The project accumulated two scoring paths: src/evaluate.py (binary, timestamp-keyed CSV
contract, writes data/processed/evaluation_results.csv) and src/metrics.py (three-class +
event + radio, used by the ESN trainer and baselines_v2). They answer different questions
and both stay. What was missing is a single place that scores the deployable candidates --
the threshold baselines, the float ESN, and the int16/int8 quantized ESN -- through the
IDENTICAL operational policy and metric set on the same test split, so a reviewer can read
one table.

This module is that place. It:
  - loads the bounded v2 test split via src.ml_data (same loader the trainer uses),
  - runs each ESN variant through src.metrics.apply_alert_policy (persistence + heartbeat)
    and then binary_alert_metrics + event_metrics + radio counts,
  - reads the baseline results from artifacts/reports/baseline_metrics.json (produced by
    src.baselines_v2) so baselines and ESNs sit in the same table without re-deriving them,
  - joins per-model model size (float vs quantized, from src.quantize) and battery life
    (from src.energy_model) so size/accuracy/energy trade-offs are visible together,
  - writes docs/generated/MODEL_COMPARISON.md and artifacts/reports/model_comparison.json.

It does not retrain and it does not modify evaluate.py or metrics.py.

Usage:
    python -m src.compare_models                       # needs a trained model + reports
    python -m src.compare_models --max-rows-per-station 80000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .energy_model import HardwareAssumptions, battery_life_days
from .esn import EchoStateClassifier
from .metrics import apply_alert_policy, binary_alert_metrics, event_metrics
from .ml_data import feature_matrix, load_split
from .quantize import QuantSpec, QuantizedESN, float_size_bytes

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "artifacts" / "models" / "esn_model.npz"
BASELINE_REPORT = REPO_ROOT / "artifacts" / "reports" / "baseline_metrics.json"
DEFAULT_JSON = REPO_ROOT / "artifacts" / "reports" / "model_comparison.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "generated" / "MODEL_COMPARISON.md"

SAMPLES_PER_DAY = 288


def _evaluate_proba(df, probabilities: np.ndarray, threshold: float) -> dict:
    """Apply the operational alert policy and return the shared metric bundle."""
    resets = df["sequence_reset"].to_numpy(dtype=bool)
    truth = df["label_index"].to_numpy() > 0
    states, transmit = apply_alert_policy(probabilities, threshold, resets=resets)
    alert = states > 0
    return {
        "alert": binary_alert_metrics(truth, alert),
        "events": event_metrics(df, alert),
        "radio": {
            "transmissions": int(transmit.sum()),
            "transmissions_avoided_fraction": float(1.0 - transmit.mean()),
        },
    }


def _battery_years(avoided_fraction: float, hw: HardwareAssumptions) -> float:
    tx_per_day = SAMPLES_PER_DAY * (1.0 - avoided_fraction)
    return battery_life_days(tx_per_day, hw) / 365.25


def compare(model_path: Path, max_rows_per_station: int) -> dict:
    model = EchoStateClassifier.load(model_path)
    threshold = model.alert_threshold
    hw = HardwareAssumptions()
    print(f"Loaded {model_path} (reservoir={model.config.reservoir_size}, threshold={threshold:.3f})")

    test = load_split("test", max_rows_per_station=max_rows_per_station)
    features = feature_matrix(test)
    resets = test["sequence_reset"].to_numpy(dtype=bool)
    print(f"Test rows: {len(test)}")

    rows: dict[str, dict] = {}

    # ESN variants through the identical operational policy.
    esn_specs = {
        "esn_float32": (model.predict_proba(features, resets), float_size_bytes(model)["total"]),
    }
    for name, spec in (("esn_int16_sparse", QuantSpec(16, True)), ("esn_int8_sparse", QuantSpec(8, True))):
        quantized = QuantizedESN(model, spec)
        esn_specs[name] = (quantized.predict_proba(features, resets), quantized.size_bytes()["total"])

    for name, (proba, size_bytes) in esn_specs.items():
        metrics = _evaluate_proba(test, proba, threshold)
        metrics["size_bytes"] = size_bytes
        metrics["battery_life_years"] = _battery_years(metrics["radio"]["transmissions_avoided_fraction"], hw)
        rows[name] = metrics

    # Threshold baselines from the existing report (already leakage-safe, train-calibrated).
    if BASELINE_REPORT.exists():
        baseline = json.loads(BASELINE_REPORT.read_text())
        for name, entry in baseline.get("models", {}).items():
            avoided = entry["radio"]["transmissions_avoided_fraction"]
            rows[name] = {
                "alert": entry["alert"],
                "events": entry["events"],
                "radio": {
                    "transmissions": entry["radio"].get("transmissions"),
                    "transmissions_avoided_fraction": avoided,
                },
                "size_bytes": None,  # threshold detectors carry only per-station scalars
                "battery_life_years": _battery_years(avoided, hw),
            }
    else:
        print(f"  NOTE: {BASELINE_REPORT} not found; run 'make baselines' to include baselines")

    return {
        "model": str(model_path),
        "test_rows": int(len(test)),
        "alert_threshold": float(threshold),
        "models": rows,
    }


def write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# Generated Model Comparison",
        "",
        "Every deployable detector scored through the identical operational alert policy",
        "(persistence + heartbeat) on the same bounded synthetic-v2 2023 test split. ESN",
        "variants are evaluated live; threshold baselines are read from",
        "`artifacts/reports/baseline_metrics.json`. Battery years use the placeholder",
        "hardware assumptions in `src/energy_model.py`. Software validation, not field claims.",
        "",
        "| Model | Size (B) | Alert prec | Alert recall | FPR | Event recall | Median lead (min) | Tx avoided | Battery (yr) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value, spec="{:.4f}"):
        return "n/a" if value is None else spec.format(value)

    for name, m in results["models"].items():
        events = m["events"]
        lines.append(
            f"| {name} | {fmt(m['size_bytes'], '{:d}') if m['size_bytes'] else 'n/a'} | "
            f"{fmt(m['alert']['precision'])} | {fmt(m['alert']['recall'])} | "
            f"{fmt(m['alert']['false_positive_rate'])} | {fmt(events['event_recall'])} | "
            f"{fmt(events.get('median_lead_minutes'), '{:.1f}')} | "
            f"{fmt(m['radio']['transmissions_avoided_fraction'])} | "
            f"{fmt(m['battery_life_years'], '{:.1f}')} |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        "- **Event recall and median lead** are where the ESN separates from the threshold",
        "  baselines: it detects more distinct fault events and, at a positive median lead,",
        "  alerts before the anomaly label rather than after. This is the AI claim.",
        "- **Tx avoided and battery years** are nearly identical across the ESN and the",
        "  better baseline. Gating is a system property, not an ESN-specific one.",
        "- **Size** shows the quantized ESN variants cost a fraction of the float model while",
        "  holding event recall (see `docs/generated/QUANTIZATION_RESULTS.md`).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "./venv/bin/python -m src.train_esn --max-rows-per-station 80000 --reservoir-size 48",
        "./venv/bin/python -m src.baselines_v2 --max-rows-per-station 80000",
        "./venv/bin/python -m src.compare_models --max-rows-per-station 80000",
        "```",
        "",
        "Full numbers in `artifacts/reports/model_comparison.json`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-rows-per-station", type=int, default=80000)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    results = compare(args.model, args.max_rows_per_station)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(results, args.markdown)

    header = f"{'model':<26}{'size':>8}{'ev.rec':>9}{'lead':>8}{'tx_avoid':>10}{'batt_yr':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name, m in results["models"].items():
        size = str(m["size_bytes"]) if m["size_bytes"] else "n/a"
        lead = m["events"].get("median_lead_minutes")
        lead_s = "n/a" if lead is None else f"{lead:.1f}"
        print(
            f"{name:<26}{size:>8}{m['events']['event_recall']:>9.4f}{lead_s:>8}"
            f"{m['radio']['transmissions_avoided_fraction']:>10.4f}{m['battery_life_years']:>9.1f}"
        )
    print(f"\nWrote {args.json}\nWrote {args.markdown}")


if __name__ == "__main__":
    main()
