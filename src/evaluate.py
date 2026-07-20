"""
Step 4 evaluation harness for EmberWatch.

This is the shared scoring contract every model in this project -- the baselines here,
and eventually Amogh's ESN -- is measured with, so results are directly comparable no
matter who trained what or how.

"Alert" (the positive class) means the node would wake the radio and transmit: ground
truth label is Warming or Anomaly, as opposed to Normal. False positive rate matters
specifically because it's the direct driver of the project's core claim (fewer radio
transmissions / battery life multiplier) -- a high false positive rate means the node
transmits more often than necessary.

Two ways to use this:

1. In-process (for code that already has y_true/y_pred arrays in memory, like
   src/baselines.py): call score_predictions(model_name, y_true, y_pred).

2. From the command line (for anyone -- e.g. Amogh -- scoring a model that ran outside
   this codebase): produce a predictions CSV with exactly two columns,
       timestamp, predicted_alert
   where predicted_alert is 0/1 or True/False, one row per timestamp your model made a
   decision for, then run:
       python src/evaluate.py --model-name esn --predictions path/to/your_predictions.csv
   Ground truth is read from data/processed/features_nab.csv by default (override with
   --ground-truth). Predictions are matched to ground truth by exact timestamp (inner
   join) -- if your predictions don't cover every ground-truth row, evaluate() runs on
   whatever matched and a warning is printed with the coverage count.

Either path calls the same evaluate() function and appends one row to
data/processed/evaluation_results.csv, keyed by model_name (rerunning a model overwrites
its previous row rather than appending a duplicate), so that file ends up as the single
side-by-side comparison table across every model tried.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data" / "processed" / "features_nab.csv"
RESULTS_PATH = REPO_ROOT / "data" / "processed" / "evaluation_results.csv"

METRIC_COLUMNS = ["accuracy", "precision", "recall", "false_positive_rate", "tp", "tn", "fp", "fn"]


def evaluate(y_true, y_pred) -> dict:
    """
    y_true, y_pred: array-like of booleans (or 0/1), same length, aligned by row.
    True/1 means "alert" (Warming or Anomaly); False/0 means Normal.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must be the same length, got {len(y_true)} vs {len(y_pred)}")

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"  {name}:")
    print(f"    accuracy:             {metrics['accuracy']:.4f}")
    print(f"    precision:            {metrics['precision']:.4f}")
    print(f"    recall:               {metrics['recall']:.4f}")
    print(f"    false_positive_rate:  {metrics['false_positive_rate']:.4f}")
    print(f"    (tp={metrics['tp']}, tn={metrics['tn']}, fp={metrics['fp']}, fn={metrics['fn']})")


def record_result(model_name: str, metrics: dict, n_rows: int, results_path: Path = RESULTS_PATH) -> None:
    """Upserts one row into the shared results log, keyed by model_name."""
    row = {
        "model_name": model_name,
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_rows": n_rows,
        **{col: metrics[col] for col in METRIC_COLUMNS},
    }
    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing = existing[existing["model_name"] != model_name]
        out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(results_path, index=False)


def score_predictions(model_name: str, y_true, y_pred, results_path: Path = RESULTS_PATH) -> dict:
    """In-process entrypoint: scores, prints, and records a result for a model that
    already has y_true/y_pred arrays in memory."""
    metrics = evaluate(y_true, y_pred)
    record_result(model_name, metrics, len(np.asarray(y_true)), results_path)
    print_metrics(model_name, metrics)
    return metrics


def load_ground_truth(path: Path = DEFAULT_GROUND_TRUTH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["alert"] = df["label"].isin(["Warming", "Anomaly"])
    return df[["timestamp", "alert"]]


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if series.dtype == object:
        return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    return series.astype(int).astype(bool)


def score_predictions_csv(
    model_name: str,
    predictions_path: Path,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH,
    results_path: Path = RESULTS_PATH,
) -> dict:
    """CLI entrypoint: predictions_path is a CSV with columns timestamp, predicted_alert."""
    truth = load_ground_truth(ground_truth_path)
    preds = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    if "timestamp" not in preds.columns or "predicted_alert" not in preds.columns:
        raise ValueError(
            f"{predictions_path} must have columns 'timestamp' and 'predicted_alert', "
            f"found: {list(preds.columns)}"
        )
    preds["predicted_alert"] = _coerce_bool(preds["predicted_alert"])

    merged = pd.merge(truth, preds[["timestamp", "predicted_alert"]], on="timestamp", how="inner")
    if len(merged) < len(truth):
        print(
            f"  WARNING: predictions only matched {len(merged)}/{len(truth)} ground-truth "
            "timestamps; scoring on the matched subset only"
        )

    return score_predictions(model_name, merged["alert"], merged["predicted_alert"], results_path)


def main():
    parser = argparse.ArgumentParser(description="EmberWatch Step 4 evaluation harness")
    parser.add_argument("--model-name", required=True, help="Name to record this model's results under")
    parser.add_argument("--predictions", required=True, type=Path, help="CSV with columns timestamp, predicted_alert")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    score_predictions_csv(args.model_name, args.predictions, args.ground_truth, args.results)
    print(f"\n  recorded to {args.results}")


if __name__ == "__main__":
    main()
