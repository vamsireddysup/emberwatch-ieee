"""Sample, event, and radio-policy metrics for EmberWatch models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ml_data import LABELS


def confusion_matrix(y_true, y_pred, n_classes: int = 3) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def classification_metrics(y_true, y_pred) -> dict:
    matrix = confusion_matrix(y_true, y_pred, len(LABELS))
    per_class = {}
    f1_values = []
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index].sum())}
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def binary_alert_metrics(y_true, y_pred) -> dict:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(truth & pred))
    tn = int(np.sum(~truth & ~pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def tune_alert_threshold(y_true, probabilities, target_fpr: float = 0.01) -> tuple[float, dict]:
    truth = np.asarray(y_true, dtype=int) > 0
    alert_score = np.asarray(probabilities)[:, 1:].sum(axis=1)
    candidates = []
    for threshold in np.linspace(0.20, 0.90, 71):
        metrics = binary_alert_metrics(truth, alert_score >= threshold)
        feasible = metrics["false_positive_rate"] <= target_fpr
        # Prefer feasible high recall, then F1. Otherwise minimize FPR before maximizing recall.
        rank = (int(feasible), metrics["recall"] if feasible else -metrics["false_positive_rate"], metrics["f1"])
        candidates.append((rank, float(threshold), metrics))
    _, threshold, metrics = max(candidates, key=lambda item: item[0])
    return threshold, metrics


def apply_alert_policy(
    probabilities,
    threshold: float,
    warming_confirmations: int = 2,
    clear_confirmations: int = 3,
    heartbeat_samples: int = 72,
    resets=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply persistence and return operational states plus transmission decisions."""
    probabilities = np.asarray(probabilities)
    if resets is None:
        resets = np.zeros(len(probabilities), dtype=bool)
        if len(resets):
            resets[0] = True
    resets = np.asarray(resets, dtype=bool)
    if len(resets) != len(probabilities):
        raise ValueError("resets and probabilities must have the same length")
    states = np.zeros(len(probabilities), dtype=np.int8)
    transmit = np.zeros(len(probabilities), dtype=bool)
    current = 0
    warming_run = 0
    normal_run = 0
    since_tx = heartbeat_samples
    for index, prob in enumerate(probabilities):
        if resets[index]:
            current = 0
            warming_run = 0
            normal_run = 0
            since_tx = heartbeat_samples
        alert_score = float(prob[1] + prob[2])
        candidate = 2 if prob[2] >= threshold else (1 if alert_score >= threshold else 0)
        previous = current
        if candidate == 2:
            current = 2
            warming_run = normal_run = 0
        elif candidate == 1:
            warming_run += 1
            normal_run = 0
            if warming_run >= warming_confirmations and current == 0:
                current = 1
        else:
            normal_run += 1
            warming_run = 0
            if normal_run >= clear_confirmations:
                current = 0
        states[index] = current
        state_changed = current != previous
        heartbeat = since_tx >= heartbeat_samples
        transmit[index] = state_changed or heartbeat
        since_tx = 0 if transmit[index] else since_tx + 1
    return states, transmit


def event_metrics(df: pd.DataFrame, predicted_alert) -> dict:
    pred = np.asarray(predicted_alert, dtype=bool)
    if len(df) != len(pred):
        raise ValueError("df and predicted_alert must have the same length")
    working = df[["timestamp", "event_id", "label"]].copy()
    working["predicted_alert"] = pred
    if "station" in df.columns:
        working["station"] = df["station"].to_numpy()
        event_keys = ["station", "event_id"]
    else:
        event_keys = ["event_id"]
    events = working.loc[working["event_id"] > 0].groupby(event_keys, sort=False)
    detected = 0
    lead_minutes = []
    for _, event in events:
        alerts = event.loc[event["predicted_alert"], "timestamp"]
        if alerts.empty:
            continue
        detected += 1
        anomaly = event.loc[event["label"] == "Anomaly", "timestamp"]
        if not anomaly.empty:
            lead_minutes.append((anomaly.min() - alerts.min()).total_seconds() / 60.0)

    normal = working["label"].eq("Normal").to_numpy()
    false_alert = pred & normal
    if "station" in working:
        reset = working["station"].ne(working["station"].shift()).to_numpy()
        durations = working.groupby("station")["timestamp"].agg(lambda values: values.max() - values.min())
        duration_days = sum(value.total_seconds() / 86400.0 for value in durations)
    else:
        reset = np.zeros(len(working), dtype=bool)
        if len(reset):
            reset[0] = True
        duration_days = (working["timestamp"].max() - working["timestamp"].min()).total_seconds() / 86400.0
    previous_false = np.r_[False, false_alert[:-1]]
    false_episode_starts = false_alert & (~previous_false | reset)
    duration_days = max(duration_days, 1 / 288)
    total_events = int(events.ngroups)
    return {
        "event_count": total_events,
        "events_detected": detected,
        "event_recall": detected / total_events if total_events else 0.0,
        "median_lead_minutes": float(np.median(lead_minutes)) if lead_minutes else None,
        "p10_lead_minutes": float(np.percentile(lead_minutes, 10)) if lead_minutes else None,
        "false_alert_episodes_per_day": float(false_episode_starts.sum() / duration_days),
    }
