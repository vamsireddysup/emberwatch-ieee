"""Pure NumPy echo state network designed for deterministic MCU export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ESNConfig:
    input_size: int
    reservoir_size: int = 48
    output_size: int = 3
    spectral_radius: float = 0.85
    leak_rate: float = 0.25
    input_scale: float = 0.5
    connectivity: float = 0.15
    ridge: float = 1e-2
    seed: int = 20260722
    clip_normalized: float = 6.0


class EchoStateClassifier:
    """ESN with fixed reservoir and weighted ridge-regression readout."""

    def __init__(self, config: ESNConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.input_weights = rng.uniform(
            -config.input_scale,
            config.input_scale,
            size=(config.reservoir_size, config.input_size + 1),
        ).astype(np.float32)
        recurrent = rng.uniform(-1.0, 1.0, size=(config.reservoir_size, config.reservoir_size))
        recurrent[rng.random(recurrent.shape) > config.connectivity] = 0.0
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        if radius == 0:
            raise ValueError("Generated reservoir has zero spectral radius; increase connectivity")
        self.reservoir_weights = (recurrent * (config.spectral_radius / radius)).astype(np.float32)
        self.feature_mean = np.zeros(config.input_size, dtype=np.float32)
        self.feature_std = np.ones(config.input_size, dtype=np.float32)
        self.output_weights = np.zeros(
            (config.output_size, 1 + config.input_size + config.reservoir_size), dtype=np.float32
        )
        self.alert_threshold = 0.5
        self.metadata: dict = {}

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, dtype=np.float32) - self.feature_mean) / self.feature_std
        return np.clip(normalized, -self.config.clip_normalized, self.config.clip_normalized)

    def reservoir_states(self, features: np.ndarray, resets: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        inputs = self._normalize(features)
        n_rows = len(inputs)
        if resets is None:
            resets = np.zeros(n_rows, dtype=bool)
            if n_rows:
                resets[0] = True
        resets = np.asarray(resets, dtype=bool)
        if len(resets) != n_rows:
            raise ValueError("resets and features must have the same length")

        states = np.empty((n_rows, self.config.reservoir_size), dtype=np.float32)
        valid_after_washout = np.ones(n_rows, dtype=bool)
        state = np.zeros(self.config.reservoir_size, dtype=np.float32)
        since_reset = 0
        for row in range(n_rows):
            if resets[row]:
                state.fill(0.0)
                since_reset = 0
            augmented_input = np.empty(self.config.input_size + 1, dtype=np.float32)
            augmented_input[0] = 1.0
            augmented_input[1:] = inputs[row]
            candidate = np.tanh(
                self.input_weights @ augmented_input + self.reservoir_weights @ state
            )
            state = (1.0 - self.config.leak_rate) * state + self.config.leak_rate * candidate
            states[row] = state
            valid_after_washout[row] = since_reset >= 2
            since_reset += 1
        return states, valid_after_washout

    def _design_matrix(self, features: np.ndarray, states: np.ndarray) -> np.ndarray:
        normalized = self._normalize(features)
        return np.column_stack(
            [np.ones(len(normalized), dtype=np.float32), normalized, states]
        ).astype(np.float32, copy=False)

    def fit(self, features: np.ndarray, labels: np.ndarray, resets: np.ndarray | None = None) -> "EchoStateClassifier":
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != self.config.input_size:
            raise ValueError(f"Expected feature matrix with {self.config.input_size} columns")
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")

        self.feature_mean = np.nanmean(features, axis=0).astype(np.float32)
        self.feature_std = np.nanstd(features, axis=0).astype(np.float32)
        self.feature_std[self.feature_std < 1e-6] = 1.0

        states, valid = self.reservoir_states(features, resets)
        design = self._design_matrix(features, states)[valid]
        fit_labels = labels[valid]
        counts = np.bincount(fit_labels, minlength=self.config.output_size).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError(f"Every class must occur in training data; counts={counts.tolist()}")
        # Square-root balancing is less brittle than full inverse-frequency weighting.
        class_weights = np.sqrt(counts.sum() / (self.config.output_size * counts))
        sample_scale = np.sqrt(class_weights[fit_labels]).astype(np.float32)
        weighted_design = design * sample_scale[:, None]
        targets = np.eye(self.config.output_size, dtype=np.float32)[fit_labels]
        weighted_targets = targets * sample_scale[:, None]

        weighted_design_64 = weighted_design.astype(np.float64)
        weighted_targets_64 = weighted_targets.astype(np.float64)
        # Some Accelerate/OpenBLAS builds emit spurious floating-point warnings during
        # finite matrix multiplication. Validate the result explicitly instead.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            gram = weighted_design_64.T @ weighted_design_64
            rhs = weighted_design_64.T @ weighted_targets_64
        if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
            raise FloatingPointError("Non-finite values reached the ESN ridge system")
        regularizer = np.eye(gram.shape[0], dtype=np.float64) * self.config.ridge
        regularizer[0, 0] = 0.0
        self.output_weights = np.linalg.solve(gram + regularizer, rhs).T.astype(np.float32)
        if not np.isfinite(self.output_weights).all():
            raise FloatingPointError("ESN ridge solve produced non-finite output weights")
        self.metadata["train_class_counts"] = counts.astype(int).tolist()
        self.metadata["class_weights"] = class_weights.tolist()
        return self

    def predict_logits(self, features: np.ndarray, resets: np.ndarray | None = None) -> np.ndarray:
        states, _ = self.reservoir_states(features, resets)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = self._design_matrix(features, states) @ self.output_weights.T
        if not np.isfinite(logits).all():
            raise FloatingPointError("ESN inference produced non-finite logits")
        return logits

    def predict_proba(self, features: np.ndarray, resets: np.ndarray | None = None) -> np.ndarray:
        logits = self.predict_logits(features, resets).astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        exponent = np.exp(logits)
        return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)

    def predict(self, features: np.ndarray, resets: np.ndarray | None = None) -> np.ndarray:
        return np.argmax(self.predict_logits(features, resets), axis=1).astype(np.int8)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.__dict__,
            "alert_threshold": self.alert_threshold,
            "metadata": self.metadata,
        }
        np.savez_compressed(
            path,
            input_weights=self.input_weights,
            reservoir_weights=self.reservoir_weights,
            output_weights=self.output_weights,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            manifest=np.array(json.dumps(payload)),
        )

    @classmethod
    def load(cls, path: Path) -> "EchoStateClassifier":
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["manifest"]))
            model = cls(ESNConfig(**payload["config"]))
            model.input_weights = data["input_weights"].astype(np.float32)
            model.reservoir_weights = data["reservoir_weights"].astype(np.float32)
            model.output_weights = data["output_weights"].astype(np.float32)
            model.feature_mean = data["feature_mean"].astype(np.float32)
            model.feature_std = data["feature_std"].astype(np.float32)
            model.alert_threshold = float(payload["alert_threshold"])
            model.metadata = payload.get("metadata", {})
        return model
