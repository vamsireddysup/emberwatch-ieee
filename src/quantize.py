"""Post-training quantization and sparse packing for the EmberWatch ESN.

Why this exists: the exported float32 model is about 11.3 KB of weights, but the
deployment target is an STM32WL55 Cortex-M4 where the model was specified to be int8 or
int16 and to fit in a few kilobytes. Two independent savings are available and this
module applies both:

1. Symmetric integer quantization of every weight matrix. Weights are stored as int8 or
   int16 with a per-row (per output channel) float scale; inference dequantizes on the
   fly and accumulates in float. This is weight-only quantization: it buys the flash
   reduction, which is the binding constraint here, while keeping numerics close enough
   to the float model to verify by parity. Full integer accumulation would additionally
   buy speed, but it is a larger change and should be driven by measured MCU latency
   rather than assumed.

2. Sparse packing of the reservoir matrix. The reservoir is generated with
   `connectivity=0.15`, so roughly 85 percent of its 48x48 entries are exactly zero, yet
   the float export stores them densely. CSR packing keeps only the non-zeros.

Quantization is symmetric and per-row:

    scale_r = max(|W[r, :]|) / qmax          (qmax = 127 for int8, 32767 for int16)
    q[r, c] = clip(round(W[r, c] / scale_r), -qmax, qmax)
    W_hat[r, c] = q[r, c] * scale_r

Per-row scales matter for the readout matrix, whose three rows have quite different
magnitudes; a single per-tensor scale would crush the smaller rows. A row that is
entirely zero gets scale 1.0 so the reconstruction stays exactly zero.

Feature mean/std stay float32: they are only 12 values (48 bytes) and they sit directly
in the normalization path where error would propagate into every downstream term.

Usage:
    python -m src.quantize                       # ablation on the v2 test split
    python -m src.quantize --max-rows-per-station 20000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .esn import EchoStateClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "artifacts" / "models" / "esn_model.npz"
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "reports" / "quantization_metrics.json"


@dataclass(frozen=True)
class QuantSpec:
    """bits is 8 or 16; sparse_reservoir packs the reservoir matrix as CSR."""

    bits: int = 8
    sparse_reservoir: bool = True

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1

    @property
    def dtype(self) -> type:
        return np.int8 if self.bits == 8 else np.int16

    @property
    def name(self) -> str:
        return f"int{self.bits}" + ("_sparse" if self.sparse_reservoir else "_dense")


def quantize_per_row(weights: np.ndarray, spec: QuantSpec) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row quantization. Returns (integer codes, float scales)."""
    matrix = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    magnitude = np.abs(matrix).max(axis=1)
    # An all-zero row would divide by zero; scale 1.0 reconstructs it exactly as zero.
    scales = np.where(magnitude > 0, magnitude / spec.qmax, 1.0)
    codes = np.round(matrix / scales[:, None])
    codes = np.clip(codes, -spec.qmax, spec.qmax).astype(spec.dtype)
    return codes, scales.astype(np.float32)


def dequantize_per_row(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (codes.astype(np.float32) * scales[:, None].astype(np.float32)).astype(np.float32)


def pack_csr(codes: np.ndarray, scales: np.ndarray) -> dict:
    """CSR packing of a quantized matrix, keeping only structurally non-zero entries.

    Zeros here are the reservoir's connectivity mask, not quantization underflow, so
    dropping them is lossless with respect to the float model.
    """
    rows, cols = codes.shape
    if cols > 256:
        raise ValueError(f"column indices are packed as uint8; got {cols} columns")
    row_ptr = [0]
    values: list[int] = []
    indices: list[int] = []
    for row in range(rows):
        nonzero = np.flatnonzero(codes[row])
        indices.extend(int(c) for c in nonzero)
        values.extend(int(v) for v in codes[row, nonzero])
        row_ptr.append(len(values))
    return {
        "values": np.asarray(values, dtype=codes.dtype),
        "indices": np.asarray(indices, dtype=np.uint8),
        "row_ptr": np.asarray(row_ptr, dtype=np.uint16),
        "scales": scales,
        "shape": (rows, cols),
    }


def unpack_csr(packed: dict) -> np.ndarray:
    rows, cols = packed["shape"]
    dense = np.zeros((rows, cols), dtype=packed["values"].dtype)
    row_ptr = packed["row_ptr"]
    for row in range(rows):
        span = slice(int(row_ptr[row]), int(row_ptr[row + 1]))
        dense[row, packed["indices"][span].astype(int)] = packed["values"][span]
    return dense


class QuantizedESN:
    """Quantized mirror of EchoStateClassifier with identical inference semantics.

    Holds integer codes plus scales, and reconstructs dequantized float matrices for
    computation, which is exactly what the C implementation does per row at runtime.
    """

    def __init__(self, model: EchoStateClassifier, spec: QuantSpec):
        self.config = model.config
        self.spec = spec
        self.alert_threshold = model.alert_threshold
        self.feature_mean = model.feature_mean.astype(np.float32)
        self.feature_std = model.feature_std.astype(np.float32)

        self.input_codes, self.input_scales = quantize_per_row(model.input_weights, spec)
        self.output_codes, self.output_scales = quantize_per_row(model.output_weights, spec)
        reservoir_codes, reservoir_scales = quantize_per_row(model.reservoir_weights, spec)
        self.reservoir_scales = reservoir_scales
        self.reservoir_packed = pack_csr(reservoir_codes, reservoir_scales) if spec.sparse_reservoir else None
        self.reservoir_codes = reservoir_codes

        self._input_w = dequantize_per_row(self.input_codes, self.input_scales)
        self._output_w = dequantize_per_row(self.output_codes, self.output_scales)
        codes = unpack_csr(self.reservoir_packed) if spec.sparse_reservoir else self.reservoir_codes
        self._reservoir_w = dequantize_per_row(codes, self.reservoir_scales)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, dtype=np.float32) - self.feature_mean) / self.feature_std
        return np.clip(normalized, -self.config.clip_normalized, self.config.clip_normalized)

    def predict_proba(self, features: np.ndarray, resets: np.ndarray | None = None) -> np.ndarray:
        inputs = self._normalize(features)
        n_rows = len(inputs)
        if resets is None:
            resets = np.zeros(n_rows, dtype=bool)
            if n_rows:
                resets[0] = True
        resets = np.asarray(resets, dtype=bool)

        leak = self.config.leak_rate
        state = np.zeros(self.config.reservoir_size, dtype=np.float32)
        logits = np.empty((n_rows, self.config.output_size), dtype=np.float32)
        for row in range(n_rows):
            if resets[row]:
                state = np.zeros(self.config.reservoir_size, dtype=np.float32)
            augmented = np.empty(self.config.input_size + 1, dtype=np.float32)
            augmented[0] = 1.0
            augmented[1:] = inputs[row]
            candidate = np.tanh(self._input_w @ augmented + self._reservoir_w @ state)
            state = ((1.0 - leak) * state + leak * candidate).astype(np.float32)
            design = np.concatenate([[np.float32(1.0)], inputs[row], state]).astype(np.float32)
            logits[row] = self._output_w @ design

        shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)

    def size_bytes(self) -> dict:
        """Flash footprint of the model constants, excluding code."""
        item = self.spec.bits // 8
        input_bytes = self.input_codes.size * item + self.input_scales.size * 4
        output_bytes = self.output_codes.size * item + self.output_scales.size * 4
        if self.spec.sparse_reservoir:
            packed = self.reservoir_packed
            reservoir_bytes = (
                packed["values"].size * item
                + packed["indices"].size * 1
                + packed["row_ptr"].size * 2
                + packed["scales"].size * 4
            )
            nonzero = int(packed["values"].size)
        else:
            reservoir_bytes = self.reservoir_codes.size * item + self.reservoir_scales.size * 4
            nonzero = int(np.count_nonzero(self.reservoir_codes))
        norm_bytes = (self.feature_mean.size + self.feature_std.size) * 4
        return {
            "input_weights": input_bytes,
            "reservoir_weights": reservoir_bytes,
            "output_weights": output_bytes,
            "normalization": norm_bytes,
            "total": input_bytes + reservoir_bytes + output_bytes + norm_bytes,
            "reservoir_nonzero": nonzero,
        }


def float_size_bytes(model: EchoStateClassifier) -> dict:
    input_bytes = model.input_weights.size * 4
    reservoir_bytes = model.reservoir_weights.size * 4
    output_bytes = model.output_weights.size * 4
    norm_bytes = (model.feature_mean.size + model.feature_std.size) * 4
    return {
        "input_weights": input_bytes,
        "reservoir_weights": reservoir_bytes,
        "output_weights": output_bytes,
        "normalization": norm_bytes,
        "total": input_bytes + reservoir_bytes + output_bytes + norm_bytes,
        "reservoir_nonzero": int(np.count_nonzero(model.reservoir_weights)),
    }


def _alert_scores(probabilities: np.ndarray) -> np.ndarray:
    return probabilities[:, 1:].sum(axis=1)


def compare(model_path: Path, max_rows_per_station: int, report_path: Path) -> dict:
    """Ablation: float baseline versus each quantization variant on the v2 test split."""
    from .metrics import binary_alert_metrics, event_metrics
    from .ml_data import feature_matrix, load_split

    model = EchoStateClassifier.load(model_path)
    print(f"Loaded {model_path} (reservoir={model.config.reservoir_size}, threshold={model.alert_threshold})")

    test = load_split("test", max_rows_per_station=max_rows_per_station)
    features = feature_matrix(test)
    resets = test["sequence_reset"].to_numpy()
    truth = test["label"].ne("Normal").to_numpy()
    print(f"Test rows: {len(test)}, alert rows: {int(truth.sum())}")

    baseline_proba = model.predict_proba(features, resets)
    baseline_alert = _alert_scores(baseline_proba) >= model.alert_threshold
    results = {
        "model": str(model_path),
        "test_rows": int(len(test)),
        "alert_threshold": float(model.alert_threshold),
        "variants": {},
    }

    def record(name: str, proba: np.ndarray, sizes: dict) -> None:
        alert = _alert_scores(proba) >= model.alert_threshold
        entry = {
            "size_bytes": sizes,
            "alert": binary_alert_metrics(truth, alert),
            "events": event_metrics(test, alert),
            "max_abs_prob_error": float(np.abs(proba - baseline_proba).max()),
            "alert_decision_agreement": float(np.mean(alert == baseline_alert)),
        }
        results["variants"][name] = entry

    record("float32", baseline_proba, float_size_bytes(model))
    for spec in (
        QuantSpec(bits=16, sparse_reservoir=True),
        QuantSpec(bits=8, sparse_reservoir=True),
        QuantSpec(bits=8, sparse_reservoir=False),
    ):
        quantized = QuantizedESN(model, spec)
        record(spec.name, quantized.predict_proba(features, resets), quantized.size_bytes())

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    header = f"{'variant':<16}{'bytes':>8}{'vs float':>10}{'prec':>8}{'rec':>8}{'FPR':>9}{'ev.rec':>8}{'agree':>8}{'maxerr':>10}"
    print("\n" + header)
    print("-" * len(header))
    float_total = results["variants"]["float32"]["size_bytes"]["total"]
    for name, entry in results["variants"].items():
        total = entry["size_bytes"]["total"]
        alert = entry["alert"]
        print(
            f"{name:<16}{total:>8}{float_total / total:>9.2f}x"
            f"{alert['precision']:>8.4f}{alert['recall']:>8.4f}{alert['false_positive_rate']:>9.5f}"
            f"{entry['events']['event_recall']:>8.4f}{entry['alert_decision_agreement']:>8.4f}"
            f"{entry['max_abs_prob_error']:>10.2e}"
        )
    print(f"\nWrote {report_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-rows-per-station", type=int, default=80000)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    compare(args.model, args.max_rows_per_station, args.report)


if __name__ == "__main__":
    main()
