import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.esn import ESNConfig, EchoStateClassifier
from src.export_c_quant import export_quantized
from src.quantize import QuantSpec, QuantizedESN, pack_csr, quantize_per_row, unpack_csr

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fitted_model(reservoir_size: int = 10) -> EchoStateClassifier:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(150, 6)).astype(np.float32)
    labels = np.repeat([0, 1, 2], 50)
    resets = np.zeros(150, dtype=bool)
    resets[0] = True
    model = EchoStateClassifier(ESNConfig(input_size=6, reservoir_size=reservoir_size, seed=9))
    return model.fit(features, labels, resets)


class QuantizationTest(unittest.TestCase):
    def test_round_trip_error_is_bounded_by_step_size(self):
        rng = np.random.default_rng(3)
        weights = rng.normal(size=(7, 11)).astype(np.float32)
        for bits in (8, 16):
            codes, scales = quantize_per_row(weights, QuantSpec(bits=bits))
            reconstructed = codes.astype(np.float32) * scales[:, None]
            # Symmetric round-to-nearest cannot exceed half a step on any row.
            self.assertTrue(np.all(np.abs(reconstructed - weights) <= scales[:, None] / 2 + 1e-6))

    def test_all_zero_row_survives_quantization(self):
        weights = np.zeros((3, 5), dtype=np.float32)
        weights[1] = 0.4
        codes, scales = quantize_per_row(weights, QuantSpec(bits=8))
        reconstructed = codes.astype(np.float32) * scales[:, None]
        self.assertTrue(np.all(reconstructed[0] == 0.0))
        self.assertTrue(np.all(reconstructed[2] == 0.0))

    def test_csr_packing_is_lossless(self):
        model = _fitted_model()
        codes, scales = quantize_per_row(model.reservoir_weights, QuantSpec(bits=8))
        np.testing.assert_array_equal(unpack_csr(pack_csr(codes, scales)), codes)

    def test_sparse_packing_is_smaller_than_float(self):
        model = _fitted_model(reservoir_size=48)
        quantized = QuantizedESN(model, QuantSpec(bits=8, sparse_reservoir=True))
        sizes = quantized.size_bytes()
        float_bytes = (
            model.input_weights.size + model.reservoir_weights.size + model.output_weights.size
        ) * 4
        self.assertLess(sizes["total"], float_bytes)
        self.assertLess(sizes["reservoir_nonzero"], model.reservoir_weights.size)

    def test_quantized_probabilities_track_float_model(self):
        model = _fitted_model()
        rng = np.random.default_rng(21)
        sequence = rng.normal(size=(24, 6)).astype(np.float32)
        reference = model.predict_proba(sequence)
        for bits, tolerance in ((16, 5e-4), (8, 6e-2)):
            quantized = QuantizedESN(model, QuantSpec(bits=bits, sparse_reservoir=True))
            np.testing.assert_allclose(
                quantized.predict_proba(sequence), reference, atol=tolerance
            )


@unittest.skipUnless(shutil.which("cc"), "C compiler is required for Python/C parity")
class QuantizedCParityTest(unittest.TestCase):
    def _run_c_model(self, model: EchoStateClassifier, spec: QuantSpec, sequence: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            export_quantized(model, spec, temp / "emberwatch_model_q.h")
            wrapper = temp / "wrapper.c"
            wrapper.write_text(
                '#include "emberwatch_inference_q.h"\n'
                "static ew_q_state_t state;\n"
                "void reset_model(void) { ew_q_reset(&state); }\n"
                "void predict_model(const float *features, float *probabilities) {\n"
                "  ew_q_output_t output = ew_q_predict(&state, features);\n"
                "  for (int i = 0; i < EW_Q_OUTPUT_SIZE; ++i) probabilities[i] = output.probabilities[i];\n"
                "}\n",
                encoding="ascii",
            )
            library = temp / "libemberwatch_q.dylib"
            compile_result = subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-shared",
                    "-fPIC",
                    f"-I{temp}",
                    f"-I{REPO_ROOT / 'firmware' / 'include'}",
                    str(REPO_ROOT / "firmware" / "src" / "emberwatch_inference_q.c"),
                    str(wrapper),
                    "-lm",
                    "-o",
                    str(library),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            c_model = ctypes.CDLL(str(library))
            pointer = ctypes.POINTER(ctypes.c_float)
            c_model.predict_model.argtypes = [pointer, pointer]
            c_model.reset_model()
            probabilities = []
            for row in sequence:
                out = np.zeros(3, dtype=np.float32)
                c_model.predict_model(row.ctypes.data_as(pointer), out.ctypes.data_as(pointer))
                probabilities.append(out.copy())
        return np.asarray(probabilities)

    def test_quantized_c_matches_quantized_python(self):
        model = _fitted_model()
        rng = np.random.default_rng(8)
        sequence = rng.normal(size=(12, 6)).astype(np.float32)
        for bits in (8, 16):
            spec = QuantSpec(bits=bits, sparse_reservoir=True)
            expected = QuantizedESN(model, spec).predict_proba(sequence)
            actual = self._run_c_model(model, spec, sequence)
            # Same arithmetic in both; only float association order differs.
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
