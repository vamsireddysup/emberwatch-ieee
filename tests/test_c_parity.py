import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.esn import ESNConfig, EchoStateClassifier
from src.export_c import export_model

REPO_ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(shutil.which("cc"), "C compiler is required for Python/C parity")
class CParityTest(unittest.TestCase):
    def test_exported_inference_matches_python(self):
        rng = np.random.default_rng(8)
        train_features = rng.normal(size=(150, 6)).astype(np.float32)
        train_labels = np.repeat([0, 1, 2], 50)
        resets = np.zeros(150, dtype=bool)
        resets[0] = True
        model = EchoStateClassifier(ESNConfig(input_size=6, reservoir_size=10, seed=9))
        model.fit(train_features, train_labels, resets)

        sequence = rng.normal(size=(12, 6)).astype(np.float32)
        python_probabilities = model.predict_proba(sequence)
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            export_model(model, temp / "emberwatch_model.h")
            wrapper = temp / "wrapper.c"
            wrapper.write_text(
                '#include "emberwatch_inference.h"\n'
                "static ew_model_state_t state;\n"
                "void reset_model(void) { ew_model_reset(&state); }\n"
                "void predict_model(const float *features, float *probabilities) {\n"
                "  ew_model_output_t output = ew_model_predict(&state, features);\n"
                "  for (int i = 0; i < EW_MODEL_OUTPUT_SIZE; ++i) probabilities[i] = output.probabilities[i];\n"
                "}\n",
                encoding="ascii",
            )
            library = temp / "libemberwatch.dylib"
            compile_result = subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-shared",
                    "-fPIC",
                    f"-I{temp}",
                    f"-I{REPO_ROOT / 'firmware' / 'include'}",
                    str(REPO_ROOT / "firmware" / "src" / "emberwatch_inference.c"),
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
            c_probabilities = []
            for row in sequence:
                output = np.zeros(3, dtype=np.float32)
                c_model.predict_model(
                    row.ctypes.data_as(pointer), output.ctypes.data_as(pointer)
                )
                c_probabilities.append(output)
        np.testing.assert_allclose(np.asarray(c_probabilities), python_probabilities, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
