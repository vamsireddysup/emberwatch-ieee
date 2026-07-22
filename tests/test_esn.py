import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.esn import ESNConfig, EchoStateClassifier


class ESNTest(unittest.TestCase):
    def test_fit_save_load_is_deterministic(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(180, 6)).astype(np.float32)
        labels = np.repeat([0, 1, 2], 60)
        resets = np.zeros(len(labels), dtype=bool)
        resets[[0, 60, 120]] = True
        model = EchoStateClassifier(ESNConfig(input_size=6, reservoir_size=12, seed=3))
        model.fit(features, labels, resets)
        before = model.predict_proba(features, resets)
        self.assertTrue(np.isfinite(model.output_weights).all())
        self.assertTrue(np.isfinite(before).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = EchoStateClassifier.load(path)
            after = loaded.predict_proba(features, resets)
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(after.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
