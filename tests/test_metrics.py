import unittest

import numpy as np
import pandas as pd

from src.metrics import apply_alert_policy, binary_alert_metrics, event_metrics


class MetricsTest(unittest.TestCase):
    def test_binary_metrics(self):
        result = binary_alert_metrics([0, 0, 1, 1], [0, 1, 1, 0])
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertAlmostEqual(result["false_positive_rate"], 0.5)

    def test_policy_persistence(self):
        probabilities = np.array([[0.9, 0.1, 0], [0.3, 0.7, 0], [0.2, 0.8, 0], [0.9, 0.1, 0]])
        states, _ = apply_alert_policy(probabilities, 0.6, heartbeat_samples=100)
        self.assertEqual(states.tolist(), [0, 0, 1, 1])

    def test_event_metrics(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=6, freq="5min"),
                "event_id": [0, 1, 1, 1, 0, 0],
                "label": ["Normal", "Normal", "Warming", "Anomaly", "Normal", "Normal"],
            }
        )
        result = event_metrics(frame, [0, 0, 1, 1, 0, 0])
        self.assertEqual(result["events_detected"], 1)
        self.assertEqual(result["median_lead_minutes"], 5.0)


if __name__ == "__main__":
    unittest.main()
