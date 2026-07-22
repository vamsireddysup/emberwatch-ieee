import csv
import tempfile
import unittest
from pathlib import Path

from src.dashboard import read_telemetry


class DashboardTest(unittest.TestCase):
    def test_read_telemetry_coerces_types_and_limits_rows(self):
        fields = ["received_at", "device_id", "sequence", "uptime_s", "asset_temp_c", "ambient_temp_c", "delta_c", "confidence", "state", "battery_mv", "flags", "rssi_dbm", "snr_db"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for sequence in range(3):
                    writer.writerow(dict(zip(fields, ["2026-01-01T00:00:00+00:00", 1, sequence, 0, 30.5, 20.2, 10.3, 0.8, 0, 3300, 8, -90, 7.5])))
            rows = read_telemetry(path, limit=2)
        self.assertEqual([row["sequence"] for row in rows], [1, 2])
        self.assertIsInstance(rows[0]["asset_temp_c"], float)


if __name__ == "__main__":
    unittest.main()
