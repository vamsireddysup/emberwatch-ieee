import unittest

from src.protocol import (
    FLAG_ALERT,
    FLAG_MODEL_VALID,
    STATE_ANOMALY,
    Telemetry,
    decode_receiver_line,
    encode_receiver_line,
)


class ProtocolTest(unittest.TestCase):
    def test_round_trip(self):
        source = Telemetry(42, 65530, 123456, 63.42, -5.25, 0.91, STATE_ANOMALY, 3190, FLAG_ALERT | FLAG_MODEL_VALID)
        decoded = Telemetry.decode(source.encode())
        self.assertEqual(decoded.device_id, source.device_id)
        self.assertAlmostEqual(decoded.asset_temp_c, source.asset_temp_c, places=2)
        self.assertAlmostEqual(decoded.confidence, source.confidence, places=2)

    def test_crc_rejects_corruption(self):
        packet = bytearray(Telemetry(1, 2, 3, 30, 20, 0.5, 0, 3300).encode())
        packet[12] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            Telemetry.decode(bytes(packet))

    def test_receiver_line(self):
        source = Telemetry(7, 9, 100, 45.5, 21.2, 0.8, 1, 3250)
        decoded, rssi, snr = decode_receiver_line(encode_receiver_line(source, -97, 6.25))
        self.assertEqual(decoded.sequence, 9)
        self.assertEqual(rssi, -97)
        self.assertAlmostEqual(snr, 6.2)


if __name__ == "__main__":
    unittest.main()
