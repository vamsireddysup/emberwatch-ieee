"""Emit valid receiver USB lines for integration testing without radio hardware."""

from __future__ import annotations

import argparse
import math
import time

from .protocol import FLAG_ALERT, FLAG_MODEL_VALID, Telemetry, encode_receiver_line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--device-id", type=int, default=1)
    args = parser.parse_args()
    for sequence in range(args.count):
        excess = max(0.0, sequence - args.count * 0.45) * 0.8
        asset = 36.0 + 1.5 * math.sin(sequence / 4) + excess
        ambient = 24.0 + 0.5 * math.sin(sequence / 8)
        state = 2 if excess > 6 else (1 if excess > 2 else 0)
        confidence = 0.55 if state == 0 else min(0.99, 0.60 + excess / 20)
        flags = FLAG_MODEL_VALID | (FLAG_ALERT if state else 0)
        packet = Telemetry(
            device_id=args.device_id,
            sequence=sequence,
            uptime_s=sequence * 300,
            asset_temp_c=asset,
            ambient_temp_c=ambient,
            confidence=confidence,
            state=state,
            battery_mv=3290 - sequence,
            flags=flags,
        )
        print(encode_receiver_line(packet, -92, 7.5), end="", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
