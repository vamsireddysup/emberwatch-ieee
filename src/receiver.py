"""Read custom receiver USB frames, validate packets, and append telemetry to CSV."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .protocol import decode_receiver_line

FIELDS = [
    "received_at",
    "device_id",
    "sequence",
    "uptime_s",
    "asset_temp_c",
    "ambient_temp_c",
    "delta_c",
    "confidence",
    "state",
    "battery_mv",
    "flags",
    "rssi_dbm",
    "snr_db",
]


def parse_line(line: str) -> dict:
    telemetry, rssi, snr = decode_receiver_line(line)
    row = asdict(telemetry)
    row.update(
        {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "delta_c": round(telemetry.asset_temp_c - telemetry.ambient_temp_c, 2),
            "rssi_dbm": rssi,
            "snr_db": snr,
        }
    )
    return row


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row[field] for field in FIELDS})


def run_stream(lines, output: Path) -> tuple[int, int]:
    valid = invalid = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            row = parse_line(line)
            append_row(output, row)
            valid += 1
            print(
                f"device={row['device_id']} seq={row['sequence']} state={row['state']} "
                f"asset={row['asset_temp_c']:.2f}C ambient={row['ambient_temp_c']:.2f}C "
                f"confidence={row['confidence']:.2f}"
            )
        except ValueError as exc:
            invalid += 1
            print(f"invalid frame: {exc}", file=sys.stderr)
    return valid, invalid


def serial_lines(port: str, baud: int):
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required for --port; install requirements.txt") from exc
    with serial.Serial(port, baudrate=baud, timeout=1) as connection:
        while True:
            raw = connection.readline()
            if raw:
                yield raw.decode("ascii", errors="replace")
            else:
                time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Serial device; omit to read receiver lines from stdin")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--output", type=Path, default=Path("artifacts/telemetry/receiver_log.csv"))
    args = parser.parse_args()
    lines = serial_lines(args.port, args.baud) if args.port else sys.stdin
    try:
        run_stream(lines, args.output)
    except KeyboardInterrupt:
        print("receiver stopped")


if __name__ == "__main__":
    main()
