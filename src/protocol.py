"""Versioned EmberWatch telemetry packet and USB framing implementation."""

from __future__ import annotations

import dataclasses
import struct

MAGIC = 0xE7
VERSION = 1
MESSAGE_TELEMETRY = 1
PACKET_NO_CRC = struct.Struct("<BBBBHHIhhBBH")
PACKET = struct.Struct("<BBBBHHIhhBBHH")

FLAG_ALERT = 1 << 0
FLAG_BATTERY_LOW = 1 << 1
FLAG_SENSOR_FAULT = 1 << 2
FLAG_MODEL_VALID = 1 << 3

STATE_NORMAL = 0
STATE_WARMING = 1
STATE_ANOMALY = 2
STATE_SENSOR_FAULT = 3


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    crc = initial
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclasses.dataclass(frozen=True)
class Telemetry:
    device_id: int
    sequence: int
    uptime_s: int
    asset_temp_c: float
    ambient_temp_c: float
    confidence: float
    state: int
    battery_mv: int
    flags: int = FLAG_MODEL_VALID

    def encode(self) -> bytes:
        asset = round(self.asset_temp_c * 100)
        ambient = round(self.ambient_temp_c * 100)
        confidence = round(max(0.0, min(1.0, self.confidence)) * 255)
        body = PACKET_NO_CRC.pack(
            MAGIC,
            VERSION,
            MESSAGE_TELEMETRY,
            self.flags,
            self.device_id,
            self.sequence,
            self.uptime_s,
            asset,
            ambient,
            confidence,
            self.state,
            self.battery_mv,
        )
        return body + struct.pack("<H", crc16_ccitt(body))

    @classmethod
    def decode(cls, packet: bytes) -> "Telemetry":
        if len(packet) != PACKET.size:
            raise ValueError(f"Expected {PACKET.size} bytes, got {len(packet)}")
        fields = PACKET.unpack(packet)
        magic, version, message_type, flags = fields[:4]
        if magic != MAGIC or version != VERSION or message_type != MESSAGE_TELEMETRY:
            raise ValueError(f"Unsupported header magic={magic:#x}, version={version}, type={message_type}")
        expected = crc16_ccitt(packet[:-2])
        if fields[-1] != expected:
            raise ValueError(f"CRC mismatch: received {fields[-1]:#06x}, expected {expected:#06x}")
        return cls(
            device_id=fields[4],
            sequence=fields[5],
            uptime_s=fields[6],
            asset_temp_c=fields[7] / 100.0,
            ambient_temp_c=fields[8] / 100.0,
            confidence=fields[9] / 255.0,
            state=fields[10],
            battery_mv=fields[11],
            flags=flags,
        )


def encode_receiver_line(telemetry: Telemetry, rssi_dbm: int, snr_db: float) -> str:
    return f"EW1,{telemetry.encode().hex()},{rssi_dbm},{snr_db:.1f}\n"


def decode_receiver_line(line: str) -> tuple[Telemetry, int, float]:
    parts = line.strip().split(",")
    if len(parts) != 4 or parts[0] != "EW1":
        raise ValueError("Receiver line must be EW1,<hex>,<rssi>,<snr>")
    try:
        packet = bytes.fromhex(parts[1])
        rssi = int(parts[2])
        snr = float(parts[3])
    except ValueError as exc:
        raise ValueError(f"Invalid receiver field: {exc}") from exc
    return Telemetry.decode(packet), rssi, snr
