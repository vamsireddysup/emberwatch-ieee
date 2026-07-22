# Telemetry and Receiver Protocol

## Radio payload version 1

All multi-byte values are little-endian. Total length is 22 bytes.

| Offset | Type | Field |
|---:|---|---|
| 0 | u8 | magic `0xE7` |
| 1 | u8 | protocol version `1` |
| 2 | u8 | message type: `1` telemetry |
| 3 | u8 | flags: alert, battery-low, sensor-fault, model-valid |
| 4 | u16 | device ID |
| 6 | u16 | sequence number, wrapping |
| 8 | u32 | node uptime seconds |
| 12 | i16 | asset temperature x100 C |
| 14 | i16 | ambient temperature x100 C |
| 16 | u8 | confidence 0-255 |
| 17 | u8 | state: 0 Normal, 1 Warming, 2 Anomaly, 3 SensorFault |
| 18 | u16 | battery millivolts |
| 20 | u16 | CRC-16/CCITT-FALSE over bytes 0-19 |

Temperature uses hundredths internally so the packet preserves calibration information;
the UI may display the competition-required tenths resolution.

## USB serial frame

The custom receiver emits one ASCII line per valid LoRa reception:

```text
EW1,<44 lowercase hex characters>,<rssi_dbm>,<snr_db>\n
```

Example receiver software validates framing, length, magic, version, and CRC before
logging. Invalid lines are counted and reported but never treated as telemetry.

## Compatibility

Any layout change increments the protocol version. New message types may reuse the
common four-byte prefix but must define their own fixed length and CRC location.
