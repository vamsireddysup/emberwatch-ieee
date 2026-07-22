# Generated Energy and Battery-Life Results

Radio-energy model output. The transmission counts are from the pipeline reports;
the hardware currents/timings are documented PLACEHOLDERS (see the table) and must
be replaced with measured values before any energy claim is final.

## Two separate claims

1. **System (gating vs always-on):** silent-unless-changed operation avoids most
   transmissions and multiplies battery life. Large, but a threshold detector also
   achieves it; it is the value of any on-device decision.
2. **AI (ESN vs threshold):** at a similar transmission budget the ESN catches more
   events and warns earlier. See `docs/generated/MODEL_COMPARISON.md`. That is the
   ESN's real advantage, not packet count.

## Battery life by policy

| Policy | Transmissions/day | Daily draw (uAh) | Battery life (days) | Years | x vs always-on |
|---|---:|---:|---:|---:|---:|
| always_on | 288.0 | 478.0 | 4237 | 11.60 | 1.0x |
| baseline_gated | 5.4 | 54.1 | 37451 | 102.53 | 8.8x |
| esn_gated | 5.8 | 54.7 | 37015 | 101.34 | 8.7x |

## Placeholder hardware assumptions

| Parameter | Value | Note |
|---|---:|---|
| Sleep current | 1.5 uA | STM32WL55 Stop2, datasheet-typical |
| Active current | 5.0 mA | MCU run at ~48 MHz for sense+inference |
| TX current | 45.0 mA | SX126x LoRa +14 dBm |
| Sense time | 20 ms | two NTC channels, settled + oversampled |
| Inference time | 5 ms | 48-unit ESN on M4, conservative cap |
| TX time | 120 ms | 22-byte payload at ~SF9/125 kHz |
| Battery capacity | 2700 mAh | e.g. 2x AA Li/FeS2 |
| Derating | 0.75 | temperature, self-discharge, cutoff |

Replace these in `HardwareAssumptions` in `src/energy_model.py` with measured
numbers from the hardware team, then rerun `make energy`.

## Reproduce

```bash
./venv/bin/python -m src.energy_model
```

Full breakdown including per-state charge is in `artifacts/reports/energy_metrics.json`.
