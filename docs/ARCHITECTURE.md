# Software and System Architecture

## End-to-end path

```text
asset NTC ---- ADC --+                    +--> state/confidence
                    |                    |
ambient NTC -- ADC --+--> features --> ESN --> radio policy --> LoRa packet
                                           |                       |
                                      local history          custom receiver
                                                                  |
                                                             USB serial
                                                                  |
                                                        logger / dashboard
```

The sensor node samples both thermistors on the same clock. Firmware converts ADC codes
to Celsius, rejects impossible values, updates a fixed-size history, computes relative
thermal features, runs the ESN, applies temporal persistence, and transmits only on state
change, abnormal confidence, heartbeat, or fault.

## Software boundaries

| Area | Modules | Contract |
|---|---|---|
| Data ingestion | `src/ingest.py` | Raw public data to normalized UTC station series |
| Simulation | `src/synthetic_v2.py` | Ambient series to labeled asset/fault sequences |
| Features | `src/features.py`, `src/ml_data.py` | Stable MCU-computable feature vector |
| Model | `src/esn.py`, `src/train_esn.py` | Three probabilities and class state |
| Export | `src/export_c.py` | Python model to deterministic C constants |
| Embedded reference | `firmware/` | No-allocation features, inference, policy, protocol |
| Link protocol | `src/protocol.py` | Versioned payload and CRC contract |
| Receiver | `src/receiver.py` | USB framing, validation, CSV logging |
| Dashboard | `src/dashboard.py`, `web/dashboard.html` | Local telemetry API and monitor |
| Evaluation | `src/evaluate.py`, `src/metrics.py` | Sample, event, and radio metrics |

## On-device model boundary

The model consumes six values in this exact order:

1. ambient temperature C
2. asset-minus-ambient delta C
3. asset slope over 1 minute, C/min
4. asset slope over 5 minutes, C/min
5. asset slope over 15 minutes, C/min
6. asset variance over 30 minutes, C squared

Training exports means and standard deviations. The MCU applies the same normalization,
clips each normalized input to `[-6, 6]`, advances the reservoir, and computes three
linear readout logits. Softmax is only needed for confidence; the class can be selected
from logits directly.

## Reliability behavior

- Sensor-open, sensor-short, NaN, impossible temperature, or stale history bypasses ML
  and emits a sensor-fault state.
- A sequence number exposes packet loss and receiver resets.
- CRC-16/CCITT rejects serial/radio corruption.
- Model and protocol versions are explicit.
- State persistence prevents one noisy sample from repeatedly waking the radio.
- A periodic heartbeat proves the node is alive even when normal.
