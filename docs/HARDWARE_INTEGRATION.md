# Hardware Integration Contract

## Measurement chain

For a low-side NTC divider with pull-up resistor `R_FIXED`, ADC full-scale code
`ADC_MAX`, and measured code `adc`:

```text
R_NTC = R_FIXED * adc / (ADC_MAX - adc)
1/T_K = 1/T0_K + ln(R_NTC/R0) / BETA
T_C = T_K - 273.15
```

The formula reverses for the opposite divider topology. Use measured resistor values and
either the thermistor beta equation or manufacturer Steinhart-Hart coefficients. Do not
freeze firmware constants until the hardware team supplies the exact topology and parts.

## Proposed cadence

- Wake and sample both channels every 60 seconds.
- Take multiple ADC readings after settling; reject min/max and average the remainder.
- Update one-minute history each wake.
- Run feature inference when sufficient history exists.
- Require two consecutive `Warming` decisions; accept `Anomaly` immediately; require
  three confident `Normal` decisions to clear an alert.
- Transmit on state change, anomaly escalation, sensor/battery fault, or heartbeat.
- Default heartbeat: 6 hours. Final value comes from measured link and battery budget.

The current datasets are five-minute samples. Firmware can compute one-minute history and
run the model every five minutes, or the team can retrain with real one-minute rig data.
The cadence used for a deployed model must match its training metadata.

## Bring-up order

1. Stream raw ADC and converted temperatures over USB; compare against a reference probe.
2. Verify open/short detection and impossible-value handling.
3. Stream every computed feature and compare row-for-row with Python fixture outputs.
4. Run the generated ESN on an identical feature sequence in C and Python.
5. Encode a telemetry packet in firmware and decode it with `src/protocol.py`.
6. Send receiver lines over USB and log them with `src/receiver.py`.
7. Measure current for sleep, sampling, inference, receive, and transmit states.

## Hardware values still required

| Parameter | Status |
|---|---|
| MCU/board and clock | Awaiting team |
| Flash and RAM budget | Awaiting team |
| Thermistor part and coefficients | Awaiting team |
| Divider topology/resistor/tolerance | Awaiting team |
| ADC bits/reference/oversampling | Awaiting team |
| Battery chemistry/capacity | Awaiting team |
| LoRa region and stack | Awaiting team |
| Receiver board | Awaiting team |

## Physical validation dataset

Log timestamp, asset reference temperature, ambient reference temperature, both ADC
codes, converted temperatures, load/heater command, fault scenario, model state,
confidence, packet sequence, RSSI, SNR, and supply current. Keep calibration, training,
and final test runs separate. Controlled experiments should include normal ambient ramps,
load changes, loose-contact-like local heating, cooling impairment, and sensor faults.
