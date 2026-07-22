# Hardware and End-to-End Validation Plan

## 1. Temperature calibration

Mount asset and ambient thermistors beside a traceable reference probe without thermal
contact between the two channels. Exercise at least five stable points spanning the
expected range, plus slow ramps. Record raw ADC code and supply/reference voltage.

Report bias, mean absolute error, RMSE, maximum absolute error, repeatability, and 0.1 C
display resolution. Split calibration points from final verification points. Repeat after
enclosure assembly because contact pressure and self-heating can change error.

## 2. Feature parity

Stream timestamp, converted temperatures, and all six C features over USB for a fixed
temperature sequence. Feed the same temperatures into a Python fixture. Require each
feature to agree within a documented floating-point tolerance before testing the model.

The current model must be called every five minutes. A different cadence requires a new
dataset and model because slopes and reservoir memory are time-dependent.

## 3. Controlled fault rig

Use a safe low-voltage heater or power resistor attached to a representative thermal
mass. Do not create hazardous mains faults. Run independent trials for:

- normal ambient ramps with no asset-specific heating;
- ordinary load steps that should remain normal;
- gradual local heating representing a resistive connection;
- reduced cooling/insulation around the asset sensor;
- accelerating heating profile;
- ambient sensor and asset sensor open/short/disconnect faults.

Keep entire trials out of training when reporting final detection. Report event detection,
time to first warning, time before the chosen unsafe reference point, false episodes per
device-hour, and missed trials. Save all raw logs, not only plots.

## 4. Power measurement

Measure voltage and current with enough bandwidth to capture radio bursts. Integrate
energy for sleep, ADC/sensor settling, feature calculation, ESN inference, packet encode,
LoRa transmit, and receiver operation.

```text
energy_Wh = sum(voltage_V * current_A * duration_s) / 3600
estimated_life_h = usable_battery_Wh / average_power_W
```

Repeat normal heartbeat, frequent false-alert, and real-alert scenarios. The software
metric "periodic transmissions avoided" is not itself battery-life improvement; convert
actual avoided packets through measured transmit energy and include sleep/self-discharge.

## 5. RF and reader

Test line-of-sight and representative obstructed placements. Record frequency, bandwidth,
spreading factor, coding rate, transmit power, antenna, packet size, distance, RSSI, SNR,
packet delivery ratio, and jurisdictional constraints. Demonstrate sequence-gap reporting,
bad-CRC rejection, receiver reset recovery, and USB logging.

## 6. Acceptance gates

- No unexplained Python/C model mismatch.
- No dynamic allocation in the deployed feature/model/policy path.
- Sensor faults bypass ML and create an explicit fault packet.
- Validation threshold remains frozen before final test trials.
- Every headline number points to a raw log, method, configuration, and script.
- Synthetic, rig, and field results are labeled separately.
