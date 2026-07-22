# EmberWatch Project State

Last updated: 2026-07-22 by Codex

## Objective

EmberWatch is an end-to-end, low-power system that detects abnormal heating at grid
assets such as transformers and reclosers before that heating becomes an ignition source.
Two temperature channels measure the asset and nearby ambient air. A lightweight
three-class model runs on the sensor MCU and classifies `Normal`, `Warming`, or
`Anomaly`. The result gates LoRa transmissions to reduce radio energy.

The project is not a wildfire detector. Dryad-style forest sensing and much published
LoRa work identify fire after it starts; EmberWatch targets the electrical ignition
source and its thermal precursor.

## Current repository

The tracked data pipeline downloads and processes NAB and weather data, calibrates thermal
behavior from ETT, generates five station-specific physics-grounded v2 datasets, engineers
thermal dynamics features, runs threshold baselines, and records metrics. The canonical
generator is `src/synthesize_thermal.py`; its assumptions are documented in
`data/synthetic_v2/README.md`.

The software expansion adds a multi-station ESN pipeline, an alternate experimental
generator, event and radio metrics, MCU model export, a binary telemetry protocol, a USB
receiver/logger and dashboard, host-buildable embedded code, tests, and collaboration
docs.

A later software-completion pass added: int8/int16 quantized sparse model export (1.7 KB,
event recall unchanged; `docs/generated/QUANTIZATION_RESULTS.md`); a radio-energy and
battery-life model with an editable placeholder hardware block
(`docs/generated/ENERGY_RESULTS.md`); a single comparison table across baselines and all
ESN variants (`docs/generated/MODEL_COMPARISON.md`); a per-station robustness analysis with
a working per-device threshold-trim mitigation for the RAWS-station FPR
(`docs/generated/STATION_ROBUSTNESS.md`); and a hardware setup and execution guide
(`docs/HARDWARE_SETUP_GUIDE.md`). These clarified that the transmission-reduction benefit is
a system property shared with the threshold baseline, while the ESN's distinct advantage is
event recall and lead time.

## Current model candidate

The 2026-07-22 candidate uses a 48-unit reservoir and a bounded maximum of 80,000
chronology-preserving rows per station for each split (400,000 rows total per split). The
validation-selected alert threshold is 0.70. On the synthetic-v2 2023 test sample it
achieved 0.807 alert precision, 0.472 alert recall, 0.0082 false-positive rate, 0.929
sampled-event recall, and 25-minute median lead. Detailed results are in
`docs/generated/ML_RESULTS.md`. These are synthetic engineering metrics, not field claims.

The float generated header is about 28 KB; the quantized int8 header holds the same model
in about 1.7 KB of constants with unchanged event recall, meeting the original int8/few-KB
MCU target. Reservoir state is 192 bytes. Exact linked flash, stack, latency, and energy
still require the target MCU build and measurement.

## Verified local data

- Five real ambient station streams: Hood River, Hillsboro, Troutdale, Klamath, Gerber.
- Four years of local synthetic-v2 data, about 2.08 million five-minute samples.
- Time split: 2020-2021 train, 2022 validation, 2023 test.
- 389 injected events across overload, loose connection, cooling degradation, and
  thermal runaway scenarios.
- NAB real-machine set: 22,683 feature rows but only four source anomaly timestamps;
  useful as a reality check, not sufficient proof of transformer fault performance.

## Ownership

Hardware team owns sensor selection, analog front end, PCB, power measurements, RF
range tests, enclosure, physical calibration, and demonstration assembly.

Software agents own data provenance, reproducible simulation, feature/model code,
evaluation, C export and inference reference, receiver/logger, dashboard evolution,
plots, and technical documentation. Hardware constants must be supplied by the hardware
team and captured in `docs/HARDWARE_INTEGRATION.md` before production firmware is frozen.

## Immediate next hardware inputs

- Exact MCU/development board and available flash/RAM.
- NTC thermistor part numbers, nominal resistance, beta or Steinhart-Hart coefficients,
  divider topology and resistor tolerance.
- ADC resolution, reference strategy, and whether channels are switched or continuous.
- LoRa region/frequency, stack, spreading-factor plan, and legal test limits.
- Battery chemistry/capacity and target sampling/transmission intervals.
- Receiver MCU/board and chosen USB connector.

## Known limitations

- Synthetic faults are engineering hypotheses, not field measurements. Claims must say
  simulation or synthetic validation until heated-rig and field data exist.
- NAB machine temperature and Oregon ambient temperature are unrelated physical systems.
- Climate and sensor drift are addressed through relative features and training-only
  normalization, but label thresholds and calibration still require periodic review.
- Receiver-to-network recommendation is USB serial for the competition prototype. Wi-Fi
  or Ethernet can be added at a gateway later; Bluetooth is not currently justified.

## Next milestones

1. Obtain hardware constants and create the thermistor calibration fixture.
2. Integrate generated model header into STM32 firmware following
   `docs/HARDWARE_SETUP_GUIDE.md`; Python/C host parity (float and quantized) already passes.
3. Record controlled heating data and compare it with the synthetic fault assumptions.
4. Measure energy per sample, inference, idle period, and LoRa packet, then replace the
   placeholder `HardwareAssumptions` in `src/energy_model.py` and rerun `make energy`.
5. Model-size/quantization ablation is done (`docs/generated/QUANTIZATION_RESULTS.md`); the
   Gerber/Klamath holdout FPR has a working per-device threshold-trim mitigation
   (`docs/generated/STATION_ROBUSTNESS.md`). A per-station readout retrain is the remaining
   optional robustness improvement before freezing.
6. Produce competition plots, two-page description, simulation evidence, and five-minute
   demonstration script before the 2026-09-10 deadline.
