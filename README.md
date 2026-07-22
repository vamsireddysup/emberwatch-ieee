# EmberWatch

EmberWatch is a low-power temperature monitoring system for transformers, reclosers, and
other grid assets in wildfire-prone regions. Its purpose is to detect abnormal heating at
the potential ignition source before a fire starts. It is not a forest fire sensor.

Two NTC thermistors measure asset and ambient temperature. A small echo state network
(ESN) runs on the sensor MCU, classifies each thermal sequence as `Normal`, `Warming`, or
`Anomaly`, and gates LoRa transmissions. A custom reader decodes the radio payload and
forwards CRC-validated telemetry over USB serial to a host logger.

## Start here

- [Current project state](docs/PROJECT_STATE.md)
- [System architecture](docs/ARCHITECTURE.md)
- [ML design and commands](docs/ML_PIPELINE.md)
- [Hardware integration](docs/HARDWARE_INTEGRATION.md)
- [Hardware setup and execution guide](docs/HARDWARE_SETUP_GUIDE.md)
- [Hardware and end-to-end validation](docs/VALIDATION_PLAN.md)
- [Telemetry protocol](docs/PROTOCOL.md)
- [Software verification record](docs/TESTING.md)
- [Competition requirements](docs/COMPETITION_REQUIREMENTS.md)
- [Claude/Codex workflow](docs/COLLABORATION.md)

## Dataset architecture

Real ambient measurements come from five Oregon and Northern California stations. The
canonical dataset-v2 pipeline in `src/calibrate_ett.py` and
`src/synthesize_thermal.py` calibrates noise and load shapes from real ETT transformer
measurements, simulates a first-order RC thermal system and healthy twin, and injects
overload, loose-connection, cooling-degradation, and thermal-runaway events. It produces
chronological 2020-2021 train, 2022 validation, and 2023 test data. See
`data/synthetic_v2/README.md` for the model and limitations.

The original NAB machine-temperature pairing remains as a real-machine sanity check, but
it is not a physically paired transformer/ambient dataset. `src/synthetic_v2.py` is an
alternate experimental generator that writes under `artifacts/`; it does not replace the
canonical ETT-calibrated v2 pipeline.

## Repository map

```text
src/          ingestion, simulation, features, baselines, ESN, export, receiver
firmware/     portable C inference and packet codec
tests/        Python tests plus C and Python/C parity checks
docs/         architecture, decisions, integration, requirements, generated results
archives/     preserved superseded material and archive index
data/         local raw/processed datasets; intentionally gitignored
artifacts/    generated models, reports, telemetry logs; intentionally gitignored
```

No project file is deleted when it becomes obsolete. See `archives/INDEX.md`.

## Install

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Train and compare

The current larger candidate uses five stations, a 48-unit reservoir, and at most 80,000
chronology-preserving rows per station in each time split:

```bash
./venv/bin/python -m src.train_esn --max-rows-per-station 80000 --reservoir-size 48
./venv/bin/python -m src.baselines_v2 --max-rows-per-station 80000
```

The trainer fits on 2020-2021, selects its alert threshold on 2022 validation data, and
reports 2023 test performance. It writes an ignored NumPy artifact, a detailed JSON
report, [the tracked result summary](docs/generated/ML_RESULTS.md), and
`firmware/generated/emberwatch_model.h`.

Synthetic results are engineering validation only. The NAB set is a real-machine sanity
check but is not physically paired transformer/ambient data. Final claims require the
controlled heating rig and physical sensor calibration described in the integration doc.

## Receiver without hardware

```bash
./venv/bin/python -m src.simulate_receiver --count 20 --interval 0.2 \
  | ./venv/bin/python -m src.receiver --output artifacts/telemetry/demo.csv
```

With the custom receiver connected:

```bash
./venv/bin/python -m src.receiver --port /dev/cu.usbmodemXXXX
```

Run the local dashboard against the receiver log:

```bash
./venv/bin/python -m src.dashboard --log artifacts/telemetry/receiver_log.csv
```

The receiver-to-host choice is USB for the prototype. A later Raspberry Pi or gateway
can bridge the same log to Wi-Fi or Ethernet without changing the sensor radio packet.

## Reproduce synthetic data

Regenerate the canonical ETT calibration and physics-grounded dataset-v2 files:

```bash
./venv/bin/python src/calibrate_ett.py
./venv/bin/python src/synthesize_thermal.py --station all --seed 42
```

Generate feature files with the model schema using the per-station commands documented
in `data/synthetic_v2/README.md` and `src/features.py`. For an independent experimental
simulation under `artifacts/`:

```bash
./venv/bin/python -m src.synthetic_v2
```

See [synthetic data assumptions](docs/SYNTHETIC_DATA.md) before using either generator in
a claim.

## Verify

```bash
./venv/bin/python -m py_compile src/*.py
./venv/bin/python -m unittest discover -s tests -v
mkdir -p build
cc -std=c11 -Wall -Wextra -Werror -Ifirmware/include \
  firmware/src/emberwatch_protocol.c tests/c/test_protocol.c -o build/test_protocol
./build/test_protocol
```

The test suite trains a small ESN, exports it, compiles the C inference implementation,
and checks Python/C probabilities on the same sequence.
