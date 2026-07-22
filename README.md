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
- [Hardware and end-to-end validation](docs/VALIDATION_PLAN.md)
- [Telemetry protocol](docs/PROTOCOL.md)
- [Software verification record](docs/TESTING.md)
- [Competition requirements](docs/COMPETITION_REQUIREMENTS.md)
- [Claude/Codex workflow](docs/COLLABORATION.md)

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

The exact generator for the pre-existing local v2 files was not tracked. Those files are
preserved. The deterministic replacement writes to a new artifact directory by default:

```bash
./venv/bin/python -m src.synthetic_v2
```

See [synthetic data assumptions](docs/SYNTHETIC_DATA.md) before using it in a claim.

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
