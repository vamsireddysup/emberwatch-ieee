# Software Verification Record

Last run: 2026-07-22

## Software-completion pass (energy, comparison, robustness, quantization, hardware guide)

```bash
make test          # 15 Python tests pass; C protocol and features/policy executables pass
make export-quant  # quantized header generated; quantized C compiles under -Werror
make energy        # battery-life model runs; writes docs/generated/ENERGY_RESULTS.md
make compare       # one comparison table; writes docs/generated/MODEL_COMPARISON.md
make robustness    # per-station analysis; writes docs/generated/STATION_ROBUSTNESS.md
```

The `docs/HARDWARE_SETUP_GUIDE.md` main-loop skeleton was compiled and linked against the
real firmware modules (`emberwatch_features/inference_q/policy/protocol`) under
`-std=c11 -Wall -Wextra -Werror` and run, to confirm the integration code handed to the
hardware team is correct.

Quantization result: int8-sparse model is 1709 bytes (6.59x smaller than float) with event
recall unchanged at 0.9388 on the 400k-row test split; C<->Python parity holds for int8 and
int16 (`tests/test_quantized_parity.py`).

## Earlier run (Codex foundation)

## Automated checks

```bash
make test
cc -std=c11 -Wall -Wextra -Werror -Ifirmware/include -Ifirmware/generated \
  -c firmware/src/emberwatch_inference.c -o build/emberwatch_inference.o
```

Results: eight Python tests passed; C protocol and C feature/policy executables passed;
inference compiled with warnings treated as errors. The Python suite trains a temporary
ESN, exports a temporary C header, compiles a shared C inference library, and verifies
the C probabilities against Python on the same sequence.

## Integration smoke checks

Receiver simulation and CSV logging:

```bash
./venv/bin/python -m src.simulate_receiver --count 3 --interval 0 \
  | ./venv/bin/python -m src.receiver --output artifacts/telemetry/smoke.csv
```

Result: three frames passed magic/version/length/CRC validation and were logged.

The dashboard JSON loader is covered by the Python suite. The live server is started with
`python -m src.dashboard --log <receiver.csv>` and exposes `/health` and `/api/telemetry`.
Desktop (1280 px) and mobile (390 x 844 px) browser checks rendered 60 packets, a nonblank
temperature chart, responsive metrics, and no horizontal overflow.

Replacement generator check:

```bash
./venv/bin/python -m src.synthetic_v2 --stations hood_river \
  --start 2020-01-01 --end 2020-02-01 --events-per-year 12 \
  --output-dir artifacts/synthetic_smoke
```

Result: 8,928 five-minute rows and one event generated with raw, feature, catalog, and
manifest outputs. Existing v2 data was not changed.

## Model experiments

```bash
./venv/bin/python -m src.train_esn --max-rows-per-station 80000 --reservoir-size 48
./venv/bin/python -m src.baselines_v2 --max-rows-per-station 80000
./venv/bin/python -m src.loso_experiment --max-rows-per-station 30000 --reservoir-size 32
```

Outputs are in `docs/generated/`. Detailed ignored JSON reports are under
`artifacts/reports/`. The generated deployable model is under `firmware/generated/`.

## Still unverified

- Thermistor ADC-to-temperature accuracy on physical hardware.
- Feature parity between Python and STM32 ADC recordings.
- STM32 linked flash, stack high-water mark, inference latency, and energy.
- LoRa packet exchange through the selected radio stack and custom reader.
- Legal RF range, packet loss, and receiver recovery.
- Controlled heating-rig detection and false-alarm behavior.
- Long-duration drift, condensation, enclosure thermal effects, and battery behavior.
