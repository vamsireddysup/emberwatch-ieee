# Machine Learning Pipeline

## Claim and target

The model predicts `Normal`, `Warming`, or `Anomaly` from asset/ambient thermal dynamics.
The operational alert is `Warming OR Anomaly`. The desired advantage over a fixed
threshold is stable recall with fewer false transmissions across weather conditions.

## Data layers

- Real ambient weather supplies seasonal and station variability.
- ETT transformer data supplies measured noise/load-shape calibration through
  `src/calibrate_ett.py`.
- The canonical `src/synthesize_thermal.py` healthy/fault twin simulation creates paired
  nominal asset temperature and controlled excess for four fault families.
- NAB remains an independent real-machine anomaly sanity check, with explicit caveats.
- Heated-rig recordings will supersede synthetic assumptions for final calibration.

The canonical generator is deterministic by seed and reproduces `data/synthetic_v2/`.
The alternate `src/synthetic_v2.py` generator writes to `artifacts/` by default and
refuses to overwrite existing data without `--force`.

## Leakage controls

- 2020-2021 train, 2022 validation, 2023 test.
- Fit feature normalization and ESN readout on train only.
- Tune alert threshold/persistence on validation only.
- Use test once for final model comparison.
- Keep event windows together; never randomly split rows from the same event.
- A cross-station experiment should hold one station out entirely.

## ESN

The reservoir update is:

```text
x_t = (1-leak) x_(t-1) + leak*tanh(W_in*[1,u_t] + W_res*x_(t-1))
logits = W_out*[1,u_t,x_t]
```

`W_in` and sparse `W_res` are deterministic fixed weights. Only ridge-regression readout
weights are fitted. This gives temporal memory without backpropagation through time and
maps cleanly to fixed-size C arrays.

Class weighting prevents the majority `Normal` class from dominating. Normalization
parameters are exported with the model. MCU inference clips normalized features to
protect against extreme sensor values; sensor faults are handled before ML.

## Metrics

- Three-class confusion matrix, per-class precision/recall/F1, macro F1.
- Binary alert precision, recall, F1, false-positive rate, and specificity.
- Event detection rate and lead time before first anomaly label.
- False-alert episodes per device-day rather than only false rows.
- Fraction of periodic radio transmissions avoided after persistence and heartbeat.
- Model flash estimate, state RAM, and measured MCU inference energy/latency.

## Commands

Bounded smoke run using local v2 files:

```bash
./venv/bin/python -m src.train_esn --max-rows-per-station 12000 --reservoir-size 24
```

Larger candidate run:

```bash
./venv/bin/python -m src.train_esn --max-rows-per-station 80000 --reservoir-size 48
```

Regenerate the canonical dataset:

```bash
./venv/bin/python src/calibrate_ett.py
./venv/bin/python src/synthesize_thermal.py --station all --seed 42
```

Run an alternate simulation only into a new directory during review:

```bash
./venv/bin/python -m src.synthetic_v2 --output-dir artifacts/synthetic_review
```

Export a saved model:

```bash
./venv/bin/python -m src.export_c artifacts/models/esn_model.npz firmware/generated/emberwatch_model.h
```

Test geographic generalization by excluding each station from training and validation:

```bash
./venv/bin/python -m src.loso_experiment --max-rows-per-station 30000 --reservoir-size 32
```

The trainer writes model and report artifacts under `artifacts/` and refreshes the
tracked summary in `docs/generated/ML_RESULTS.md`.
