# Synthetic Thermal Datasets

The canonical dataset-v2 pipeline is tracked in `src/calibrate_ett.py` and
`src/synthesize_thermal.py`. It produces `data/synthetic_v2/` from ETT calibration and
real station ambient data using seed 42. The detailed column, label, fault, and realism
contract lives in `data/synthetic_v2/README.md`.

`src/synthetic_v2.py` is a second, independent experimental generator added for model
sensitivity studies. It writes under `artifacts/` by default, uses a simpler thermal
formulation, and must not be presented as the source of the canonical v2 files.

## Alternate nominal model

Real station ambient temperature drives a synthetic load profile with daily, weekly, and
correlated stochastic components. A first-order thermal system moves asset temperature
toward `ambient + base rise + load-squared rise`; noise scale comes from the local ETT
calibration when available.

## Alternate fault hypotheses

- `overload`: relatively fast saturating excess heat.
- `loose_connection`: progressive resistive heating with intermittent modulation.
- `cooling_degradation`: slow rise consistent with reduced heat rejection.
- `thermal_runaway`: accelerating nonlinear excess.

Severity controls peak excess. Excess below 2 C is `Normal`, 2-6 C is `Warming`, and 6 C
or above is `Anomaly`. These are transparent simulation labels, not safety limits. The
thresholds must be reviewed against controlled rig data and domain expertise.

## Reproducibility

The canonical pipeline is reproduced with:

```bash
./venv/bin/python src/calibrate_ett.py
./venv/bin/python src/synthesize_thermal.py --station all --seed 42
```

The alternate generator records seed, date range, stations, event rate, and version in a
manifest and refuses to overwrite output unless `--force` is explicit. Keep alternate
review datasets under `artifacts/` so canonical v2 data remains distinct.
