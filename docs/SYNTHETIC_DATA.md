# Synthetic Thermal Dataset

The local `data/synthetic_v2/` and `data/processed/features_v2_*.csv` files predate the
tracked generator. They are preserved and remain useful for current experiments, but
their exact provenance cannot be reconstructed from git. `src/synthetic_v2.py` is the
deterministic replacement for future generated datasets; it does not pretend to recreate
the old bytes.

## Nominal model

Real station ambient temperature drives a synthetic load profile with daily, weekly, and
correlated stochastic components. A first-order thermal system moves asset temperature
toward `ambient + base rise + load-squared rise`; noise scale comes from the local ETT
calibration when available.

## Fault hypotheses

- `overload`: relatively fast saturating excess heat.
- `loose_connection`: progressive resistive heating with intermittent modulation.
- `cooling_degradation`: slow rise consistent with reduced heat rejection.
- `thermal_runaway`: accelerating nonlinear excess.

Severity controls peak excess. Excess below 2 C is `Normal`, 2-6 C is `Warming`, and 6 C
or above is `Anomaly`. These are transparent simulation labels, not safety limits. The
thresholds must be reviewed against controlled rig data and domain expertise.

## Reproducibility

The generator records seed, date range, stations, event rate, and version in a manifest.
It refuses to overwrite output unless `--force` is explicit. Write review datasets under
`artifacts/` so the unexplained legacy data remains untouched.
