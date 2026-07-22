# Generated Per-Station Robustness

Each station's own 2023 test split, scored by the deployed model, under two
input-normalization schemes: the shipped fleet-wide normalization vs per-station
self-calibration (feature mean/std from that station's train-split Normal rows
only). Reservoir and readout weights are identical in both; only input scaling
differs. Software validation, not field claims.

| Station | FPR global | FPR self-cal | FPR trim | Trim thresh | Event rec. global | Event rec. trim |
|---|---:|---:|---:|---:|---:|---:|
| gerber | 0.0112 | 0.0407 | 0.0061 | 0.71 | 1.000 | 0.867 |
| hillsboro | 0.0050 | 0.0517 | 0.0050 | 0.70 | 0.947 | 0.947 |
| hood_river | 0.0052 | 0.0892 | 0.0052 | 0.70 | 0.900 | 0.900 |
| klamath | 0.0154 | 0.0341 | 0.0072 | 0.71 | 0.909 | 0.909 |
| troutdale | 0.0043 | 0.0532 | 0.0043 | 0.70 | 0.909 | 0.909 |

## Reading

- The global-normalization column reproduces the known pattern: the RAWS
  wildfire-region stations (Klamath, Gerber) carry the highest false positive rate;
  the worst here is `klamath` at 0.0154.
- **Naive self-calibration did NOT help** (0/5 stations
  improved). Swapping only the input normalization at inference raises FPR across the
  board, because the readout weights and the alert threshold were both fit against the
  global normalization. Change the input scaling underneath a fixed readout and the
  decision boundary no longer sits where it was tuned.
- **Per-station threshold trim works** (5/5 stations
  met the 0.010 FPR budget, 4/5 within
  0.05 of their original event recall). The threshold is chosen on each station's 2022
  validation split (a commissioning proxy) and never lowered below the shipped value, so
  it is leakage-safe and can only trade recall for fewer false alarms. In particular the
  RAWS stations Klamath and Gerber, the ones over budget under global settings, are
  brought under budget. This is the cheap firmware path: during commissioning, raise the
  alert threshold per device until the observed Normal-period FPR meets budget. It
  touches no shipped weight.
- A fuller fix (retraining the ridge readout per station) is left to the ML pipeline
  owner; the threshold trim is enough to bring the RAWS stations into budget now.
- The shipped model artifact is unchanged by this analysis.

## Reproduce

```bash
./venv/bin/python -m src.station_robustness --max-rows-per-station 80000
```

Full numbers in `artifacts/reports/station_robustness.json`.
