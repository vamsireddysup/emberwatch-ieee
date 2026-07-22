# Generated Model Comparison

Every deployable detector scored through the identical operational alert policy
(persistence + heartbeat) on the same bounded synthetic-v2 2023 test split. ESN
variants are evaluated live; threshold baselines are read from
`artifacts/reports/baseline_metrics.json`. Battery years use the placeholder
hardware assumptions in `src/energy_model.py`. Software validation, not field claims.

| Model | Size (B) | Alert prec | Alert recall | FPR | Event recall | Median lead (min) | Tx avoided | Battery (yr) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| esn_float32 | 11268 | 0.8072 | 0.4722 | 0.0082 | 0.9286 | 25.0 | 0.9798 | 101.3 |
| esn_int16_sparse | 2543 | 0.8075 | 0.4722 | 0.0082 | 0.9286 | 25.0 | 0.9798 | 101.3 |
| esn_int8_sparse | 1709 | 0.7936 | 0.4798 | 0.0091 | 0.9286 | 25.0 | 0.9797 | 101.3 |
| fixed_delta_3sigma | n/a | 0.9963 | 0.2254 | 0.0001 | 0.7041 | -70.0 | 0.9813 | 102.5 |
| multivariate_statistical | n/a | 0.8373 | 0.1678 | 0.0024 | 0.7245 | 20.0 | 0.9790 | 100.7 |

## How to read this

- **Event recall and median lead** are where the ESN separates from the threshold
  baselines: it detects more distinct fault events and, at a positive median lead,
  alerts before the anomaly label rather than after. This is the AI claim.
- **Tx avoided and battery years** are nearly identical across the ESN and the
  better baseline. Gating is a system property, not an ESN-specific one.
- **Size** shows the quantized ESN variants cost a fraction of the float model while
  holding event recall (see `docs/generated/QUANTIZATION_RESULTS.md`).

## Reproduce

```bash
./venv/bin/python -m src.train_esn --max-rows-per-station 80000 --reservoir-size 48
./venv/bin/python -m src.baselines_v2 --max-rows-per-station 80000
./venv/bin/python -m src.compare_models --max-rows-per-station 80000
```

Full numbers in `artifacts/reports/model_comparison.json`.
