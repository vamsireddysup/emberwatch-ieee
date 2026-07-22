# Generated ML Results

Run profile: `bounded`. Seed: `20260722`.

These values are software validation on the local synthetic-v2 data. They are not field-performance claims.

| Metric | Validation | Test |
|---|---:|---:|
| Rows | 400000 | 400000 |
| Three-class macro F1 | 0.5719 | 0.5946 |
| Alert precision | 0.7607 | 0.8072 |
| Alert recall | 0.4571 | 0.4722 |
| Alert false-positive rate | 0.0099 | 0.0082 |
| Event recall | 0.8624 | 0.9286 |
| Median lead minutes | 67.5000 | 25.0000 |
| False alert episodes/device-day | 0.2176 | 0.1842 |
| Periodic transmissions avoided | 0.9796 | 0.9798 |

Validation-selected alert threshold: `0.700`.

## Reproduce

```bash
./venv/bin/python -m src.train_esn --max-rows-per-station 80000 --reservoir-size 48
```

Inspect `artifacts/reports/esn_metrics.json` for confusion matrices and per-class values.
