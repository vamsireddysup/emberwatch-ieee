# Generated Quantization Results

Post-training quantization and sparse packing of the 48-unit ESN, evaluated on the
bounded synthetic-v2 2023 test split (400,000 rows). These are software validation
numbers, not field-performance claims.

Weights only are quantized (symmetric, per output row); accumulation stays in float. The
reservoir is packed CSR because it is generated with `connectivity=0.15`, so about 85
percent of its entries are structurally zero. Feature mean/std stay float32.

| Variant | Constant bytes | vs float | Alert precision | Alert recall | Alert FPR | Event recall | Decision agreement | Max prob error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float32 | 11268 | 1.00x | 0.8132 | 0.4514 | 0.0076 | 0.9388 | 1.0000 | 0 |
| int16 sparse | 2543 | 4.43x | 0.8133 | 0.4514 | 0.0076 | 0.9388 | 1.0000 | 6.4e-05 |
| int8 sparse | 1709 | 6.59x | 0.8022 | 0.4591 | 0.0083 | 0.9388 | 0.9985 | 2.1e-02 |
| int8 dense | 3249 | 3.47x | 0.8022 | 0.4591 | 0.0083 | 0.9388 | 0.9985 | 2.1e-02 |

"Constant bytes" counts only the model weight/scale/index constants, not C code or the
reservoir RAM state (192 bytes of float). "Decision agreement" is the fraction of rows
whose binary alert decision matches the float model at the deployed 0.70 threshold.

## Reading

- int16 sparse is numerically indistinguishable from float (agreement 1.0000) at 4.43x
  smaller. It is the safe default if a reviewer wants zero accuracy risk.
- int8 sparse reaches 1709 bytes, 6.59x smaller than float, and holds event recall
  exactly (0.9388). Row-level alert decisions differ on 0.15 percent of rows; precision
  drops 1.1 points and recall rises 0.8 points, which is drift, not degradation.
- Sparsity is the larger single win for the reservoir (int8 sparse 1709 vs int8 dense
  3249); it is only available because the reservoir connectivity mask is fixed and known.

The int8-sparse model meets the original MCU target of an int8/int16 model in a few
kilobytes. Final selection between int16 and int8 should be revisited against measured
MCU flash budget and, if available, controlled-rig accuracy.

## Reproduce

```bash
./venv/bin/python -m src.quantize --max-rows-per-station 80000
./venv/bin/python -m src.export_c_quant
cc -std=c11 -Wall -Wextra -Werror -Ifirmware/include -Ifirmware/generated \
  -c firmware/src/emberwatch_inference_q.c -o build/inference_q.o
```

`src/export_c_quant.py` writes `firmware/generated/emberwatch_model_q.h`; the runtime is
`firmware/src/emberwatch_inference_q.c`. Python and C are checked bit-for-close in
`tests/test_quantized_parity.py`. Inspect `artifacts/reports/quantization_metrics.json`
for the full metric set.
