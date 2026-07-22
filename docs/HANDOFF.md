# Work Handoffs

## 2026-07-22 - Claude quantization and sparse export

- Objective: meet the original MCU constraint (int8/int16 model, a few kilobytes) that
  the float export did not satisfy (11.3 KB of float32 weights).
- Owner: Claude (AI/data). New files only; no existing file rewritten except append-only
  Makefile targets and this handoff / the decision log.
- Files added: `src/quantize.py` (per-row symmetric quantization + CSR reservoir packing
  + accuracy ablation), `src/export_c_quant.py`, `firmware/include/emberwatch_inference_q.h`,
  `firmware/src/emberwatch_inference_q.c`, `firmware/generated/emberwatch_model_q.h`
  (generated), `tests/test_quantized_parity.py`, `docs/generated/QUANTIZATION_RESULTS.md`.
- Files touched: `Makefile` (added `quantize`, `export-quant` targets).
- Approach: weight-only quantization, float accumulation, per-output-row scales; reservoir
  packed CSR because `connectivity=0.15` leaves ~85% structural zeros. Float path
  (`export_c.py`, `emberwatch_inference.c`) left intact as the reference.
- Verification: full suite 15/15 (was 9; +6 quantization/parity incl. C↔Python parity for
  int8 and int16). Quantized C compiles under `-Wall -Wextra -Werror`. Ablation on the
  400,000-row test split: int8-sparse = 1709 bytes (6.59x smaller), event recall unchanged
  at 0.9388, 99.85% row-level decision agreement with float. int16-sparse = 2543 bytes,
  numerically identical to float.
- Commands: `make quantize`, `make export-quant`, `python -m src.quantize`,
  `python -m src.export_c_quant [--bits 16]`. Results in
  `artifacts/reports/quantization_metrics.json`, summary in
  `docs/generated/QUANTIZATION_RESULTS.md`.
- Unresolved / next: final int8-vs-int16 choice needs the hardware flash budget; not
  frozen. Also flagged to the team but not yet acted on: the transmission-reduction claim
  (~98%) is matched by the threshold baseline, so the ESN's defensible edge is lead time
  and event recall, not packets avoided.

## 2026-07-22 - Codex software foundation

- Objective: extend the reproducible canonical v2 pipeline with the complete
  ML-to-receiver software path while preserving all existing material.
- Scope: documentation, alternate simulation, ESN, metrics, C export/reference, protocol,
  receiver/dashboard, and tests. The remote canonical ETT-calibrated v2 generator was
  merged and retained.
- Assumptions: five-minute model cadence; STM32WL55/Cortex-M4 target; two NTC channels;
  LoRa node-to-reader and USB serial reader-to-host.
- Existing files removed: none.
- Verification: Python tests, C packet build, Python/C inference parity, receiver pipe,
  synthetic generator smoke run, bounded ESN training, and matching baselines completed.
- Exact commands and remaining physical checks: `docs/TESTING.md` and
  `docs/VALIDATION_PLAN.md`.
- Current candidate: 48 reservoir units; 400,000 rows/split; test alert precision 0.807,
  recall 0.472, FPR 0.0082; synthetic sampled-event recall 0.929; median lead 25 minutes.
- Station holdout: recall 0.411-0.594. Gerber and Klamath exceeded 1% FPR, indicating
  per-device calibration or adaptation should be evaluated before model freeze.
- Next review: Claude should challenge data leakage, physical fault assumptions, model
  size, report claims, and competition narrative against the implementation and docs.
