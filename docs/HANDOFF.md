# Work Handoffs

## 2026-07-22 - Codex software foundation

- Objective: make the local v2 data reproducible and build the complete ML-to-receiver
  software path while preserving all existing material.
- Scope: documentation, synthetic generator, ESN, metrics, C export/reference, protocol,
  receiver, and tests.
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
