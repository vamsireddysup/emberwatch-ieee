# EmberWatch Agent Instructions

Read `docs/PROJECT_STATE.md` first, then `docs/ARCHITECTURE.md` and
`docs/COLLABORATION.md` before changing code.

The product detects an overheating grid asset before it becomes an ignition source. It
does not detect an already-burning wildfire. Preserve this distinction in code, plots,
reports, and presentation language.

## Working rules

- Never delete project material. Move superseded material to `archives/` and add an
  entry to `archives/INDEX.md` with the date, old path, reason, and replacement.
- Treat ignored datasets and generated artifacts as valuable local state. Do not
  overwrite them unless the command has an explicit `--force` option and the user asked
  for regeneration.
- Keep MCU inference deterministic and implementable without dynamic allocation.
- Fit normalization, thresholds, and model weights on training data only. Use validation
  for model selection and test data once for final reporting.
- Report both sample metrics and event metrics. False-positive rate and avoided radio
  transmissions are first-class metrics because they connect AI to battery life.
- Update `docs/PROJECT_STATE.md` and `docs/DECISION_LOG.md` after material changes so
  Claude and Codex can hand work back and forth without relying on chat history.
- Prefer small, reviewable changes. Record generated commands and assumptions.

## Verification

Run at minimum:

```bash
./venv/bin/python -m unittest discover -s tests -v
./venv/bin/python -m py_compile src/*.py
```

For ML changes also run a bounded smoke training command documented in
`docs/ML_PIPELINE.md`. For firmware changes compile the host-side C smoke test.
