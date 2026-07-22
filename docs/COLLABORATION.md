# Claude, Codex, and Team Workflow

## Shared source of truth

Chat history is not authoritative. At the start of work, read `docs/PROJECT_STATE.md`.
At the end of material work, update that file and append to `docs/DECISION_LOG.md`.

## Handoff format

Add a short entry to `docs/HANDOFF.md` containing:

- objective and owner;
- files changed;
- assumptions or hardware values used;
- commands run and their outcomes;
- unresolved decisions;
- exact next action.

Do not have Claude and Codex edit the same file simultaneously. Split work by ownership,
for example model code versus report review, or receiver code versus hardware constants.
Before integrating, inspect `git status` and preserve unrecognized changes.

## Data and claims

Every metric must identify dataset, split, station coverage, model artifact, and decision
policy. Never present synthetic performance as field performance. Keep the untouched test
split for final comparisons; iterate thresholds only on validation.

## Archive policy

Nothing is deleted. Superseded files move to `archives/YYYY-MM-DD/`, and
`archives/INDEX.md` records why and where the replacement lives.
