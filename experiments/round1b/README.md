# Round 1b: the reorder hold-out sweep, logged

Before the experiment log existed, the reorder sweep of 2026-11-07 (factored head on/off, 15–90% held out,
and the valid-candidates comparison) was recorded only as tables in `docs/dragology-demos.md`. Its models are in
`runs/reorder/` (not committed). `record_reorder.sh` rescores every one of them with `nerve eval --json` in both
modes into `reorder_holdout.jsonl`, so the numbers paper 2 quotes in §2.2 come from a log, not from prose.

Runs: `h15` (15% held out, factored head on), `f10-h*` (head on, 50/70/85/90%), `f0-h*` (head off, 15/50/70%),
`valid-h*` (head on, trained over the 120 valid orders only). Modes: `spec-assisted` answers from the 120 valid
orders; `network-only` from all 3,125 tuples.
