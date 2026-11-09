# Evolving Agent Team run: negotiate, 2026-11-06

`nerve evolve specs/negotiate.nerve --mode claude --remote hf --iterations 12 --budget 20`, stopped by hand after
iteration 10 (11 recorded) with the score saturated. Supervisor: Claude Code (claude-fable-5-1). Roles: Claude Code
processes (claude-sonnet-5-5), run in parallel per phase. Training: one Hugging Face Job (a10g-small) per iteration.

| iter | score | learned-only | team | cost |
|---|---|---|---|---|
| 0 | 1.0000 | 1.000 | spec_auditor, coverage_hunter, adversary, trainer, energy_auditor | $0.56 |
| 1 | 0.9997 | 0.998 | spec_verifier, coverage_hunter, adversary, trainer, margin_auditor, split_prober | $1.56 |
| 2 | 0.9994 | 0.997 | coverage_hunter, adversary, trainer, energy_auditor | $1.12 |
| 3 | 1.0000 | 1.000 | coverage_hunter, adversary, trainer, energy_auditor, spec_verifier | $1.41 |
| 4 | 1.0000 | 1.000 | spec_verifier, coverage_hunter, adversary, trainer, margin_checker | $1.33 |
| 5 | 1.0000 | 1.000 | spec_verifier, coverage_hunter, adversary, trainer, margin_checker | $1.32 |
| 6 | 1.0000 | 1.000 | spec_verifier, coverage_hunter, trainer, margin_checker | $1.38 |
| 7 | 1.0000 | 1.000 | spec_verifier, coverage_hunter, adversary, trainer, margin_checker | $1.48 |
| 8 | 0.9966 | 0.995 | spec_verifier, coverage_hunter, trainer | $1.08 |
| 9 | 1.0000 | 1.000 | spec_verifier, coverage_hunter, adversary, trainer | $1.67 |
| 10 | 1.0000 | 1.000 | spec_verifier, coverage_hunter, trainer | $1.32 |

Total: $14.23. Roles invented by the supervisor that did not exist at iteration 0:
`split_prober`, `spec_verifier`, `margin_auditor`, `margin_checker`. Iteration 10's rationale is worth reading in
`history.jsonl`: with nothing left to win on the score, the supervisor restructured the team to cut cost per iteration.

Each `iter_NNN/` holds the TEAM_CONFIG the supervisor emitted, the official eval, and every role's own report.
