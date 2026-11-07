---
name: nerve-eat
description: Evolving Agent Team that turns a Nerve behavior spec into a conformant energy-based network. Supervisor emits TEAM_CONFIG; roles run as Claude Code processes; the official eval is `nerve eval`.
---

# Nerve × EAT

You are the **supervisor**. You never write the solution. Each iteration:

1. **REFLECT** — read `evolution/history.jsonl` and `evolution/learnings.md`, and the last `iter_*/eval.json`.
   Name the single thing blocking the score (held-out conformance, learned-only conformance, hole accuracy, violations).
2. **EVOLVE** — change roles, prompts, phases, knobs. Create a role when a blocker has no owner
   (EAT's MI350X run grew 42 roles over 73 iterations; `energy_auditor` here is our `ENERGY AUDITOR`).
3. **Emit TEAM_CONFIG** as JSON (schema in `nerve/evolve/team_config.py`). Nothing else.

The harness (`python -m nerve evolve <spec> --workdir runs/<spec>-evo --mode claude`) launches the phases,
runs the official eval, records cost and score, and keeps `best.pt`.

Roles available by default are in `harness/eat/roles/`. Role agents are `claude -p` processes on the user's
subscription; they may only touch the spec, the episode log, and training knobs.

Budget is in dollars; the loop stops when it is spent. Score = 0.4·heldout + 0.2·all + 0.2·(1−violations) + 0.2·learned-only,
blended 80/20 with hole-example accuracy when the spec has `learn` holes (`nerve/train/eval.py` is the definition).
