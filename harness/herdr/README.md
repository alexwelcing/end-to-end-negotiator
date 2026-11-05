# herdr layout

herdr is the terminal runtime the team lives on: one pane per role, each marked working / blocked / idle, with a
socket API agents use to spawn panes and wait for each other. Start `herdr` in the repo root; `nerve evolve` detects
it on PATH and launches each role as a pane (`nerve/evolve/herdr.py`), falling back to subprocesses otherwise.

Suggested workspace: one pane running `python -m nerve evolve ... --mode claude` (the supervisor), one per phase
role, one running `python -m nerve serve` so the energy_auditor can poke the live network, and one `wrangler dev`
in `edge/`. Remote GPUs are herdr "machines": train there, patch back with `nerve patch`, and the edge worker picks
the patch up from R2.
