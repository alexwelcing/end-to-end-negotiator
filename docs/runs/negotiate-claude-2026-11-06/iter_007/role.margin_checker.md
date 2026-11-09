Margin check not run: `nerve step` can't give the margins, and the sweep I tried instead was blocked by the sandbox.

- **What I changed:** I wrote `runs/negotiate-claude/iter_007/run/margin_check.py`, which I did not run. It is read-only and uses `NerveEnergy.load` and `jepa.energy_all` per pair. It is unverified, since I never ran it. I appended one UNAVAILABLE line to `learnings.md`.
- **Why:**
  - `nerve step` runs one (state, action) pair per call, about 1.7s each.
  - Each call prints one energy that adds the circuit term to the JEPA term, plus only 2 nearby alternative successors. It has no learned-only flag and doesn't list all 216 candidates, so it can't give a true learned-only margin.
  - My sampled sweep of about 126 pairs was refused by the sandbox, so I removed that script.
- **What I expect:** With permission for `python runs/negotiate-claude/iter_007/run/margin_check.py runs/negotiate-claude/iter_006/run/model.pt`, a later iteration gets the full learned-only margins in one run. The iter_007 model doesn't exist yet, so that command points at the iter_006 model.