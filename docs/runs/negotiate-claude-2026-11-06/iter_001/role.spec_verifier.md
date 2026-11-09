**WHAT I CHANGED:** I didn't edit the spec or any run files. I added one entry to `runs/negotiate-claude/evolution/learnings.md`. `python -m nerve check` reports 0 problems (216 states, 11 actions, no holes).

**WHY:** The invariants don't make any state unreachable, and the "164 valid / 52 excluded" split doesn't reproduce.
- **Invariants:** Both hold on all 216 states, because the field ranges already enforce them. `coverage()`'s own filter, run locally, finds 0 violating states, so valid is 216, not 164. The 164/216 in iter_000 `eval.json` doesn't match the current code and spec.
- **Reachability:** A search from the initial state over all 11 actions reaches 83 states and leaves 133 unreachable.
- **rounds=0:** Only `(a,0,0,open)` and its walk successor `(a,0,0,no_deal)` are reachable. `accept` needs rounds > 0, and `propose` always increments rounds. The other 52 rounds=0 states are unreachable (17 open, 17 no_deal, 18 deal). I read these as the "52" the charter means, but that is my interpretation.
- **rounds 1 to 3:** `propose` flips `turn` every round, so turn is fixed by parity (b at odd rounds, a at even). The 81 wrong-parity states are unreachable.

**WHAT I EXPECT:** No coverage gap for `coverage_hunter` among reachable states: all 83 appear in the log, and eval coverage is 1.0. The score should stay at 1.0, unless heldout samples unreachable grid states.