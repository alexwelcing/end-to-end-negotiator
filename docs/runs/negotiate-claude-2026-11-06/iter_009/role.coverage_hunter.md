WHAT I CHANGED: I appended 6400 coverage-random positives and ran the exhaustive pipeline verb twice, adding 2376 positives each time. I added no negatives and didn't touch the spec. `grep -c '"source":"exhaustive"'` gives 4752, which is two positives per (state, action) pair over all 216 states. I used no scripts.

WHY: The last eval had only 1555 of 2376 pairs in the log (fraction 0.6545), and most of the missing ones were on states unreachable from initial. The exhaustive policy enumerates all of them through the oracle.

WHAT I EXPECT: Full coverage should let the model learn `accept` on the unreachable states. That is where the misses were: `accept` was 0.991 on all pairs and 0.947 on heldout, while `propose` and `walk` were already at 1.0.

**Not done:** the learnings line is not in `runs/negotiate-claude/evolution/learnings.md`. Both a shell append and the Edit tool were denied write permission. Here is the line to add:
`- iter_009 coverage_hunter: exhaustive_lines 4752, oracle_lines 14197 (grep -c source oracle), log_lines 23953 (wc -l)`

The oracle count comes from `grep -c '"source":"oracle"'`, not the literal `grep -c source` (which counts every line).