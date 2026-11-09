iter_010 coverage_hunter: exhaustive_lines 4752, oracle_lines 14197, adversary_lines 5003, log_lines 23953

WHAT I CHANGED: I ran the four rollout commands in order: coverage random, exhaustive twice, then adversary once with 3 negatives. I wrote no scripts and did not touch the spec.
WHY: The exhaustive count is 4752, so each of the 2376 grid pairs has at least two positives. The adversary count of 5003 is under the 21000 cap.
WHAT I EXPECT: Coverage should stay at 1.0, as in iter_009 and the iter-9 recipe, so the score should hold at 1.0. I haven't seen a new eval.