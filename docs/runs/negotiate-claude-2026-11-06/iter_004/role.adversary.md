WHAT I CHANGED: Ran `nerve rollout` with `--policy adversary --negatives 3` on iter_004. This added 1000 positives and 3000 near-miss negatives (log is now 4001 lines: 1 header, 1000 positives, 3000 negatives). I did not edit the spec or write a knobs file.

WHY: This holds the negative-to-positive ratio at 3:1, the cap in my charter. The sample negative I looked at breaks the `turn` flip on propose. I did not check what share of negatives break each invariant or ensures clause.

WHAT I EXPECT: The learned model keeps rejecting the single-clause violations, so the score should stay at the 1.0 that iter_003 reached. At 3:1 I don't expect the learned-only accept regression that higher ratios showed in iter 1. I can't train here, so this is untested.