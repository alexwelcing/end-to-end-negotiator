WHAT I CHANGED: I appended 1000 positives and 3000 near-miss negatives to `episodes.v1.jsonl`. I used `--policy adversary --negatives 3`, so the ratio is exactly 3:1. Each negative has the same state and action as its positive and breaks one invariant or ensures clause. The log has 4000 step lines plus its header. I didn't write a knobs file because my write to `knobs.adversary.json` was blocked for approval. The negatives count is a CLI flag, so nothing is lost.

WHY: This follows the charter's iter-0 ratio. Higher ratios in iter 1 crowded out accept positives.

WHAT I EXPECT: The score should stay at or near 1.0, as in iter 4, with violation rate 0. I haven't run training or eval, so that is an expectation, not a result.