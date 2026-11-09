WHAT I CHANGED: I ran `python -m nerve rollout` with `--policy coverage random --episodes 1800` on the iter_001 `spec.nerve`. I left the horizon alone. That added 16,000 positives to `episodes.v1.jsonl`, which now has 64,001 lines. I did not edit the spec.

WHY: This reproduces the iter-0 coverage level. The log had only 1 line before the run. The spec_verifier found no reachable state that lacked coverage, so I kept N at the iter-0 level.

WHAT I EXPECT: I did not check coverage after the rollout. I expect it to match iter-0: all 83 reachable states logged (per the spec_verifier note) and a fraction of 1.0 on the 1804 pairs eval counts. I expect the score to stay at about 1.0 after training.