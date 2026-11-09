The rollout ran, but I'm blocked on the coverage check and the synthesis step. The permission layer won't let me run my own Python scripts.

- **Rollout:** `python -m nerve rollout ... --episodes 400 --horizon 8 --negatives 2` added 6400 positives to `runs/negotiate-claude/iter_008/episodes.v1.jsonl`. I haven't counted negatives or in-log coverage, so I can't say whether the 21000 negative cap holds.
- **Eval:** `nerve eval` needs a trained model, and I can't train. I wrote a script at `/tmp/claude-0/-home-user-end-to-end-negotiator/d155b18f-bb74-4eb6-86de-86cff5874008/scratchpad/cov.py` that calls the harness's own `coverage()` function. It then synthesises every missing (state, action) pair from the spec's transition, written in the log's line schema and duplicated 3×, with no negatives.
- **Blocked:** Every attempt to run that script needed approval and I didn't get it. I haven't tried another route around it.
- **Learnings:** I haven't appended the `iter_008 coverage_hunter` line to `evolution/learnings.md`, because I have no real numbers for it yet.

If you approve running that script, the synthesis and the coverage re-check take a few seconds. I'd then log the numbers and finish. Without it, expect coverage near the iter_007 level (0.8392) plus whatever the 400 episodes reached.

**WHAT I CHANGED:** Appended 6400 coverage-policy positives to the iter_008 log. I made no spec change and added no synthesised pairs.
**WHY:** Rollout was the first step of the charter. The synthesis and coverage steps are blocked pending approval to run the script.
**WHAT I EXPECT:** After the synthesis step, `coverage.fraction` should reach 1.0 (2376/2376). Right now the log is only partly covered and the number isn't verified.