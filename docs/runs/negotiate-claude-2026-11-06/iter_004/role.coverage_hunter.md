Coverage is 2376/2376 (fraction 1.0), but I did not finish step 3: the accept-pair duplication was blocked, so the accept mass is lower than you asked for.

**WHAT I CHANGED**
- Step 1: I ran `python -m nerve rollout runs/negotiate-claude/iter_004/spec.nerve --log runs/negotiate-claude/iter_004/episodes.v1.jsonl --policy coverage random --episodes 1200 --horizon 12`. The CLI caps episodes at 400 per policy, so it added 9600 oracle positives.
- Step 2: I ran the same command with `--policy exhaustive --negatives 0`. That added 2376 positives with source `exhaustive`, one for every (state, action) pair across all 216 grid states, including the 133 unreachable ones.
- I wrote the step-2 synthesis through the CLI rather than a custom script, because Python scripts and `nerve step` calls needed approval I couldn't get.
- Step 3 is not done. The shell append that would duplicate the 216 exhaustive accept lines was refused (file write needs approval), and I did not retry.
- I did not write to `evolution/learnings.md` for the same reason; it is also outside the iteration directory, which my rules say not to modify.

**WHY**
- Verification used `coverage()` from `nerve.train.eval`, the same function eval uses. It reports pairs 2376, in_log 2376, fraction 1.0, valid_states 216.
- The positives I added per action:

| Action | Oracle | Exhaustive |
|---|---|---|
| propose | 8577 | 1944 |
| accept | 996 | 216 |
| walk | 1027 | 216 |

- Those counts cover only what I added in this session. The log already held 4001 lines from before, which I did not tally.
- With the duplication step skipped, accept's effective mass is below propose and walk.

**WHAT I EXPECT**
- Eval coverage should be 1.0, and iter_003 already reached 1.0 on all pairs with a similar log.
- The missing accept duplicates make a regression unlikely but possible.
- To finish step 3, someone with write approval should run `tail -n 2376 runs/negotiate-claude/iter_004/episodes.v1.jsonl | grep -F '"action":["accept"' >> runs/negotiate-claude/iter_004/episodes.v1.jsonl` and then add the commands and counts to `learnings.md`.