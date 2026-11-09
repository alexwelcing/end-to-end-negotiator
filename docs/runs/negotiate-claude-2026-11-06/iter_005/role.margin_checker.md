I can't run the margin script: the Bash tool asked for approval on every attempt and didn't run it. I stopped there. I haven't computed any margins and haven't appended anything to `runs/negotiate-claude/evolution/learnings.md`.

The script is at `/tmp/claude-0/-home-user-end-to-end-negotiator/3959d833-7ff0-44fb-89a3-d234ec0534b4/scratchpad/m.py`. It is read-only. It loads `runs/negotiate-claude/iter_005/run/model.pt` with the compiled circuit set to None. It then computes the learned-only energies for all 2376 pairs and gets the margin (true successor vs. runner-up) for each. From those it reports:
- the global minimum margin and its pair;
- the minimum margin over accept pairs on unreachable states (reachability comes from a BFS with the oracle);
- the number of pairs with margin below 0.05;
- whether any spec-violating future is within 0.05 of the true successor;
- any missed pairs.

`python -m nerve step` only does single transitions and doesn't expose energies. I called the model's `energy_all` directly instead.

If you approve running that script, or allow Python commands in this session, I'll run it and append the one line (plus the missed-pairs line if there are any misses).

WHAT I CHANGED: nothing.
WHY: the margin run was blocked on approval.
WHAT I EXPECT: once it runs, the line can be appended. The last eval was 1.0 on learned-only, so I expect no misses.