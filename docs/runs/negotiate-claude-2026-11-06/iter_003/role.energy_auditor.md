I can't run the audit script: the shell is asking for approval on each attempt, and none was given. I stopped there and didn't try another route around the prompt.

**Done so far**
- `iter_003` already has a trained model, `run/model.pt`, and a finished `eval.json`. It reports `learned_only` exact = 1.0, with accept = 1.0. The log covers 2376 of 2376 pairs. Training used 6512 pairs and 18294 negatives, which is 2.81 negatives per positive.
- I wrote the audit script at `/tmp/claude-0/-home-user-end-to-end-negotiator/0562e1af-ddeb-4854-a9dc-1c0a75cab414/scratchpad/audit.py`. It lists every learned-only miss with its state, action, true and predicted successor, energies and runner-up margin. It also marks whether the state is reachable from initial and whether the pair is in the log as a positive. It samples 30 accept pairs at wrong-parity states and 30 at reachable states, and reports median and minimum margin for each. It counts positives per action from the log too.

**Not done**
- I haven't run it, so `evolution/learnings.md` has no entry from me.
- **Per-action positive counts:** not produced; the script computes them.
- **Margin medians and minimums:** not produced; the script computes them.
- **Coverage vs. margin call:** not made. I'd expect zero misses, since `eval.json` says 1.0. If so, the accept misses from `iter_002` (96.8%) are gone, and the log now holds every pair as a positive.

**Reporting**
- **WHAT I CHANGED:** nothing in the repo; I only wrote the audit script above.
- **WHY:** the shell asked for approval on running it, and I didn't try to work around that.
- **WHAT I EXPECT:** once the script runs, it should list no misses, and the margin figures will show how close the accept pairs at wrong-parity states are to failing. If you approve the run, I'll write the results into `evolution/learnings.md`.