I couldn't run the audit, so I haven't written the verdict to `evolution/learnings.md`. Bash calls that run Python (`python -m nerve ...` and my audit script) were refused for approval. I won't claim results I didn't compute.

The script is ready at `/tmp/claude-0/-home-user-end-to-end-negotiator/cb2f2f50-86d4-4235-a4a9-56aa224e3bf5/scratchpad/aud.py`. It loads `iter_002/run/model.pt` and does the three checks from the charter:
1. It lists every pair where the learned-only prediction differs from the spec successor, with state, action, true and predicted energy, and margin.
2. It reports the minimum accept margin over the runner-up and flags any below 10% of the median.
3. It checks agreement on unreachable states, found by walking the spec from the initial state.

If you approve Python execution, I'll run it and write the paragraph. The iter_002 `eval.json` shows learned-only at 0.997, with 5 accept pairs wrong. The iteration 1 file shows 0.998, with 4 wrong.

WHAT I CHANGED: Nothing. No files were modified, and the spec and run are untouched.
WHY: Approval for running Python was denied, so the audit never ran.
WHAT I EXPECT: Once approved, the verdict will say whether the 5 wrong accept pairs are a coverage problem or a margin problem. I have no result yet. The only hint is that coverage in the log fell from 0.82 in iteration 1 to 0.63 in iteration 2 while learned-only barely moved.