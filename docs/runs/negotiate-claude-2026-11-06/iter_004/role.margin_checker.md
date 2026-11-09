I can't run the margin check. The shell is asking for approval on every command that would run my script, and I got no answer. I haven't written anything to `evolution/learnings.md`.

The script is ready at `/tmp/claude-0/-home-user-end-to-end-negotiator/d936f0a5-e273-442d-aa01-d952ff834df4/scratchpad/m.py`. It loads the iter_004 model (`runs/negotiate-claude/iter_004/run/model.pt`) and computes the learned-only energies with the compiled circuit off. For all 216 states × 11 actions it reports:
- the gap between the true successor and the runner-up, with the minimum overall, per action, on the held-out split and on the unreachable-state pairs (including accept);
- the 10 lowest-margin pairs;
- any mispredicted pair, listed exactly;
- any spec-violating future within 0.05 of the true successor.

Approving `python <that path>` from the repo root lets me run it and write the results to the learnings file.

**WHAT I CHANGED:** Nothing. I wrote the margin-check script in the scratchpad and made no edits to the repo, spec or knobs.
**WHY:** Both the script run and the copy into the iteration directory were blocked at the approval prompt.
**WHAT I EXPECT:** The model should still be at 1.0 learned-only. The previous eval.json reported 1.0 on all, held-out and learned-only, but I have no margin numbers to confirm it.