eval.json doesn't exist yet (the harness trains after my phase), so I can't append the learnings line now.

WHAT I CHANGED: Wrote `runs/negotiate-claude/iter_009/knobs.trainer.json` with epochs 60, lr 0.002, d 64, hidden 256, hole_weight 1.0. These match the iter_005 file. I left the spec and the compiled circuit alone.
WHY: The charter fixes these values. Iter 8 regressed to epochs 40 / lr 0.001 when it couldn't find this file.
WHAT I EXPECT: Scores similar to iter 8 (about 0.997) or better. I haven't appended the `iter_009 trainer: ...` line to `evolution/learnings.md` because `iter_009/eval.json` is only written after the harness trains. Once it exists I'll add the line, plus the missed accept pairs if `learned_only.per_action.accept` is below 1.0.