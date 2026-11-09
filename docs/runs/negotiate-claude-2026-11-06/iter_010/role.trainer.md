iter_010 trainer: wrote knobs epochs 60 lr 0.002 d 64 hidden 256 hole_weight 1.0; matches iter_009: yes

WHAT I CHANGED: I wrote `runs/negotiate-claude/iter_010/knobs.trainer.json` with the five required values. I changed nothing else.
WHY: Those are the iter_009 values, which I read from `runs/negotiate-claude/iter_009/knobs.trainer.json` to confirm. That iteration scored 1.0 on every split.
WHAT I EXPECT: I expect the same result as iter_009 (score 1.0, no violations). The harness trains after my phase, so this is untested until then.