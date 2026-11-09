I wrote the knobs file, but training hasn't run yet, so I have no eval numbers to report or compare against iter-3.

WHAT I CHANGED: I wrote `iter_004/knobs.trainer.json` as `{"epochs": 60, "lr": 0.003, "d": 32, "hidden": 256, "hole_weight": 1.0}`. `learnings.md` has no margin_checker note on a learned-only margin below 0.05, so I did not raise epochs to 80.

WHY: Those are the values in my charter. Note that `iter_003/knobs.json` on disk says something different: epochs 120, d 64, hole_weight 4. Please check which one is the real iter-3 config. If the on-disk one is, my file isn't an exact copy of it.

WHAT I EXPECT: The previous eval scored 1.0 everywhere with zero violations, 339,954 params and 114 seconds. I expect 1.0 on all/seen/heldout/learned-only again, but only the harness's `eval.json` can show it. When it exists I'll report the numbers and whether each is >= iter-3. The smaller d=32 model should have fewer params, which counts as lower than iter-3 but isn't a quality regression.