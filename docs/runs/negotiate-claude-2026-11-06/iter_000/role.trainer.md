WHAT I CHANGED: I wrote `runs/negotiate-claude/iter_000/knobs.trainer.json` with `{"epochs": 60, "lr": 0.003, "d": 64}`. I didn't touch the spec or the episode log. The log has 34,849 episodes.

WHY: The spec has no `learn` holes, so `hole_weight` is irrelevant. I couldn't run `nerve train` myself, so I used the knob values from my charter as given. The state space is tiny, so d=64 with 60 epochs on this much data should be enough.

WHAT I EXPECT: Held-out and learned-only conformance should both be high, since the transitions are simple and deterministic. Hole learning can't be measured because there are no holes. I have no eval to compare against yet. If the first eval shows weak learned-only conformance, the next lever is more epochs or more rollouts of rarely visited states.