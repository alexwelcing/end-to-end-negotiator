I haven't trained anything yet, so there is no new eval.json to read. The harness trains after my phase. The numbers below come from iter_005's eval.json.

- **WHAT I CHANGED:** I wrote `knobs.json` and `knobs.trainer.json` in iter_006 with epochs 60, lr 0.001, d 64, hidden 256, hole_weight 1.0. No `knobs.json` existed in iter_005 to copy, so I used the default recipe. I didn't touch the spec or the compiled circuit.
- **WHY:** The charter fixes this recipe. It reproduced 339954 params in iters 3 and 5. I will compare the new params to 339954 once the harness trains.
- **WHAT I EXPECT:** The same result as iter_005.

| Measure | Expected |
|---|---|
| all exact | 1.0 |
| seen exact | 1.0 |
| heldout exact | 1.0 |
| learned-only exact | 1.0 |
| violation_rate | 0.0 |
| per_action (propose, accept, walk) | 1.0 each |
| coverage | 2376/2376 pairs, fraction 1.0 |
| params | 339954 |
| score | 1.0 |

I'm not reacting to anything unless a margin_checker report shows a margin below 0.05.