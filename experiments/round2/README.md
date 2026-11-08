# Round 2: ten Dragology demos

Written 2026-11-07, before any round-2 model was trained. All runs: Rust trainer on the local GPU, 300 epochs,
AdamW weight decay 0.05, factored head on, answers from valid states (the new default). Scripts in this folder;
results in `results.jsonl`.

## Questions and predictions

**Q1. Hanoi's written rules.** Are the `move` rules the program on cells the network never saw?
Prediction: ≥ 99% held-out exact at 15% and 50% held out, no violations. Would change our minds: < 95%.

**Q2. Hanoi's learned solver (the main question).** `hint` has no rule, only one example per position (the
optimal next move toward peg 2). Scored on positions whose example the network never saw.
Prediction: about 85% at 15% held out, 55% at 50%, 30% at 70%. Holding out the whole region where the largest
disk is on the middle peg (27 positions, `--holdout-rule "d4 == 1"`): under 30%, because the right move there
depends on the recursive structure, not on nearby examples. Would change our minds: ≥ 80% on that region, which
would mean it learned something like the recursive rule.

**Q3. Do wrong hints stay legal?** The `ensures` on `hint` says a hint is one legal move or none.
Prediction: ≥ 95% of held-out hints are legal even when they are not optimal, because legality is learned from
the 729 written `move` cells. Would change our minds: under 80%.

**Q4. Kanban at scale (360 boards among 20,736 slot tuples).** Spec-assisted vs network on its own.
Prediction: spec-assisted ≥ 98% at 50% held out and ≥ 85% at 85%; the network on its own ≤ 90% at 50% with
≥ 5% violations, and under 40% at 85%. Would change our minds: the network on its own within 2 points of
spec-assisted at 85%, which would mean the invariants are learned, not just enforced.

**Q5. Seed spread.** Standard deviation across three seeds ≤ 1.5 points at 50% held out, for every finite demo.
Would change our minds: any demo above 3 points, which would make single-seed tables from round 1 unreliable.

**Q6. The other eight demos.** Each is the program at 15% held out (≥ 99% held-out exact, spec-assisted). The
ones with constraints between several objects (sokoban pushing, rush-hour blocking, scheduler overlaps) are
the first to fall below 90% at 50% held out.

## Change of goal, mid-round (2026-11-07)

Each interface gets its own network, trained on that whole spec, so generalising to cells it never saw is not
the goal. The question that matters is whether a network trained on the whole spec **is** the program: exact on
every transition, no rule broken, reliably across seeds, in reasonable time on the local GPU. Q1–Q6 above asked
about generalisation, so they are set aside; their partial results stay in `results.jsonl` (tags `h0.15`,
`h0.5`, `h0.7`, `h0.85`) as a side result. Continuous specs cannot be trained on every case, so they are still
scored on fresh random cases.

**Q7 (replaces Q1–Q6).** Trained on the full spec (`--holdout 0`), every finite demo reaches 100% exact on all
its transitions with no violation, on every seed tried; Hanoi reproduces all 81 solver examples; linked sliders
reaches ≥ 99% within 1% of range. Would change our minds: any finite demo below 100% after 300 epochs, which
would mean the network cannot represent that rule exactly and needs more capacity or a different head.

## Results

### Q1–Q6 (hold-out sweeps; stopped at the goal change, one seed each)

From `nerve summarize`, tag `h*` and `rule-*` rows. Predictions were far off.

| question | predicted | measured |
|---|---|---|
| Q1 hanoi written rules | ≥ 99% held-out exact at 15% and 50% | 94.8% at 15%, 95.2% at 50% (93.9% on salt b), 93.2% at 70%; 0–0.5% violations |
| Q2 hanoi solver on unseen positions | ~85% / 55% / 30% at 15/50/70% | **13.3% / 5.4% / 5.3%**; region `d4 == 1`: **0%** (predicted under 30%) |
| Q3 wrong hints stay legal | ≥ 95% legal | 93% at 15%, 95% at 50%, 75% at 70%, 81% on the region |
| Q4 kanban at 85% held out | spec-assisted ≥ 85%; network alone < 40% | spec-assisted 88.0% (0.2% violations); network alone 70.6% (20.6% violations) |
| Q5 seed spread | sd ≤ 1.5 points | not measured (one seed per split before the stop) |
| Q6 other eight demos | ≥ 99% at 15% | not run |

What this says: the network memorises what it trains on (100% on every seen cell and example) and does not
generalise a solver from examples (the opposite of Q2's prediction, by about an order of magnitude). It is a
side result now that the goal is full-spec training, but it is the honest answer to "did the hole learn the
rule": no, it learned the examples.

### Q7 (full-spec training)

The sweep is complete (2026-11-08). All 14 finite interfaces are 100% exact with 0 violations on every seed
trained (three seeds each, one for kanban and scheduler); `nerve summarize` prints the table and `docs/paper2/` the
generated version. The scheduler, the largest at 41,472 cells and 165,524 parameters, was exact from its first
evaluation. The continuous specs:
`tracks` 99.6% within 1% of range (0.1% violations), `slider_continuous` 100% (0), and **`linked_sliders`
10.4% within 1%, mean error 7.5% of range, 89.6% violations**. Q7's "≥ 99%" fails for linked sliders: three
coupled continuous fields that move and stop together, with `ensures` holding the group's shape to 0.001, are
beyond the descent-from-eleven-seeds inference as it stands. This is the round's one falsified prediction so
far and the paper reports it as such.
