# Dragology's demos, ranked for Nerve

Source: [`declarative-dragging/dragology/src/demos`](https://github.com/declarative-dragging/dragology/tree/main/src/demos),
listed 2026-11-07: 115 entries (106 demo files and 9 folders). For the 12 marked † I read the state type in the
source. The rest are placed by name and size and should be checked before anyone builds them.

## What makes a demo hard for Nerve

Three questions decide it, in this order.

1. **Can today's spec language write the state?** It has bools, enums, bounded ints, gridded and continuous floats.
   It has no lists, records or text (ROADMAP 6–7), so a list has to become one position field per item, as
   `specs/reorder.nerve` does.
2. **Can the finite trainer enumerate the states?** It scores every state as a candidate successor. Thousands of
   states are fine (reorder has 3,125 position tuples, 120 of them valid). Millions are not, until candidates are
   restricted to a gesture's futures (`Encoder.candidates()`, not done).
3. **What shape is the update rule?** "+1 on a number" is solved by the factored head. Reordering, and legality
   that depends on other fields, are what the demos below test.

## Built in Nerve so far

| Dragology demo | Nerve spec | What it tests |
|---|---|---|
| `study-switch`† (by name) | `toggle` | `between` over two states |
| `study-slider`, `linear-track`, `constrained-slider` | `slider`, `slider_continuous` | `vary`, a guard on the drag's action |
| `stickiness` | `counter` | `closest` with stickiness, plus a `learn` hole |
| `lock-past`, `snap-to-islands` (by name) | `tracks` | `closest` with `each` vary per branch, `floating`, `far` |
| **`perm`†** | **`reorder`** | **Dragology's permutation demo: 5 items, futures = `moveItem(items, i, j)`** |

### Reorder: how far the network generalises (2026-11-07, local GPU)

300 epochs, AdamW weight decay 0.05, seed 0. "Held out" is the fraction of the 3,000 legal (order, move) cells the
network never trains on; it is scored on exactly those.

| held out | trained on | factored head on | factored head off |
|---|---|---|---|
| 15% | 2,574 | **426/426**, 0 violations | 415/426 (97.4%), 0.4% violations |
| 50% | 1,514 | 1407/1486 (94.7%), 2.6% violations | 1273/1486 (85.7%), 7.1% violations |
| 70% | 919 | 1283/2081 (61.7%), 26.6% violations | 948/2081 (45.6%), 37.7% violations |
| 85% | 455 | 940/2545 (36.9%), 53% violations | — |
| 90% | 298 | 806/2702 (29.8%), 63% violations | — |

At 15% it is the program, like every other spec. With little data it was the hardest one so far: at 70% held
out, negotiate still scores 100%, slider 98% and counter 83%. The violations were the telling part. The trainer
scored all 3,125 position tuples, and only 120 are orders, so a starved network picked states where two items
share a place. The factored head helps (85.7% → 94.7% at 50%) but was built for "+1 on a number", not for "is a
permutation".

**The fix: answer only with states the invariants allow.** The spec already knows them, so no answer can break
an invariant. This is [dragology.md](dragology.md) §2's "illegal states cost infinite energy" applied to the
network's own step. `nerve eval --valid-only` rescores any checkpoint that way, and finite training now
defaults to it (`--all-states` restores the old behaviour; the Python reference still scores every state).

| held out | trained and answered over all 3,125 | same networks, answering over the 120 orders | retrained over the 120 orders |
|---|---|---|---|
| 15% | 426/426 | 426/426 | (already perfect) |
| 50% | 94.7%, 2.6% violations | **99.9%**, 0 violations | 99.9%, 0 violations |
| 70% | 61.7%, 26.6% violations | **95.3%**, 0.13% violations | 95.2%, 0.10% violations |
| 85% | 36.9%, 53% violations | **78.5%**, 2.1% violations | 78.3%, 2.2% violations |
| 90% | 29.8%, 63% violations | **68.7%**, 5.3% violations | 68.2%, 6.1% violations |

The whole gain comes from where the network may answer, not from what it learns: a network trained over every
tuple already ranks real orders correctly, and retraining over the 120 adds nothing. With the factored head off,
the same restriction lifts 70% held out from 45.6% to 90.1%. What is left at 85–90% is a legal order that is the
wrong one, sometimes with the moved item not where it was dropped (the remaining violations are that `ensures`).
That is the part a permutation-equivariant head ([dragology.md](dragology.md) §4, question 1) would have to fix:
the same update rule whichever item moves.

## Next: doable with today's language, and hard

| demo | state in Dragology | Nerve encoding | why it is hard |
|---|---|---|---|
| **`hanoi`†** | `pegs: number[][]`, 3–4 disks | peg per disk: 3⁴ = 81 states | legality depends on other fields: Dragology writes `amb([0, 1, 2])` plus `require(newPeg[0] < newPeg[1])`, an oracle inside the drag spec. Illegal drops exist, unlike reorder |
| `kanban`† | `columns: { cards }[]` | column and position per card | a 2-D reorder: moves between columns shift two lists |
| `study-reorderable-list`† | `items: string[]`, 4 items | as reorder | subsumed by `reorder` |
| `tromino`† | `missingSquare: Vec2`, board level | the square's coordinates | the state is tiny; the tiling is all render |
| `insert-and-remove` (by name) | — | count plus positions | a list whose length changes |

## Needs restricted candidates or a structured state

| demo | state in Dragology | blocker |
|---|---|---|
| `fifteen`† | 4×4 tile positions | 16!/2 ≈ 10¹³ states. Even the 3×3 version has 181,440 |
| `rush-hour`† | cars `{x, y, w, h, dir}` on 6×6 | large state space; a move slides one car until it is blocked |
| `sokoban`† | player plus boxes, walls and goals on a grid | large state space; pushing chains through two objects |
| `card-piles`† | piles of cards with continuous offsets | continuous positions plus grouping that merges and splits |
| `lists-in-lists`†, `lists-in-lists-vanilla` | rows of tiles or nested rows | a recursive list type |
| `todo`† | todos with free text | text fields |
| `lists-2d`, `lists-2d-sizes`, `canvas-of-lists`, `outline`, `two-kinds-of-tree`, `sprouting-tree` | (by name) | lists of lists and trees |

## Mathematical structure editors

The largest files: rich algebra on words, diagrams and tilings. They are the "hardest" in Dragology's own terms,
and each needs a structured state type before Nerve can write its spec.

`permutation-lab`† (64 KB; a word of letters and closed columns, modes, views), `braid` (72 KB), `braid-old`,
`graspable-math`, `tableaux`, `catalan-explorer`, `zx-string-diagram`, `frobenius-algebra`, `frobenius-tangle`,
`aztec-diamond`, `plane-partition`, `knot/`, `lambda/`, `proof-by-dragging/`, `regex-nfa/`,
`rotations-rebalancing/`, `twisted-trees/`, `git-graph-surgery`, `graph`, `nodes-and-noodles`, `stock-and-flow/`,
`tiles`, `tactile-tessellations`, `grid-poly`, `grid-poly-vary`.

## Continuous geometry and layout

Need continuous training with several continuous fields (done for one field in `tracks` and `slider_continuous`).

`bezier`, `rectangle`, `robot-arm`, `gears`, `scene-3d`, `layout-editor`, `gantt`, `scheduler`, `pivot-table`,
`face-creator`, `multi-circle-points`, `multi-circle-points-vanilla`, `point-on-curve`, `point-in-circle`,
`point-in-polygon`, `point-on-circle-via-constraint`, `orbiting-planet`, `orbiting-planet-with-background`,
`dragon`, `spinny`, `stretchy-rot`, `stretchy-xy`, `angle`, `angle-via-transform`, `clock`, `linked-sliders`,
`ring-of-beads`, `chain-of-links`, `carousel`, `simple-triangle`, `simple-triangle-custom`.

## Drag mechanics and studies

Mostly about how a drag feels, not about the state. Several map onto operators Nerve already has.

`study-rotary-dial`, `study-switch-clickable`, `study-three-way-switch`, `study-timeline`,
`study-timeline-embed`, `study-timeline-substate`, `error-study-reorderable-list`, `perm-floating`,
`perm-double`, `floating-dynamic-switch`, `weird-floating-vs-fixed`, `magnetic-targets`,
`snap-to-islands-in-out`, `linear-track-chained`, `between-with-vary`, `switch-with-sharpness`,
`with-pointer-delay`, `alt-to-copy`, `drag-to-copy`, `insert-and-remove-original`, `emerge-from`,
`change-frame`, `click-and-drag`, `simplest-clicker`, `text-box`, `lerping-keys`, `lerping-text`,
`component-elements`, `component-elements-lerp`, `controlled`, `drag-spec-designer`, `animate-algebra/`,
`bluefish/`.
