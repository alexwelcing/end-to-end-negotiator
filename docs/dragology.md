# Dragology's drag algebra as an energy, and what Nerve should take from it

Written 2026-11-07 after reading the Dragology library (`src/ops/*`, `docs/AGENTS.md`) and the Declarative
Dragging paper (Horowitz, Blinn, Heer, UIST 2026). Nerve borrowed three words from it (`closest`, `between`,
`vary`) and then ignored them: `Encoder.candidates()` returns the whole state space and `futures` is decoration.
This note is what the words actually mean, why they are the same thing as Nerve's inference, and the concrete
design that follows.

## 1. What Dragology actually does

A Dragology interface is a pure function `render: State -> SVG` with drag specs attached to elements. A drag spec is
a tree of **operators**. Every operator, on every pointer frame, returns the same three things:

```
frame(pointer) -> { preview: State (or a blend), dropState: State, gap: number }
```

`gap` is the distance, in screen pixels, between the pointer and where the dragged element lands in the preview.
The operators are:

| operator | what it computes each frame |
|---|---|
| `fixed(s)` | preview and drop are `s`; gap is the element's distance in `render(s)` |
| `closest([specs], {stickiness, lockPast})` | runs every child, follows the one with the lowest gap; `stickiness` keeps the current child until another is better by that many pixels; `lockPast` locks onto a child once it has carried the element far enough from home |
| `between([states])` | renders every state, reads the element's position in each, and **interpolates the whole SVG** inside the convex hull of those positions (Delaunay or natural-neighbour weights); drop snaps to the nearest vertex |
| `vary(s0, params, {constraint, pin})` | numerical minimisation of the gap over numeric paths into the state, under inequality constraints (`inOrder`, `lessThan`, `equal`, `and`) |
| `.whenFar(bg, {gapIn, gapOut})` | switch to a background spec when the foreground's best gap exceeds a threshold, with hysteresis (`gapIn <= gapOut` or it oscillates) |
| `.withFloating()` | the dragged element follows the pointer; the rest of the scene previews the closest drop |
| `.withSnapRadius(px)` | only snap inside a radius |
| `.onDrop(f)`, `.during(f)` | post-process the drop state / the preview state |
| `.switchToStateAndFollow`, `.withChaining` | commit an intermediate state mid-drag and continue the drag from it |
| `.substate(lens)`, `.reactTo(value)`, `.dropTarget(id)`, `.withOverlay(svg)`, `custom` | lens into part of the state; rebuild the spec when a value changes; a state while hovering an element; extra drawing; escape hatch |

Two facts matter for Nerve. First, the operators have **memory**: `closest` remembers its last choice
(`lastBest`) and its lock (`lockedIndex`), `whenFar` remembers which side of the threshold it is on. A drag spec is a
small state machine of its own, run at pointer rate. Second, the author **hand-enumerates the futures** (`allOrders =
items.map(j => moveItem(items, i, j))`); nothing checks that those states are reachable or legal.

## 2. The identification: `gap` is an energy, and `closest` is Nerve's `step`

Nerve's inference is `s' = argmin over candidates of E(s, a, s')`. Dragology's `closest` is
`argmin over candidates of gap(render(s'), pointer)`. `vary` is gradient descent on the gap over continuous
fields — exactly `NerveEnergy._continuous()` with a different energy. `between` is the continuous relaxation of
`closest`: the argmin over the convex hull instead of the vertices, read out as interpolation weights.

So the two systems are the same inference with two energies that know different things:

* **`E_spec(s, a, s')`** — Nerve's energy. Knows the dynamics and the rules: which successors the spec permits
  under this gesture (compiled, exact), which the learned holes prefer, which break an invariant. Knows nothing
  about where anything is drawn.
* **`E_geom(s', p)`** = `gap` — Dragology's energy. Knows where the dragged element lands in each candidate and how
  far that is from the pointer. Knows nothing about legality or dynamics; trusts the author's list.

A drag frame in the combined system is one minimisation:

```
s'(p) = argmin_{s' in C(s, g)}   E_spec(s, g, s')  +  lambda * gap(render(s'), p)  +  E_mem(s', m)
```

where `g` is the gesture (an action with the pointer as a continuous parameter), `C(s, g)` is the candidate set the
`futures` block derives from the spec (not a hand-written list), and `E_mem` carries the operator memory
(`stickiness` is a bonus of δ pixels on the incumbent branch; `lockPast` is an infinite penalty on the others once
locked; `whenFar` is a two-state machine on the best gap with thresholds `gapIn < gapOut`). The modifier algebra
becomes an algebra of energy terms and guards, which is a thing the spec language already has words for.

What each side gains:

* Dragology gains **derived, checked futures**. The candidate list comes from the oracle (every `move(i, j)` the
  spec allows from `s`), illegal states cost infinite energy by Theorem 1, and the drag's result is scored by
  `Oracle.violations` like any other transition — a `closestAllowed` the author does not have to write.
* Nerve gains **the geometric term and the interaction dynamics** it left out of scope, in a form that is declarative
  and small, and it gains a *reason* for the learned energy in a drag: where the spec is silent (a `learn` hole, or
  several legal futures at equal gap) the learned term breaks the tie — a learned snap preference.

## 3. What to add to the spec language (`futures` v2)

```yaml
render:                                   # derived, not learned: element -> position expression over state
  knob:  { x: "30 + 60 * (status == 'on')", y: 30 }
  item:  { for: items, x: "40 * index", y: 0 }

futures:
  drag_knob:
    grab: knob
    between: { over: status, values: [off, on] }          # existing syntax, now with a render to measure
    drop_transition: elastic-out
  drag_block:
    grab: block
    closest:
      over: track                                         # one branch per value of an enum/int field
      each: { vary: { over: pos, constraint: "0 <= pos <= 200" } }
      stickiness: 8
    when_far: { gap: 50, then: stay }                     # `stay` = fixed(current state)
    floating: true
  drag_item:
    grab: item
    closest: { over: items, moves: "move(i, j)" }         # candidates = oracle.step(s, move(i, j)) for all j
    floating: true
```

Semantics, per gesture `g` with grabbed element `e`:

1. **Candidates.** `C(s, g)` is derived from the spec: `over: f, values: V` gives `{s with f := v}` for `v` in `V`
   restricted to successors the oracle reaches from `s` under some action instance of the gesture's action
   (declared or inferred); `moves:` names the action instances explicitly; `each:` nests a spec per branch.
2. **Energy.** `E = E_spec + lambda * gap + E_mem` as above, `gap` from `render`. `between` reads interpolation
   weights off the same gaps (Delaunay over element positions, as Dragology does); `vary` descends the energy over
   the named continuous field — `NerveEnergy._continuous()` already does this with `E_spec`; add the gap.
3. **Memory.** `stickiness`, `lock_past`, `when_far` become fields of a per-drag sub-record `drag: {branch, locked,
   far}` with transitions that are ordinary guarded updates on the gaps. They are part of the spec, so they are
   checked, logged and (where finite) compiled like everything else. The only inexact units in a compiled drag
   machine are the comparators on continuous gaps.
4. **Commit.** `drop` steps the oracle with the chosen action; `chaining` commits each intermediate branch change as
   its own transition, which is what makes a drag a trajectory in the episode log rather than one step.

**What exists now (2026-11-07).** `nerve/spec/drag.py` is the pure-Python version of §2: `DragSession` derives the
candidates from the oracle (`candidates()`: for every value of the dragged field, the action instance that reaches
it legally, plus `stay`), scores them with the gap from the spec's `render` block plus an optional learned energy,
and implements `closest` with `stickiness`, `between` (segment weights for two states, inverse-gap weights beyond),
`vary` (grid + golden section on the gap, then the oracle realises the value through the gesture's action, so the
spec's own `clamp` and guards apply), and `far` with hysteresis. `nerve drag specs/toggle.nerve --gesture drag_knob
30,30 55,30 70,30 95,30` prints the frames and the drop; `--model` adds a trained network's energies as the
tie-breaker. `tests/test_drag.py` pins the semantics: the toggle commits past the midpoint and not before, the
sticky dial holds its branch inside δ, `vary` tracks the pointer and stops at the spec's clamp, an ungrabbed slider
goes nowhere, `far` falls back and comes back with hysteresis, and `render` expressions go through the same
whitelisted evaluator as everything else. The four hand-written specs carry `render` blocks and `grab` elements.
Not done: `between`'s Delaunay weights for three or more states, the per-drag sub-record in the episode log,
`Encoder.candidates()` (evaluation still scores the whole space on purpose), and the Dragology-side adapter of §5.

Code that still changes for the full design: `nerve/spec/schema.py` (Future gains `grab`, `each`, `moves`, modifiers; a `Render` block),
`nerve/spec/encode.py::candidates()` (honour the futures), a new `nerve/spec/drag.py` (frame loop: candidates,
energies, memory, commit; pure Python, no DOM), `nerve/model/energy.py` (accept an extra per-candidate energy),
`edge/src/nn.ts` (same), `docs/play` (the browser build gets pointer input), and `tests/test_drag.py` (a toggle drag
commits `on` past the midpoint and not before; a sticky `closest` holds its branch inside δ; a `vary` slider tracks
the pointer within tolerance; an illegal hand-listed future is never chosen).

## 4. The research questions this opens

1. **Equivariance by type.** The factored head's shift kernel is what made `+1` the same function at every value of
   an ordinal field (paper, Prop. 2). Dragology's flagship demos are permutations (the permutation lab, reorderable
   lists, card piles). The analogue for a `list<T>` field is a permutation-equivariant head (attention or DeepSets
   over items), for a 2-D position a translation-equivariant one. The type system of the spec should pick the
   symmetry group of each field's update rule. That is a cleaner statement of what the factored head is: not a trick
   for ints, an instance of a rule.
2. **`between` in representation space.** Dragology interpolates SVG because that is all it has. The JEPA gives a
   metric on states independent of rendering; interpolating in latent space and decoding is "semantic between" —
   what lies between two list orders is a partial reorder, not a blend of pixels. Needs the rendering decoder
   (ROADMAP 27) to be seen, but the metric exists today.
3. **Learned feel.** Dragology's docs say `.withFloating()` is "usually the best feel" and default the far gap to
   50 px. Those are `learn` holes with Dragology's semantics as the fallback: record drags, fit `stickiness`, snap
   radius, `gapIn/gapOut`, branch transitions, per gesture. Ties to ROADMAP 54 (spec from trajectories): the
   recording gives states by clustering, actions by pointer events, and now the drag parameters by regression.
4. **Drags as data.** Every pointer frame is a (state, gesture+pointer, preview) triple scored by the same oracle;
   violations mid-drag are the adversary's negatives for free. The episode log grows from "one step per action" to
   "a trajectory per gesture", which is what the hierarchical JEPA (ROADMAP 22) wants.
5. **Compiling the drag machine.** `closest` with `stickiness` over a finite set of branches is a finite guarded
   state machine with continuous inputs. The discrete part compiles to exact weights today; the comparators are
   where the circuit meets the learned term. A drag spec that is *entirely* compiled, running in the Worker at
   pointer rate with no learned weights at all, is a good first demo of the combined system.

## 5. What not to do

Do not re-implement Dragology's renderer, layers, z-order and SVG interpolation. Nerve's job is the energy and the
state; Dragology's engine is a fine consumer of a `render` function and a candidate set. The right first
integration is a `DragSpecLike` adapter: a Dragology `custom` operator whose `frame()` asks a Nerve network (local
`nn.ts`) for `C(s, g)` and `E_spec`, adds the gap, and returns `{preview, dropState, gap}`. One operator file, no
fork.
