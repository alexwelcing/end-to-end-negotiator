# The Rube Goldberg machine: a deep dive on getting there with models and no code

Plan written 2026-11-08, after machine-room milestones M1–M4 (docs/machine-room.md) and the move of
rendering into the models. It says what "no code" means for a machine, what we have, what is missing
for a machine that runs like a Rube Goldberg contraption under a visitor's hands, and the milestones,
with acceptance tests and predictions registered in advance, to get there.

## 1. The promise, made precise

An interface here is a spec and a network. The spec is a declaration (state, actions, transitions,
invariants, `ensures`, `render`, `futures`, `emits`, `scene`); the network is trained from the spec's
own oracle and checked against it on every step. A machine is a document of block instances, inputs
and wires. Nothing in a spec or a machine document is code: there is no program to call, no handler to
write, no frame loop per machine.

What *is* code is the runtime, and it is generic: it never knows which machine it runs.

| Part | Generic runtime (code, written once) | Per interface or machine (declared or learned) |
|---|---|---|
| Behaviour | oracle evaluation, spec check, tick loop, event queue | transitions, `emits`, wires: declared; the step: learned |
| Look | rasteriser (the render oracle), raster compositing, hit targets | `scene` and shapes: declared; the pixels: learned |
| Hands | drag minimisation over futures, bindings interpreter, detach/queue | `futures`, `bindings`: declared; the energy tie-break: learned |
| Hosting | lab pages, wasm module, Durable Object | nothing |

The rule for every milestone below: if a demo needs a line in a `.rs` or `.js` file that names the
demo, the design is wrong. The author writes specs and a machine document, runs `nerve train`,
`nerve train-picture` and `nerve check`, and the machine runs.

## 2. Where we stand

- Behaviour: 19 interfaces, one network each trained on its full spec; all 14 finite ones exact, the
  continuous ones within 1% of range on 99.6–100% of transitions (linked sliders excepted, 10%).
- Composition: `emits` on transitions, `machine:` documents, the composed oracle, a tick with a FIFO
  cascade and a per-tick bound, held deliveries for grabbed blocks, `with:` payloads evaluated over the
  machine when the event is raised. The odometer (Prediction 1) and the chain (Predictions 2 and 3)
  hold on the served networks.
- Look: `scene`/shapes in every spec, the render oracle, a render network per interface (in training as
  this is written), pages that draw the network's pixels and report the pixel error.
- Hands: drags as minimisation over spec-derived futures with the network's energy as tie-breaker,
  for every drag kind Dragology has; grabs inside a running machine detach the block and queue its
  deliveries; clicks are actions. No keyboard yet.
- Hosting: the lab on a server, the same lab on WebAssembly at the edge, a shared room per machine on a
  Durable Object; two browsers see and drive one machine.

## 3. What a Rube Goldberg machine needs that we do not have

A contraption is continuous motion handed from block to block, drawn as one scene, that a hand can
interfere with anywhere. Five things are missing.

### R1. Time at frame rate, inside the model

Today the chain ticks at 6 per second and the pages slide elements between ticks with a CSS
transition. That slide is code pretending to be motion. In a machine of models, motion is state: a
ball block has `pos` and `v`, its `advance` transition moves it by `dt`, the clock ticks at 30–60 Hz,
and the render network draws every tick. Nothing interpolates; the model is the animation.

Consequences:
- Continuous blocks must step at frame rate. The continuous trainer answers a step by Adam descent (60
  steps, 11 seeds): milliseconds on a GPU, too slow for 60 Hz across several blocks in WebAssembly.
  The fix is an *amortised step head*: a small network trained alongside the energy that predicts the
  next state directly, checked every tick by the energy (its answer must be the energy's argmin within
  tolerance) and by the oracle; descent remains the fallback when the check fails. The check is what
  keeps it honest, as everywhere else.
- Agreement for continuous blocks is "within 1% of range", so the machine room's agreement pill
  becomes a tolerance, shown as such.
- The CSS slide goes away wherever a picture is drawn: if a block looks like it jumps, its tick is too
  coarse, and that is the model's problem to fix, not the page's.

### R2. The hand-off rule (paper 2 §5.4, Prediction 4)

Continuous error compounds across blocks. A hand-off is a discrete event whose payload is the spec's
canonical value, not the sender's float: the ball enters the next track at that track's entry
position with a quantised speed (`with: {v: "round(ball.v)"}` or a declared set of speeds), and the
receiving block starts from a state it was trained on. Inside a block the motion is continuous and
carries its error for one episode; across a hand-off the machine is exact again. The ball demo
measures drift with and without the rule.

### R3. One scene: layout and ports

The machine room draws each block in its own panel. A contraption is one picture. The machine document
gains:
- `place:` per block, `{x, y, scale}` in the machine's own coordinates, so block pictures composite
  onto one canvas (the page composites rasters, generic code);
- `ports:` on blocks, named points in the block's coordinates (`exit: [240, 0]`, `entry: [0, 0]`),
  so a wire can be drawn as the path the ball takes and the page can show a hand-off as the ball
  leaving one picture where the next begins;
- a machine-level `scene:` for the backdrop (a table, a wall), rendered by the oracle (static).

The render networks still draw every block; the page only places the pictures where the machine says.

### R4. The blocks of a contraption

Each is a spec with a `scene`, trained on its full spec for behaviour and picture:

| Block | State | Actions / events | Notes |
|---|---|---|---|
| marble on a ramp | `pos`, `v` (continuous) | `advance(dt)`; emits `exit` at the end with `v` | the ball demo's block; slope as a constant in the spec |
| track (have it: `tracks`) | `track`, `pos` | `move`; emits `exit` | three rails; a grab moves the block between rails |
| domino row | `fallen` (0..n), `pushed` (bool) | `advance`; `push`; emits `last_falls` | finite; one domino per tick |
| seesaw | `angle` (−1, 0, 1), `load` | `drop_left/right`; emits `tips` with the side | finite |
| bucket | `fill` (0..k) | `add`; emits `full` | the counter, renamed |
| gate | `open` (bool) | `toggle`; a grab flips it | the toggle; a grab mid-run is the user's interference |
| fan / windmill | `phase` | `advance`; emits `tick` every n | the clock with a picture |
| catapult | `armed`, `loaded` | `load`; `fire`; emits `launch` | one-shot, re-armed by a later event |
| bell | `rings` | `ring` | terminal: the run ends when it emits `rang` |

Ten blocks, ten networks for behaviour, ten for pictures, all under a few minutes of GPU each. The
machine document wires them: marble exit → track entry, track exit → dominoes push, last domino →
seesaw drop, seesaw tips → bucket add, bucket full → catapult fire, launch → bell ring. The gate sits
on the track: closed, the ball stops (the track's `move` guard reads the gate through a wire payload or
the gate's `emits`), and the visitor opens it.

### R5. Hands everywhere, and keys

- Grabs: every block's futures work inside the machine already (detach, queue, drop as an external
  action). The contraption adds grabs that change the *course*: move a track to another rail, flip the
  gate, lift the seesaw. Those are ordinary `futures`.
- Keys: `bindings:` in the spec (paper 2 §5.2) map keys and modifiers to actions with parameters from
  the pointer, focus or selection; the runtime interprets the table. Space to drop the marble, R to
  reset, arrows to nudge a selected block.
- Placing and rewiring live: a machine document is text; a visitor who edits it (a textarea, generic)
  re-parses and re-runs. No code, and the most direct demonstration that the machine is its document.
- Shared: all of this already works in a room; a contraption with two people's hands in it is the
  Rube Goldberg version of the chain demo.

## 4. Milestones

Each milestone ships in the lab and on the site the day it passes, with its numbers in this file.

### M5. Time at frame rate (R1, R2)

- `machines/ball.nerve`: three `tracks` blocks in a row (the paper's ball demo), the clock at 30 Hz,
  hand-offs as events with the canonical payload.
- The amortised step head for continuous blocks, trained with the energy, checked per tick.
- The machine room's agreement pill as a tolerance for continuous blocks; the CSS slide removed where a
  picture is drawn.
- **Acceptance (Prediction 4):** over 200 runs, position error against the composed oracle after 1,
  10 and 100 hand-offs is bounded by one block's within-episode error (≤ 1%) with the rule, and
  grows past 1% within ten hand-offs without it. **Prediction 5 (new):** the step head agrees with
  descent within tolerance on ≥ 99% of ticks, and a tick of three continuous blocks takes under 4 ms
  in WebAssembly.

### M6. One scene (R3)

- `place:` and `ports:` in machine documents; a machine-level `scene:`; the page composites block
  pictures on one canvas and draws wires as paths between ports.
- The chain and the ball machine drawn as single scenes.
- **Acceptance:** a hand-off reads as continuous on screen: the ball's picture leaves one block at its
  exit port on tick t and appears at the next block's entry port on tick t+1, at every hand-off of
  100 runs (checked from the rasters' element positions, not by eye).

### M7. The contraption (R4)

- The ten blocks of R4 as specs with scenes, trained on their full specs (behaviour and picture), and
  `machines/rube.nerve` wiring them from marble to bell with the gate in the way.
- **Acceptance:** 100 runs from the dropped marble end with the bell, zero violations at any block,
  every continuous block within tolerance, every hand-off exact; in 100 further runs a scripted hand
  flips the gate or moves the track mid-run and the composed oracle's prediction from the dropped
  state is matched at k = 1, 10, 100 ticks (the chain's Prediction 3, on the contraption).

### M8. Keys and the desktop (R5)

- `bindings:` in specs and the runtime's interpreter; selection and focus as blocks (paper 2 §6.3).
- The contraption under keys: drop, reset, nudge; the board demo's traces.
- **Acceptance (Prediction 6, the paper's board prediction):** runtime and composed oracle agree on
  1,000 replayed input traces of length 50, and every input is answered within the frame.

## 5. What could go wrong

- **Continuous exactness.** Linked sliders sit at 10% within tolerance. The contraption's continuous
  blocks have one continuous field each (position) plus a small discrete one (speed class), which is
  the regime that reaches 99.6–100%. If a block needs two coupled continuous fields, split it.
- **Picture sharpness.** Render networks at 192 pixels on the long side reach 0.7–3% mean pixel error
  so far; edges are soft. More steps and a perceptual term help; so does a smaller raster per block,
  since a contraption shows ten blocks at once. The oracle's picture is always one click away, and the
  error is always on screen.
- **Speed in the browser.** A render network forward is 30–100 ms in WebAssembly today. At 30 Hz with
  ten blocks that is too slow; the plan is smaller block rasters, drawing only blocks whose state
  changed this tick (most do not), and a WebGPU path later. The behaviour networks are already fast.
- **Synchronous time with several moving things.** Two balls in one tick are two external advances
  and their cascades, in order; the paper's tick semantics covers it, the tests will say whether the
  order ever matters.
- **Authoring friction.** Ten new specs is ten rounds of `nerve check`, `train`, `train-picture`. The
  tooling is there; the risk is only our patience, and a `nerve machine new` that scaffolds a block
  spec from a template would remove most of it.

## 6. What to measure and write down

Predictions 4, 5 and 6 above join Predictions 1–3 (paper 2) in a results table in this file as the
milestones land, with the same discipline: registered before the run, falsifiers named, numbers
generated from committed logs. If the contraption works, it is paper 3: a machine of thirty networks
that a visitor can reach into, with no code between the parts.
