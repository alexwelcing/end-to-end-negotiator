# The machine room: build plan

Written 2026-11-08. Paper 2 (§3, §5, §6.1–6.3) is the design; this is the order of work, with an acceptance
test per milestone. The end state: a machine built from separate exact networks runs continuously at the edge,
every hand-off checked live, shared by everyone watching, and anyone can grab a part mid-run while the rest of
the machine re-plans. Each milestone is a demo on the day it passes.

## M1. Composition in the engine (Rust, no UI)

- `emits:` on a transition: `{ event: "predicate over old and new state" }`, evaluated by the runtime after the
  step (derived, never learned). `nerve check` type-checks the predicates.
- `machine:` documents: `blocks: {instance: type}`, `wires: [{on: inst.event, do: inst.action, with: {param:
  expr}}]`, `derived:` expressions over block fields, machine-level `ensures`, a bound on events per tick.
  Instances of one block type share one network. A wire's `with:` expressions are evaluated when the event is
  raised, over the whole machine at that moment (`inst.field`, `old.inst.field` for the state before the
  emitting step, derived names), so a payload is fixed when the event is raised and the chain's list can pick
  its own front item.
- Composed oracle: a tick = an ordered list of external actions; after each, the event queue drains FIFO, each
  delivery stepping its target in the current machine state. `nerve check machines/odometer.nerve` runs the
  machine against its own `ensures` exhaustively when the reachable set is small (≤ 10⁴ states), by sampling
  otherwise.
- Runtime with networks: each block stepped by its trained network (valid candidates), events from the spec.
- **Acceptance (paper 2, Prediction 1):** the three-digit odometer, blocks stepped by the digit network trained
  on its full spec, equals integer arithmetic on all 1,000 states × 2 actions, zero violations; six digits
  agree on 10⁵ sampled increments. A carry error fails the build.
- **Status (2026-11-08): done.** `crates/nerve/src/machine.rs` holds `MachineSpec` (parse, resolve block
  types, validate wires against the blocks' `emits`), `Machine` (a stepper per block: oracle or network;
  `tick` with the FIFO cascade and the per-tick bound; machine `ensures` with `old` the state before the tick)
  and `check` (BFS over `inputs`, exhaustive to 10⁴ reachable states, random walks beyond). `nerve check`
  dispatches on the `machine:` key. Tests in that file train the digit network on its full spec and run it as
  three blocks over all 1,000 states × 4 inputs (exact, zero violations) and as six blocks over 10⁵ sampled
  increments biased towards carries (exact, zero violations), so Prediction 1 holds. Not yet: networks
  stepping continuous blocks inside a machine are untested.

## M2. A clock, a chain, and a machine view in the lab

- A `clock` block with one action, `advance`, and an event per advance; the lab drives it at a chosen rate.
- The chain of paper 2 §6.2 as `machines/chain.nerve`: toggle → digit → reorder → three-way switch → digit
  reset, all finite, five wires.
- Lab: a machine tab that renders every block's elements with per-block offsets, shows each block's spec check
  per tick (green/red), a trace of events, start/stop/step controls, and drags that detach the grabbed block
  from the clock and resume the cascade on release (§5.4).
- **Acceptance (Predictions 2 and 3):** 10⁴ ticks, zero violations, zero disagreement with the composed oracle,
  under 16 ms per tick; 1,000 scripted mid-run grabs agree with the composed oracle at k = 1, 10, 100 ticks.
- **Status (2026-11-08): done.** `specs/clock.nerve` (ten phases, `advance` emits `tick`); `emits` added to the
  toggle (`turned_on`), the reorder list (`home`: back to its initial order; the generator script emits it) and
  the three-way switch (`home`: back to r); `machines/chain.nerve` with the five wires of paper 2 §6.2, the
  list's wire choosing its front item through `with:`. Digit and clock trained through the round-2 pipeline
  (three seeds each, all exact; `experiments/round2/results.jsonl`) and added to `models/served`, now 19
  networks. Runtime: `Machine::tick_with` takes held deliveries and detached blocks; deliveries to a detached
  block are held and returned, delivered first on a later tick; `after:` ensures are checked only on quiescent
  ticks. `crates/nerve/src/machine_api.rs` serves `/api/machines`, `/api/machine/<name>/view` and `/tick`
  (networks and oracle on the same input, per-step agreement, held deliveries, derived values, tick time) and
  the page at `/machine` (`crates/nerve/web/machine.html`): panels per block in rows, wires lit as they fire,
  a check light per block per tick, run/step/reset and a rate, a trace, and grabs through the per-interface
  drag API that detach the block until release. The machine stops on the first violation or disagreement and
  shows the transition. `crates/nerve/tests/machine_chain.rs` runs the served networks: Prediction 2, 10⁴
  ticks, 25,633 block steps, all five wires fired, zero violations, zero disagreement, 0.023 ms per tick over
  the first 300 ticks (energies uncached) and 0.009 ms after; Prediction 3, 1,000 grabs with up to 24 ticks
  held (290 deliveries held in all), agreement with the composed oracle at k = 1, 10 and 100 on every grab.
  Both predictions hold. Not yet: the ball and the board (§6.4, §6.3) need continuous blocks in machines and
  bindings; the machine view has no decision-field or energy bars, which the single-interface lab has.

## M3. The core in WebAssembly, and a Pages site

- `crates/nerve` gains a `wasm` build (wasm32-unknown-unknown, wasm-bindgen): spec parse and check, oracle,
  finite model inference (candle CPU), the drag engine, the machine runtime. No tokio, no axum, no file system:
  specs and weights are passed in as bytes.
- A TypeScript wrapper with the same surface as the lab's HTTP API (`step`, `drag`, `frame`, `release`,
  `field`, `tick`), and a parity test that runs the native binary and the WASM build on the same inputs.
- A Pages site (`site/`) that replaces the lab page: every interface runs in the visitor's browser from
  `models/served`, with deep links (`?spec=hanoi&state=…`), presenter mode (scripted sequences, annotations,
  internals on/off) and recorded traces played back as demos. Deployed from GitHub with Wrangler.
- **Acceptance:** parity on every interface's exhaustive step table and on 1,000 recorded drag frames; the
  site passes the lab's smoke test in CI.
- **Status (2026-11-08): the core and the site are built; the deploy is wired.** The crate's HTTP layer
  moved behind a `server` feature (on by default; the `nerve` binary needs it) and every API route became a
  method on `serve::App`, with `App::dispatch(path, body)` routing by the same paths the server uses. The
  `wasm` feature adds `crates/nerve/src/wasm.rs`: a wasm-bindgen `Room` that takes spec texts, machine
  documents and checkpoints as bytes (`Model::from_bytes`, `ContinuousModel::from_bytes`) and answers
  `call(path, body)`. candle's CPU backend compiles for wasm32-unknown-unknown as is; the build is
  `cargo rustc --crate-type cdylib --target wasm32-unknown-unknown --no-default-features --features wasm`
  plus `wasm-bindgen --target web`, under a `wasm` cargo profile (opt-level s, LTO: 3.1 MB of wasm before
  compression; `.cargo/config.toml` sets the getrandom backend). `scripts/build_site.py` assembles `site/dist`: the two lab pages unchanged except that
  `api()` goes through `window.nerveApi` when `site/nerve-local.js` has loaded the module, the specs, the
  machines and the served networks (then 19 specs, 3 machines and 19 networks, 13.5 MB in all; now 20 specs,
  5 machines and 20 behaviour networks up front, with the 20 render networks loaded as their interfaces are
  first drawn). Verified in a browser: the machine room ticks the
  chain at about 1.6 ms per tick with full agreement and the lab's clicks and drags answer as the server's do.
  **Live at [https://nerve-machine-room.aw-ab5.workers.dev](https://nerve-machine-room.aw-ab5.workers.dev)** (the machine room at `/machine`): a Cloudflare Worker serving `site/dist` as
  static assets (`site/wrangler.toml`; Pages is now part of Workers, and the same Worker is where M4's Durable
  Object will live). Verified on the deployed site: the chain ticks by its networks with full agreement.
  `.github/workflows/pages.yml` builds and deploys on push once `CF_DEPLOY` and the Cloudflare secrets are
  set; until then the deploy is `wrangler deploy --config site/wrangler.toml` by hand. CI builds the site and
  runs the parity check on every push. Not yet:
  the parity test that runs the native binary and the wasm build on the same step table and recorded
  frames in a real browser (CI runs `scripts/wasm_smoke.mjs` in Node instead: the same module and files,
  328 answers compared with the server, none differ); deep links, presenter mode and recorded traces;
  lazy loading of networks per tab.

## M4. The shared machine room (Durable Objects)

- One Durable Object per machine instance holds the authoritative state and runs the tick in the Worker, using
  the same WASM core; viewers connect over WebSockets (hibernation API), receive state and per-tick check
  results, and send grabs and drops as external actions with a sequence number.
- Each viewer's page predicts locally with the same WASM core between server ticks, so motion is smooth, and
  reconciles on the server's state.
- Spectator and presenter roles; a presenter can pause the clock, step, inject a corrupted weight set
  ("break it") and reset.
- **Acceptance:** two browsers see the same machine; a grab in one re-plans the chain in both within one tick;
  a corrupted block lights red and the machine stops there, in both.
- **Status (2026-11-08): the shared room is live.** `site/worker.js` is the Worker behind
  [https://nerve-machine-room.aw-ab5.workers.dev](https://nerve-machine-room.aw-ab5.workers.dev): static assets for everything but `/room/<machine>/<id>`, which is one Durable Object per
  room (`MachineRoom`, SQLite-backed). The object loads the same WebAssembly core as the pages (the module
  imported into the Worker, the specs, the machine documents and the networks of that machine's blocks
  fetched from the assets binding), holds the authoritative state, ticks it on an alarm at the room's rate,
  and sends every tick, with its per-step checks and the composed oracle's agreement, to every viewer over
  WebSockets (hibernation API; the snapshot and each socket's attachment let a woken object continue). A
  viewer's grab detaches the block for everyone: the room holds its deliveries, the viewer runs the drag
  locally on its own core, and the drop reaches the room as the external action the drag committed to
  (the drag API now returns each landing candidate's state so the page can name it). Anyone with the link
  runs, pauses, steps, resets or sets the rate; the clock stops when nobody is watching, and the machine
  stops at the first failed check for everyone. On the machine room page, **Share this machine** opens a
  room (`/machine?room=<id>&machine=chain`). Verified with two browser tabs on `wrangler dev` and on the
  deployed Worker: both see the same ticks (about 1.7 ms per tick on the server); a grab in one shows as
  held in the other with its queued deliveries; the drop applies in both in the same tick with identical
  state and the composed oracle still agreeing. Not yet: spectator and presenter roles, "break it" (a
  corrupted weight set) and the local prediction between server ticks, which a machine at 6 ticks a second
  does not need.

## Pictures from weights (2026-11-08)

Rendering, which paper 1 left out of scope and the lab had been doing in JavaScript, is now part of the model.
A spec's `scene:` and its elements' `shape`/`shapes` declare what the interface looks like (rectangles,
circles, lines and text in a small bitmap font, in the lab's named colours, with `when` predicates and
state expressions for geometry), `render.rs` rasterises a state exactly (the render oracle), and `picture.rs`
trains a decoder per interface from the state encoding to a raster of at most 192 pixels on the long side:
two linear layers to a coarse grid, five nearest-upsample-and-convolve stages, a linear output clamped to
0..1 when drawn (a sigmoid output saturated on the mostly-background targets and stopped learning). The
convolutions are one im2col matrix product per layer rather than candle's `conv2d`, whose CUDA input gradient
was found to be wrong (bias gradients matched the CPU, kernel gradients did not); with that, CPU and GPU
gradients agree to four digits. The pages show the render network's picture under the landing spots and
regions, report its mean pixel error against the oracle's drawing for the state on show, and keep only
invisible hit targets where the elements are; "spec picture" shows the oracle's drawing instead. All twenty interfaces have one (mean pixel error 0.6%–3.3% against the
oracle's drawing, 7.8 M parameters in all, trained on the GPU by `runs/pictures/train_all.sh`). The lab's
stage takes the scene's shape (a wide interface is a wide picture, not a strip in a tall box), and the site
loads a render network (1–2 MB each) the first time its interface is drawn, the behaviour networks up front,
so the first load is about 13 MB rather than 40. The ten
generated specs moved from Python scripts to `nerve gen` (Rust), `nerve site` replaced `build_site.py` and
`nerve smoke` replaced `ci_smoke.py`, so the toolchain is Rust end to end (the pages and the Node parity
check remain JavaScript, which is what a browser runs).

## M5. Time at frame rate

- **Acceptance (`docs/rube-goldberg.md`, Predictions 4 and 5):** on the ball machine, position error against
  the composed oracle after 1, 10 and 100 hand-offs stays within one rail's tolerance (1% of range) with the
  snapping hand-off and grows past it within ten hand-offs without; the rail's step head agrees with descent
  on at least 99% of steps.
- **Status (2026-11-08): done.** `specs/rail.nerve` is a rail the ball rolls along (`pos` in 0..248 and
  `rolling`; `enter` puts the ball at the start, `enter_at` at an offset, `advance` moves it 8 a tick and
  emits `exit` as it leaves, `set` is the grab), with a scene of the rail and the ball. `machines/ball.nerve`
  is three rails in a loop, the clock advancing all three every tick; the hand-off wire is `exit -> enter`,
  the spec's canonical payload, so the receiving rail starts from a state it was trained on.
  `machines/ball_drift.nerve` wires `exit -> enter_at(max(0, rN.pos - 240))` and carries the sender's
  overshoot instead: the ablation. The rail network has an amortised step head (`continuous.rs`): a small
  MLP over the state, the action and its parameters, trained alongside the energy on the oracle's next state
  (MSE on the continuous values, cross-entropy on the discrete assignment), so a step is one forward pass
  rather than 60 Adam steps over 11 seeds. The spec checks the head's answer before it leaves the block
  (invariants and `ensures`, continuous fields within the tolerance); when the check fails, which happens
  where a rule has a threshold (the last step of a rail, where `rolling` flips) and a discrete guess can be
  wrong, descent's answers are tried in energy order and the first the spec accepts is taken, as the finite
  networks already choose among the spec's valid candidates; networks saved without a head step by descent.
  The check itself got stricter on the way: a continuous answer within tolerance of the truth used to be
  taken as the truth for every rule, which let a ball sit stored at `pos` 240.3 and still `rolling` (within
  1% of the true 239.9, across the threshold) and break the next step; now the types and invariants are
  checked on the state as it will be stored, and the tolerance applies to the `ensures` only.
  Agreement for a continuous block is within 1% of range
  (`machine_api::agree`; the tick JSON carries the tolerance and the machine page says "within 1%"); every
  parameterless input of a machine is a button on the machine page (**r1.enter** drops the ball) and in a
  shared room (`external`); the CSS slide between ticks is gone wherever a picture is drawn, so motion is
  the render network's pixels, tick by tick. `crates/nerve/tests/ball_machine.rs` on the served rail network:
  Prediction 4 (200 runs each, the ball dropped at a random offset): with the snapping hand-off the error against the composed oracle is mean 1.8%, worst 3.7% of the rail after 1, 10 and 100 hand-offs alike; without it, 2.4% after one hand-off, 17% after ten and off the rail entirely (164%) after a hundred. The flat 1.8% is one rail's own drift: within an episode (590,100 ticks with both balls on the same rail) the networks' ball is mean 2.3%, worst 4.4% from the oracle's, and one tick of travel is 3.2%, so about half the runs cross the end of a rail one tick before the oracle does and stay one tick ahead for ever; the plan's "within 1%" was a guess at that drift and it is a tick, not a percent. The bound holds (the hand-off error never exceeds the within-episode worst), the level is a tick. Prediction 5: on the machine's own ticks (1,213 rail steps over 400 ticks) the head agrees with descent on 1,213 of 1,213, all within tolerance of the oracle, with no answer the spec rejected, at 0.036 ms per step on the CPU; on 500 uniformly random steps (states and parameters the machine never visits) it agrees with descent on 489, 97.8%. A tick of the ball machine in WebAssembly (the clock and three rails, each rail stepped by its head) takes 0.24 ms in Node against the plan's 4 ms budget, and scripts/wasm_smoke.mjs now checks every render network's picture and the ball machine's ticks against the native lab within tolerance.

## Next: The Most Complicated Way to Turn On a Light

The plan for the Rube Goldberg machine: fifty networks turning on the toggle through self-solving puzzles, a
negotiation, a marble and a bell, grabbable in a shared room, with milestones M5–M8 and Predictions 4–7,
is `docs/rube-goldberg.md`. M5 is above; M6–M8 are next.

## Later, on the same foundation

Spec in, app out (edit a spec, check it live, train small ones in the browser, queue large ones to the foundry,
permalink the result); break-it as a public arena with counts; two-player negotiate in one session; the gallery
page per interface and the pull-request-trains-and-deploys loop.

## Architecture

Pages (site, WASM core) · Workers (API) · Durable Objects (machines, sessions) · R2 (weights by spec hash) ·
D1 (registry) · Queues (training jobs, served by the local GPU or Hugging Face Jobs) · Access (private
previews) · Analytics Engine (per-tick telemetry). One Rust implementation of oracle, model and drags, built
natively for training and to WASM for everything that runs where a visitor is.
