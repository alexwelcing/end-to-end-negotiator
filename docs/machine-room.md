# The machine room: build plan

Written 2026-11-08. Paper 2 (§3, §5, §6.1–6.3) is the design; this is the order of work, with an acceptance
test per milestone. The end state: a machine built from separate exact networks runs continuously at the edge,
every hand-off checked live, shared by everyone watching, and anyone can grab a part mid-run while the rest of
the machine re-plans. Each milestone is a demo on the day it passes.

## M1. Composition in the engine (Rust, no UI)

- `emits:` on a transition: `{ event: "predicate over old and new state" }`, evaluated by the runtime after the
  step (derived, never learned). `nerve check` type-checks the predicates.
- `machine:` documents: `blocks: {instance: type}`, `wires: [{on: inst.event, do: inst.action, with: {param:
  expr over the event's source state}}]`, `derived:` expressions over block fields, machine-level `ensures`, a
  bound on events per tick. Instances of one block type share one network.
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
  increments biased towards carries (exact, zero violations), so Prediction 1 holds. Not yet: wires with
  `with:` parameters are parsed and evaluated but no machine in the repository uses one; networks stepping
  continuous blocks inside a machine are untested.

## M2. A clock, a chain, and a machine view in the lab

- A `clock` block with one action, `advance`, and an event per advance; the lab drives it at a chosen rate.
- The chain of paper 2 §6.2 as `machines/chain.nerve`: toggle → digit → reorder → three-way switch → digit
  reset, all finite, five wires.
- Lab: a machine tab that renders every block's elements with per-block offsets, shows each block's spec check
  per tick (green/red), a trace of events, start/stop/step controls, and drags that detach the grabbed block
  from the clock and resume the cascade on release (§5.4).
- **Acceptance (Predictions 2 and 3):** 10⁴ ticks, zero violations, zero disagreement with the composed oracle,
  under 16 ms per tick; 1,000 scripted mid-run grabs agree with the composed oracle at k = 1, 10, 100 ticks.

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

## Later, on the same foundation

Spec in, app out (edit a spec, check it live, train small ones in the browser, queue large ones to the foundry,
permalink the result); break-it as a public arena with counts; two-player negotiate in one session; the gallery
page per interface and the pull-request-trains-and-deploys loop.

## Architecture

Pages (site, WASM core) · Workers (API) · Durable Objects (machines, sessions) · R2 (weights by spec hash) ·
D1 (registry) · Queues (training jobs, served by the local GPU or Hugging Face Jobs) · Access (private
previews) · Analytics Engine (per-tick telemetry). One Rust implementation of oracle, model and drags, built
natively for training and to WASM for everything that runs where a visitor is.
