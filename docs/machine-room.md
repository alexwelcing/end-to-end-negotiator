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
  `api()` goes through `window.nerveApi` when `site/nerve-local.js` has loaded the module, the 19 specs, the
  3 machines and the 19 served networks (13.5 MB in all). Verified in a browser: the machine room ticks the
  chain at about 1.6 ms per tick with full agreement and the lab's clicks and drags answer as the server's do.
  `.github/workflows/pages.yml` builds and deploys the site to Cloudflare Pages when the repository variable
  `CF_PAGES_PROJECT` and the Cloudflare secrets are set; CI checks the wasm build on every push. Not yet:
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

## Later, on the same foundation

Spec in, app out (edit a spec, check it live, train small ones in the browser, queue large ones to the foundry,
permalink the result); break-it as a public arena with counts; two-player negotiate in one session; the gallery
page per interface and the pull-request-trains-and-deploys loop.

## Architecture

Pages (site, WASM core) · Workers (API) · Durable Objects (machines, sessions) · R2 (weights by spec hash) ·
D1 (registry) · Queues (training jobs, served by the local GPU or Hugging Face Jobs) · Access (private
previews) · Analytics Engine (per-tick telemetry). One Rust implementation of oracle, model and drags, built
natively for training and to WASM for everything that runs where a visitor is.
