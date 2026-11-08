---
title: Nerve drag lab
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Nerve

**Software without code.** Describe an interface — as its source, as a screenshot, or in words — and get a
neural network that *is* the program. Not a model that writes the program. The network takes the interface's
state and a user action and returns the next state, and it is held to a behavior spec the way code is held to
types and proofs.

This branch starts from zero. It grew out of
[end-to-end-negotiator](https://github.com/facebookresearch/end-to-end-negotiator) — one network, one task,
supervised then self-play — and jumps three tiers:

| tier | what the network is | how it is held correct | who builds it |
|---|---|---|---|
| 0 (FAIR, 2017) | a negotiation policy | a reward | a research team |
| 1 | a policy for any spec'd interface | conformance to a spec oracle | a harness |
| 2 | energy-based / JEPA dynamics of the interface | spec invariants checked on every held-out transition; explicit transitions **compiled** into exact weights | an evolving team of harness agents |
| 3 | the running product, served at the edge | bit-identical sparse weight patches from trainer to workers | the same loop, continuously |

The thesis is simple and the system is small. Nothing here is clever; it is the obvious thing once you stop
treating software as the deliverable.

## Play with it

Seventeen interfaces, each a small neural network trained to be the program, each held to its spec as you use it.

* **Hosted:** [huggingface.co/spaces/AlexWelcing/nerve-lab](https://huggingface.co/spaces/AlexWelcing/nerve-lab),
  a Docker Space built from this repository's `Dockerfile` on a free CPU; `.github/workflows/space.yml` redeploys it
  on every push to `nerve`.
* **Locally, one command** (needs a Rust toolchain, no GPU, no Python):
  ```
  cargo run --release -p nerve -- serve
  ```
  then open http://127.0.0.1:8787. The trained networks are committed under `models/served/`, so a fresh clone
  plays immediately; `docker build -t nerve-lab . && docker run -p 7860:7860 nerve-lab` does the same without Rust.
* **Train one yourself:** `cargo run --release -p nerve -- train specs/hanoi.nerve --holdout 0` trains Tower of
  Hanoi on every cell of its spec in a few CPU minutes and prints exactness and violations.

[`docs/QUICKSTART.md`](docs/QUICKSTART.md) has the rest: training on a GPU, writing a spec, reproducing the numbers.
The current results are in [`experiments/round2/`](experiments/round2/README.md) and the plan for what comes next
is paper 2, [`docs/paper2/`](docs/paper2/README.md).

## One run

```
pip install -e . && (cd edge && npm install) && scripts/demo.sh counter
```

That does, on a laptop CPU in about a minute:

1. **check** — the spec is type-checked and run against its own invariants over the whole state space.
   A spec that breaks its own `ensures` never reaches training (we caught one in `slider.nerve` writing this).
2. **compile** — every transition the spec writes down explicitly is lowered to *exact* ReLU weights, no
   training. Energy is 0 on the true successor and positive elsewhere, by construction.
3. **rollout** — coverage, random and adversarial policies operate the oracle. Every transition and every
   spec-violating negative goes to an append-only JSONL log. A training run is a pure function of that log.
4. **train** — an action-conditioned JEPA whose prediction error *is* the energy, trained with InfoNCE over
   candidate futures and a hinge on spec-violating ones. It learns the dynamics *and* the `learn` holes the spec
   could only show by example.
5. **patch** — continued training at RL-regime learning rates produces updates that are mostly invisible in
   BF16. We ship only the bits that changed (PULSE), verified bit-identical by digest.
6. **step** — an inference worker loads the base weights, applies the patch, and is the interface.
7. **export** — weights go to a Cloudflare Worker whose entire runtime is a forward pass and an argmin.
   `nerve edge` runs that same Worker locally on **workerd**, Cloudflare's open-source Workers runtime, with
   Durable Object sessions on local disk and weights served from `edge/export/`. No account, no Python on the
   serving path. The test suite checks Python and workerd agree on every transition over real HTTP.
8. **evolve** — an EAT-style supervisor emits TEAM_CONFIGs; the harness runs the team and scores it on the
   official eval; the best checkpoint is kept.

## The spec is the only source of truth

```yaml
nerve: 1
name: counter
state:
  count: {type: int, range: [0, 9]}
  armed: {type: bool}
actions:
  inc: {}
  set: {params: {to: {type: int, range: [0, 9]}}}
  nudge: {}                                  # behaviour the spec can only show, not write
transitions:
  inc: { when: "armed", count: "min(count + 1, 9)" }
  set: { count: "to" }
invariants: ["0 <= count <= 9"]
ensures:
  - { after: inc, holds: "count >= old.count" }
futures:
  drag_dial: { kind: closest, over: count, values: [0,1,2,3,4,5,6,7,8,9] }   # dragology
learn:
  - name: nudge_towards_middle
    action: nudge
    fallback: { count: "count" }             # the network may never do worse than this
    examples: [{state: {count: 0, armed: true}, next: {count: 2, armed: true}}, ...]
```

Three ideas from the projects this is built on meet in that file:

* **ink** — typed state, pure transitions, `ensures` guards, and `learn` holes with a safe fallback. ink's
  `emit-circuit` lowers finite-type functions to exact ReLU nets; `nerve compile` does the same for transitions.
* **dragology** — an interaction is *the set of future states it may lead to*. `futures` declares them, and
  energy-based inference is literally scoring those candidates and taking the argmin.
* **qwenGPU** — weights can be *computed* by a compiler, not learned. The deterministic core of an interface is
  compiled; only the holes and the generalisation are trained.

## Results (2026-11-06, learned-only: compiled circuit disabled, 15% of transitions never seen in training)

| spec | source | states × actions | held-out exact | all exact | violations |
|---|---|---|---|---|---|
| toggle | hand-written | 2 × 1 | 1.000 | 1.000 | 0 |
| counter (with a `learn` hole) | hand-written | 20 × 14 | 1.000 | 1.000 | 0 |
| slider (gridded) | hand-written | 50 × 29 | 1.000 | 1.000 | 0 |
| slider (continuous `value`, `to`) | hand-written | 2 × ∞ | 0.998 within 1% of range | — | 0.002 |
| negotiate (Deal or No Deal, finite) | hand-written | 216 × 11 | 1.000 | 1.000 | 0 |
| push-notifications toggle | **screenshot** | 84 × 27 | 1.000 | 1.000 | 0 |
| guest stepper | **screenshot** | 8 × 3 | 1.000 | 1.000 | 0 |
| volume slider | **screenshot** | 404 × 103 | 1.000 | 1.000 | 0 |

Each row is one Hugging Face Job on an `a10g-small` (34 s to 9 min), 300 epochs, AdamW weight decay 0.05, the
factored energy at weight 10, one seed. Toggle has two (state, action) cells and no held-out set (specs under 20
cells train and score on everything). The held-out sets are 34 cells (counter), 213 (slider) and 367 (negotiate).
The three screenshot rows went image → spec (one or two rounds of `nerve check` feedback to the agent, no hand
edits) → exhaustive log → network.

Re-run on a two-core CPU with the same configuration and three seeds (`docs/paper/`), held-out exact match per seed:

| spec | held-out cells | seed 0 | seed 1 | seed 2 | violations |
|---|---|---|---|---|---|
| toggle | n/a (2 cells) | 2/2 | 2/2 | 2/2 | 0 |
| counter (with hole) | 34 | 33/34 | 33/34 | 34/34 | 0 |
| slider (gridded) | 213 | 213/213 | 213/213 | 212/213 | 0 |
| negotiate | 367 | 367/367 | 367/367 | 367/367 | 0 |

One seed is not a result; the misses are single cells next to a range boundary (`inc` at count 8, `arm` at count 9,
one slider cell). The ablations in `docs/paper/` say where the factored head matters: on the shift-rule specs as more
is held out (slider with 30% of cells held out: 436 → 448 of 449; counter: 67 → 71 of 76) and not on negotiate,
which the JEPA term alone recovers exactly from half of its 2,376 cells. The continuous-slider row did not
reproduce from the default log on CPU (0.915 within tolerance, 8% violations); that row's log is not committed.

What it took to get the last cells: ordinal features beside one-hots; an **equivariant factored energy term**
(per-field next-value logits from the action and the *other* fields, with a shift kernel indexed by offset from
the field's current value, so "+1" learned at 3 is the same function at 8) weighted to outvote the JEPA term;
exhaustive logs so data gaps stop looking like generalisation failures; and the Worker holding the same weights
bit for bit (checked by digest) and computing the same argmin (energies agree to 0.05 in the parity test; the
Worker's GELU is the tanh approximation, Python's is exact). Two bugs mattered more than any idea: a variance term on a batch of one sample produced a
NaN that silently zeroed a whole run, and training and evaluation disagreed on hold-out for tiny specs.

## Why energy-based / JEPA

Software is a state machine. A state machine is a scoring function over (state, action, next-state) with a
unique minimum. That is an energy. A JEPA predicts in representation space instead of reconstructing pixels or
DOM, which is the right abstraction level for "what the interface does" as opposed to "what it looks like" —
rendering is a separate, replaceable decoder. The energy is a sum of three terms: the JEPA prediction error,
a factored per-field term (an equivariant shift kernel on ordinal fields plus an absolute term, both blind to the
field's own value), and the compiled circuit, which is exact where the spec is explicit and constant (silent)
where it is not. The circuit's energy is the Hamming distance to the true successor over the number of fields,
and its weight is raised per query above that number times the spread of the learned term, so every explicit
transition is decided by the circuit whatever the learned weights say (`tests/test_circuit_dominates.py` checks
this with random and hostile weights). Inference is argmin over candidate futures; for continuous fields it is gradient descent on
the energy from a handful of seeds.

Conformance is measured on BF16 weights that ship, never on the FP32 training copy. A number the inference
worker cannot reproduce is not a result.

## The team that built the last one

`nerve evolve specs/negotiate.nerve --mode claude --remote hf` ran 11 iterations for $14.23: a Claude Code supervisor
emitting TEAM_CONFIGs, Claude Code role agents in parallel per phase, one GPU Job per iteration. It reached 1.0 on
every metric, invented four roles that did not exist at iteration 0 (`split_prober`, `spec_verifier`,
`margin_auditor`, `margin_checker`), and, once the score was saturated, restructured the team to cut its own cost
per iteration. The full record (every TEAM_CONFIG, eval and role report) is in `docs/runs/negotiate-claude-2026-11-06/`.

## Machines

Blocks wire into machines without a line of code between them (paper 2, `docs/paper2/`; plan and status in
`docs/machine-room.md`). A transition may `emits:` an event, a predicate over the old and new state that the
runtime evaluates after the step; a `machine:` document names block instances, the external `inputs`, `wires`
from one block's event to another's action, `derived` values and machine-level `ensures`. Instances of one block
type share one network. `machines/odometer.nerve` is three copies of `specs/digit.nerve` and two carry wires:

```bash
./target/release/nerve check machines/odometer.nerve
```

runs the composed spec against its own `ensures` over all 1,000 reachable states; `machines/odometer6.nerve`
has a million and is checked by random walks. The tests in `crates/nerve/src/machine.rs` train the digit
network on its whole spec and run it as the three blocks (exact on every state and input, zero violations) and
as six (exact on 10⁵ sampled increments), which is paper 2's Prediction 1.

`machines/chain.nerve` is the machine that runs itself (paper 2 §6.2): a clock drives a toggle, the toggle
drives a digit, the digit's wrap rotates a reorder list, the list coming home turns a three-way switch, and
the switch coming back to r resets the digit. Five blocks, five networks (four of them round 2's, unchanged),
five wires. The lab serves it at **`/machine`**, the machine room: every block that steps is checked against
its spec each tick and the whole machine against the composed oracle, and you can grab a list item or the
switch's knob while it runs; the grabbed block is held, its deliveries queue, and the cascade resumes when you
let go. `crates/nerve/tests/machine_chain.rs` runs the served networks for 10⁴ ticks (zero violations, zero
disagreement, about 0.02 ms per tick) and through 1,000 scripted mid-run grabs (agreement with the composed
oracle 1, 10 and 100 ticks after every release): paper 2's Predictions 2 and 3.

## Paper

`docs/paper/` holds the manuscript (NeurIPS format, 9 pages + appendix), its sources, the CPU sweep that re-derives
the results table with seeds and ablations (`sweep.py`, `results.jsonl`), and `make_tables.py`, which regenerates
every number in the paper from `results.jsonl`. `docs/paper/GUIDANCE.md` is the list of what the paper exposed in
this branch and what to do next.

## Layout

```
specs/*.nerve            behavior specs (toggle, counter, slider, negotiate)
machines/*.nerve         machines: block instances wired by events (odometer, odometer6)
nerve/spec               schema, safe expression language, parser, oracle, self-check, exact circuit compiler,
                         drag.py: gestures as energy minimisation over spec-derived futures (docs/dragology.md)
nerve/data               append-only episode log; coverage / random / adversarial rollouts; spec-derived negatives
nerve/model              action-conditioned JEPA; combined energy (circuit + learned)
nerve/train              training loop; official eval (all / held-out / learned-only / holes / violations)
nerve/sync               PULSE sparse value patching: BF16 compute-visibility gate, delta-varint indices, zstd, digest
nerve/serve              inference worker (applies patches, HTTP), edge export
nerve/evolve             EAT loop: TEAM_CONFIG, offline and Claude supervisors, herdr/subprocess launcher
nerve/intake             code | screenshot | prose -> spec, via Claude Code or DeepSeek Harness, validated by `nerve check`
edge/                    Cloudflare Worker + Durable Object sessions + R2 weights; nn.ts is the whole runtime
edge/workerd.capnp       the same Worker on workerd locally (disk-backed weights + DO storage); `npm run local`
harness/dsh              DeepSeek Harness profile patch (providers, allowed tools, system prompt)
harness/eat              EAT skill + role prompts (spec_auditor, coverage_hunter, adversary, trainer, energy_auditor, hole_tutor)
harness/herdr            how the team lives on herdr
```

## The loop at scale

```
                 code / screenshot / prose
                           │  nerve intake   (Claude Code on your subscription, or dsh)
                           ▼
                    specs/<name>.nerve  ──── nerve check ──── ✗ back to the agent
                           │
         ┌─────────────────┼──────────────────┐
         ▼                 ▼                  ▼
   nerve compile     nerve rollout      EAT supervisor  ──►  TEAM_CONFIG  ──►  roles on herdr panes
   (exact weights)   (episode log)             ▲                                      │
         └────────────────┬────────────────────┴──────────── official eval ◄──────────┘
                          ▼
                    nerve train  (GPU box, a herdr "machine")
                          │  nerve patch  (PULSE: bit-identical sparse patches; 92% of BF16 values unchanged
                          │               after an epoch at lr 3e-6 on negotiate, 9x smaller than the checkpoint)
                          ▼
            R2 nerve-weights/<spec>/{edge.json, patches/*.nrvp}
                          │
                          ▼
     Cloudflare Worker + Durable Object  ◄──  users   (or a Container / `nerve serve` for big networks)
     └─ identical Worker on workerd locally:  nerve edge specs/x.nerve --model runs/x/v2/model.pt
```

`--mode claude` makes the supervisor and the roles real `claude -p` processes (the way EAT's are); `--mode offline`
runs a rule-based reflector so the loop is reproducible with no model at all, which is what the tests use.

## Status

Working, small, honest. Finite specs up to a few hundred thousand (state, action) pairs compile; larger
interfaces are carried by the learned model alone. Rendering (state → pixels) is deliberately out of scope for
this cut: the network is the behaviour, and behaviour is what the spec can check.

What is next, in order: (1) intake from screenshots end to end with a vision model and a `nerve check` loop,
(2) continuous-state fields with `vary` futures via Langevin steps instead of argmin over a grid, (3) a
WebGPU inner loop in `edge/src/nn.ts`, (4) a real `herdr` socket client once its API is stable.
