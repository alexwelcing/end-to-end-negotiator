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

## Why energy-based / JEPA

Software is a state machine. A state machine is a scoring function over (state, action, next-state) with a
unique minimum. That is an energy. A JEPA predicts in representation space instead of reconstructing pixels or
DOM, which is the right abstraction level for "what the interface does" as opposed to "what it looks like" —
rendering is a separate, replaceable decoder. The prediction error of the JEPA is used directly as the energy;
the compiled circuit adds a hard term where the spec is explicit and is constant (silent) where it is not.

Conformance is measured on BF16 weights that ship, never on the FP32 training copy. A number the inference
worker cannot reproduce is not a result.

## Layout

```
specs/*.nerve            behavior specs (toggle, counter, slider, negotiate)
nerve/spec               schema, safe expression language, parser, oracle, self-check, exact circuit compiler
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
                          │  nerve patch  (PULSE: ~100× smaller than a checkpoint at RL learning rates)
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
