# Roadmap

What to build next, grouped by subsystem. Each item has a one-line acceptance test so "done" is not a feeling.
Ordering inside a group is rough priority. The first section is the short list.

## Now (the next five) — status 2026-11-06: 1 ✅  2 ✅  3 ✅  4 ✅  5 🔄 (12-iteration run in progress)

1. ✅ **Apply PULSE patches inside the Worker.** `nn.ts` decodes `.nrvp` (varint indices, BF16 values, zstd via a WASM
   decoder), patches its in-memory weights, verifies the digest. *Done when:* a `nerve patch` dropped in R2/export
   changes the Worker's answers without a redeploy, and the digest check rejects a flipped byte.
2. ✅ **Screenshot intake, end to end.** (toggle 84 states, stepper 8, volume slider 404: all 1.0 learned-only) Run `nerve intake shot.png` with a vision model, loop on `nerve check` until
   zero problems, train, serve. *Done when:* a screenshot of a real toggle/stepper/slider yields a spec that trains to
   ≥0.95 learned-only conformance with no hand edits.
3. ✅ **Continuous fields.** (0.998 within tolerance at 1 000 random targets, gridded slider was 0.986) `vary` futures over `float` fields without a step grid: Langevin / gradient descent on the
   energy from a few seeds, with the compiled circuit as a hard prior on the discrete part. *Done when:* the slider
   spec with `step` removed reaches the same conformance as the gridded one, measured at 1 000 random targets.
4. ✅ **Learned-only conformance to 1.0 on all four specs.** (and on the three screenshot specs; factored energy did it) Today 0.99 on counter, 0.87–0.93 on negotiate. Curriculum
   over episodes, larger `d`, longer training, better negatives. *Done when:* `nerve train --no-circuit` scores ≥0.99
   held-out on every spec in `specs/`.
5. 🔄 **`--mode claude` evolution run on a real budget.** (2-iteration smoke run: $2.06, supervisor dropped a role and invented `hole_tutor`; 12-iteration run on negotiate in progress) One overnight run of `nerve evolve` with Claude Code as
   supervisor and roles on herdr panes, $20 budget, logged to `evolution/`. *Done when:* history shows ≥10 iterations,
   roles created by the supervisor that did not exist at iteration 0, and a best score above the offline reflector's.

## Spec language

6. **Composition.** `import` a spec into another and reference its state as a sub-record; actions namespaced. A
   todo-list is `list<todo_item>`. *Done when:* `specs/todo.nerve` composes `specs/toggle.nerve` for its checkbox.
7. **Collections and bounded lists.** `list<T, max=N>` with `append`, `remove(i)`, `move(i, j)`; encoded as padded
   one-hots with a length field. *Done when:* a 5-item reorderable list compiles and trains (dragology's reorder demo).
8. **Parametric `ensures` quantifiers.** `forall i in items: items[i].count >= 0`. *Done when:* the checker expands
   them over finite ranges and the expression grammar rejects unbounded ones.
9. **Temporal properties.** `eventually` / `always` / `until` over action sequences (bounded LTL), checked on rollouts
   and used as negatives. *Done when:* `negotiate.nerve` can say "a deal is reached within 3 rounds or status is
   no_deal" and the checker enforces it.
10. **Refinement.** `refines: other.nerve` asserts every behaviour of this spec is permitted by the other. *Done when:*
    a stricter counter refines the loose one and the checker proves it by exhaustive simulation.
11. **Nondeterminism.** `oneof [...]` transitions where several successors are legal (dragology `closest` is this).
    Energy must be low on all legal successors. *Done when:* conformance scores "legal", not "equal to one oracle".
12. **Render block.** `render:` maps state to a declarative view (SVG/DOM tree with `transform=translate`, dragology
    style). Not learned; derived. *Done when:* `nerve render` produces an SVG per state and the edge Worker serves it.
13. **Spec diff and migration.** `nerve diff a.nerve b.nerve` lists added/removed states and actions and generates a
    state migration; retrain only what changed. *Done when:* adding a field to counter retrains in < half the time.
14. **ink interop.** Emit ink source from a spec (`nerve emit-ink`) so ink's proof-checked rewrite database and
    `emit-circuit` can be the compiler for the deterministic core. *Done when:* ink's circuit and ours agree on counter.
15. **Formal export.** Emit Lean 4 definitions of state, transitions and invariants with `decide`-able proofs for finite
    specs. *Done when:* `lake build` proves counter's invariants from the generated file.

## Compiler

16. **Factored circuits.** Today one hidden unit per (state, action) pair. Compile per-field update functions
    separately and compose; most transitions touch one field. *Done when:* negotiate's circuit shrinks ≥10× with
    identical energies.
17. **Circuit distillation into the JEPA.** Use the compiled circuit as a teacher so the learned model carries the
    whole interface and the circuit can be dropped at serve time. *Done when:* learned-only = 1.0 on all specs after
    distillation and the exported model has no circuit.
18. **Partial compilation for infinite specs.** Compile the finite projection (enums, bools, bounded ints) and leave
    unbounded fields to the learned model. *Done when:* a spec with an unbounded `int` still gets a circuit for its
    enum transitions.
19. **Circuit verification.** Prove `energy(s, a, T(s,a)) == 0` and `> 0` elsewhere symbolically from the construction
    rather than by enumeration. *Done when:* a 20-line argument in `docs/circuit.md` and a property test over random specs.
20. **Weight-space merge.** Insert the compiled circuit as extra layers into a pretrained small LM (the qwenGPU move)
    so one network holds both the interface and language about it. *Done when:* Qwen3.5-0.5B + counter circuit answers
    "what happens if I press inc at 9?" correctly by executing, not recalling.

## Model and training

21. **Shared backbone across specs.** One JEPA with spec-conditioned adapters (LoRA per spec, shared encoder).
    *Done when:* training on 3 specs then fine-tuning a 4th converges in a third of the steps.
22. **Hierarchical JEPA.** Predict at two timescales: next state and state after k actions. *Done when:* k-step
    rollouts from the learned model match the oracle ≥0.95 at k=5 on negotiate.
23. **Better negatives.** Mine hard negatives from the model's own near-misses each epoch, not just random violating
    states. *Done when:* held-out conformance improves at equal epochs on all specs.
24. **Energy calibration.** Margin and temperature targets so `energy` is a usable confidence; report expected
    calibration error on violating futures. *Done when:* runtime refuses (returns alternatives) when margin < τ and
    that refusal has precision ≥0.9 on injected spec bugs.
25. **RL phase with verifiable rewards.** After supervised JEPA training, GRPO-style self-play on the spec's checker
    (negotiate is the obvious first task; the repo's lineage). *Done when:* negotiate agents trained against the
    checker reach a deal rate and fairness that the supervised model does not.
26. **Continuous training from live traffic.** Runtime logs real (s, a) pairs; the checker labels them; retrain;
    patch. *Done when:* a running Worker's weights change in response to a day of synthetic traffic without a human.
27. **Rendering decoder (optional).** Learn state → pixels on top of the frozen JEPA for specs without a `render`
    block; keep behaviour and appearance separated. *Done when:* a decoder trained on screenshots reproduces the
    counter UI for unseen states with SSIM ≥0.9.
28. **Mixed precision discipline.** Train FP32 master, evaluate BF16 always (done); add INT8 export and check
    conformance survives. *Done when:* INT8 edge export keeps learned-only within 0.01 of BF16.

## Data and harness

29. **Real-app rollouts.** When intake came from code, run the actual app under Playwright and log (s, a, s') from the
    DOM, so the oracle is checked against the real thing, not only itself. *Done when:* a disagreement between oracle
    and app is surfaced as a spec problem.
30. **Agent-driven exploration.** Replace the three fixed policies with a `coverage_hunter` that is a Claude Code
    process reading eval.json and choosing where to explore. *Done when:* it beats the fixed policies on time-to-full-coverage.
31. **Log compaction and versioning.** `episodes.v2.jsonl` with zstd, dedupe, provenance (which role, which iteration),
    and one-step migrations like dsh's `session.vN.jsonl`. *Done when:* a 10 M-step log trains without loading all of it.
32. **dsh as a first-class role runtime.** Run roles under DeepSeek Harness with the `nerve` profile, read its session
    log back into learnings. *Done when:* an evolve iteration runs with `--roles dsh` and both logs reconcile.
33. **Dataset cards.** Every training run writes a card: spec hash, log hash, config, scores, who/what produced the
    data. *Done when:* `nerve lineage runs/x/v2` prints the full chain back to the spec.

## Evolution (EAT)

34. **Richer official eval.** Add k-step rollout conformance, calibration, inference latency on workerd, and model
    size to the score; make weights configurable per spec. *Done when:* the supervisor can trade size for conformance.
35. **Supervisor memory.** Structured `learnings.md` → a small store of (symptom, intervention, effect) tuples the
    offline reflector and the Claude supervisor both read. *Done when:* a rule learned on counter transfers to slider.
36. **Role library growth.** Let the supervisor write new role prompts to `harness/eat/roles/` and promote the ones
    that improved score twice. *Done when:* a run ends with at least one promoted role not written by a human.
37. **Clean-room mode.** EAT's `sandbox.sh` equivalent: roles get only the spec, the log, and the pipeline verbs; no
    web, no other repos; every request logged through an allowlist proxy. *Done when:* a run completes with the audit
    showing zero disallowed calls.
38. **Multi-spec tournaments.** Evolve one team across all specs; score is the minimum. *Done when:* the team that wins
    on the tournament generalises to a held-out spec better than per-spec teams.
39. **herdr socket client.** Replace the subprocess fallback with herdr's real socket API: spawn panes, block on
    "blocked", read pane status. *Done when:* `nerve evolve` shows every role as a herdr pane with live state.

## Weight sync (PULSE)

40. **Patch chains and anchors.** Named base checkpoints, patch sequences, periodic re-anchoring; workers can join
    mid-stream. *Done when:* a worker started at patch 50 reconstructs the trainer's weights from anchor + 50 patches,
    bit-identical.
41. **Relay.** A Worker + R2 + Queues relay fan-out: trainer uploads once, N workers pull. *Done when:* 10 workerd
    instances on one machine converge within one patch interval.
42. **Error-feedback for trainer↔trainer.** PULSELoCo: gate pseudo-gradients, keep the remainder in an FP32 buffer,
    Nesterov outer step. For when training is split across the user's GPU boxes (herdr machines). *Done when:* two
    trainers on one spec match single-trainer conformance with ≥90% communication saved.
43. **Patch-level tests.** Fuzz the varint/zstd codec; cross-check the TS decoder against Python on random tensors.
    *Done when:* 10 000 random patches round-trip in both languages.

## Serving and edge

44. **WebGPU inner loop.** `nn.ts` matvecs on WebGPU when available (browser or workerd with GPU), CPU fallback.
    *Done when:* negotiate's 216-candidate energy evaluation drops below 1 ms in Chrome.
45. **Client-side execution.** A browser bundle that fetches `/weights`, applies patches, and runs the interface with
    zero round trips; the Durable Object becomes sync, not compute. *Done when:* the counter works offline in a tab.
46. **Cloudflare Containers origin.** For large specs, `nerve serve` in a Container behind the Worker (`ORIGIN`),
    patches pulled from R2. *Done when:* a 50 k-state spec serves at p50 < 50 ms from the Container.
47. **Session semantics.** Durable Object stores the action log, not just state; replay on weight change; export a
    session as a `.nerve` trajectory. *Done when:* after a patch, every live session's state is recomputed from its log.
48. **Multi-agent sessions.** negotiate needs two users per session: DO holds both, enforces turn, broadcasts via
    WebSocket. *Done when:* two browsers negotiate through one DO with the network as referee.
49. **Observability.** Per-step energy, margin, and violation counters to Analytics Engine; a page per spec.
    *Done when:* a regression in a patch is visible as a margin drop before any user reports it.
50. **Hard refusal at the edge.** If the best energy's margin is below τ or the checker flags a violation, the Worker
    returns the state unchanged with alternatives. *Done when:* injected bad weights never move a session into a
    violating state.

## Intake

51. **Code → spec for real repos.** Point intake at a React/SwiftUI component; extract state (useState, @State),
    handlers, and effects into actions; verify against the running component (item 29). *Done when:* three real
    open-source components round-trip to ≥0.95 conformance.
52. **Prose → spec with clarifying questions.** The agent asks when a bound or guard is missing instead of guessing,
    and records the answers in the spec as comments. *Done when:* an underspecified paragraph yields ≤3 questions and a
    spec that passes check.
53. **Intake eval set.** 30 interfaces × 3 modalities with gold specs; measure spec equivalence (same transition table
    up to renaming). *Done when:* CI reports intake accuracy per modality.
54. **Spec from trajectories.** Watch a user operate an interface (screen recording) and induce the spec: states by
    clustering, actions by UI events, transitions by observation. *Done when:* a 5-minute recording of the counter
    recovers its spec with all guards.

## Verification and safety

55. **Property-based testing of the oracle vs. the network.** Hypothesis-style generation of action sequences; shrink
    failures to minimal traces. *Done when:* `nerve fuzz` finds the planted bug in a seeded-bad model in < 1 s.
56. **Adversarial action sequences.** Search for sequences that drive the learned model into violating states even
    when single steps conform. *Done when:* k-step violation rate is reported and is zero for shipped models.
57. **Spec coverage report.** Which invariants were ever at risk in the log, which guards flipped, which ensures were
    exercised. *Done when:* `nerve coverage` renders a table and the supervisor uses it.
58. **Reproducibility.** A run is a pure function of (spec, log, config, seed) — assert it: train twice, compare
    digests. *Done when:* CI fails if two identical runs differ in BF16.
59. **Threat model doc.** What can a bad spec, a bad patch, or a bad role agent do; what stops each. *Done when:*
    `docs/threats.md` exists and each threat points at a test.

## Infrastructure and developer experience

60. **CI.** GitHub Actions: pytest, workerd parity, TS typecheck, demo script smoke. *Done when:* green on the branch.
61. **GPU training path.** `nerve train --device cuda`, mixed precision, larger specs; a herdr "machine" recipe for
    the user's GPU box. *Done when:* a 100 k-state spec trains in < 10 min on one consumer GPU.
62. **`nerve` as a dsh/Claude Code plugin.** Package the harness as a plugin so `/nerve` works inside either agent.
    *Done when:* `/plugin install nerve` gives the EAT skill and the pipeline verbs.
63. **Docs site.** Spec reference, expression grammar, CLI, architecture, the five-minute tutorial. *Done when:* a new
    reader writes a working spec without reading source.
64. **Benchmarks.** Time and memory per verb per spec, tracked over commits. *Done when:* a perf regression > 20% fails CI.
65. **Packaging.** `pip install nerve` and `npm i nerve-edge`; versioned patch format. *Done when:* a fresh machine runs
    the demo from the published packages.

## Research bets (worth a week each, might not pan out)

66. **Spec-free mode.** Learn a JEPA from trajectories alone, then *induce* a spec from the learned energy landscape
    (states with low mutual energy cluster; actions as consistent translations). The inverse of this whole pipeline.
67. **Energy as a type.** Treat the energy function as the type of the interface; two networks are "compatible" if
    their energies agree on a shared sub-spec. Compose products from compatible parts.
68. **Latent actions.** Let the model discover action embeddings beyond those named in the spec, then ask the harness
    agent to name them: the network proposes features.
69. **One network, many interfaces.** Train a single JEPA on all specs with spec text as a conditioning input; measure
    zero-shot conformance on a held-out spec given only its text. If this works, intake becomes inference.
70. **Negotiation as the test of everything.** Bring the 2017 dataset back: two Nerve agents negotiate in natural
    language over a `negotiate.nerve` state the network also enforces. The original repo's task, with the rules
    held by weights instead of code.
