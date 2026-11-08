# Paper 2 outline: Machines Without Code

> The draft in `main.tex` + `sections/` supersedes this outline. Differences made while drafting: the nine
> figure titles became one generated table (claims 1, 3) and three figures (the cost plot, the odometer spec,
> the bindings listing); claim 2 is prose in §2.2; claims 4–9 are the five registered predictions of §6 (the
> cost-model claim 5 is §3.2); the chain demo uses finite blocks only, and continuous blocks are the ball demo.

Working title: **Machines Without Code: Wiring Exact Networks into Interactive Systems** (Nerve, part II).
Paper 1 (`docs/paper/`) showed one small interface can be a network held to its spec. This paper is the forward
plan: how small exact networks become large interactive machines, what the ML architecture, the runtime and the
input model have to be, and the experiments that will tell us whether it works. It is written as a planning
paper: every planned result carries a prediction and what would falsify it, in the style of `experiments/`.

Status key: **[have]** measured in round 2; **[partial]** measured but incomplete; **[plan]** proposed, with a
registered prediction.

---

## Figure and table titles (the story in order)

1. **Trained on its whole spec, one network per interface is exact.** [have] Table: 17 interfaces (7 from
   paper 1, 10 from Dragology's demo set), valid states, (state, action) cells, parameters, GPU seconds, exact,
   violations, seeds.
2. **Answering only from states the spec allows removes rule breaks without retraining.** [have] Reorder:
   61.7% → 95.3% at 70% held out; same networks, no retraining; nothing changes on specs with no invalid states.
3. **One network per interface hits an enumeration wall.** [partial] GPU seconds against cells, roughly linear
   (hanoi 729 cells / 380 s … kanban 17,280 / 4,047 s); full-size fifteen (≈10¹³ states) and real sokoban
   levels are out of reach as single networks.
4. **Exact blocks wired by events stay exact: an odometer of digit networks.** [plan] One ~80k-parameter
   digit block, six copies, carry wiring, a million-state machine checked exhaustively at 3 digits.
5. **The cost of a machine is the sum of its blocks, not their product.** [plan] The three couplings
   (independent, event-wired, read-wired) and their training cost; kanban as the read-coupled worst case.
6. **A machine of networks runs itself and checks every hand-off.** [plan] A clocked chain: toggle → counter
   → tracks → …; each hand-off passes the spec check live (the Rube Goldberg demo).
7. **Grab anything while it runs and the chain re-plans.** [plan] A drag mid-run is a search over futures; the
   downstream blocks re-solve from the dropped state.
8. **Desktop input is state plus bindings.** [plan] Selection, focus, modifiers, context menu, undo on a card
   board; replayed keyboard-and-mouse traces checked against the spec.
9. **Continuous hand-offs do not drift when they are discrete events.** [plan] A ball along track blocks:
   error inside a block vs across hand-offs with and without snapping.

---

## 1. Introduction (written last; broad → specific)

- Software as state machines; paper 1's claim and result in one paragraph (network *is* the program; compiled
  where explicit, trained where not; held to the spec).
- What paper 1 could not say: every network was one interface of ≤ 404 states; nothing composed; no input
  beyond a click or a drag; no time.
- The qwen projects compile a *general* machine into a billion-parameter model; we go the other way:
  specific machines from ~10⁵-parameter exact parts. Both are "programs in weights"; the questions differ.
- The goal picture: a machine you can watch run through a series of separate networks, checked at every
  hand-off, that you can grab mid-run; and a desktop interface (keyboard, mouse, selection, undo) built the
  same way.
- "Here, we": round 2's exactness result on 17 interfaces; the block calculus and its exactness theorem; the
  block architecture; the runtime; four demos as registered experiments; the evaluation protocol for machines.
- Contributions list.

## 2. Where we stand: one network per interface is exact [have]

### 2.1 Round 2 in one table (Fig. 1)
- The 10 Dragology demos added: what each tests (hanoi: guards over other fields + a solver by example;
  kanban: 2-D reorder; fifteen; rush hour; sokoban: three objects in one rule; insert/remove: variable length;
  todo; scheduler; three-way switch; linked sliders: continuous, three fields).
- Scaling down to fit enumeration, and what Dragology does that Nerve cannot yet (chained drags, snap
  radius, text, drawing beyond points).
- The goal shift: generalisation to unseen cells is not the goal; each interface is trained on its whole spec.
  The question becomes exactness, reliability across seeds, and cost.
- Result: 100% exact, 0 violations on every finished finite spec, on every seed; hanoi reproduces all 81
  solver examples on 3 seeds; continuous specs 99.6–100% within 1% of range.
- Every served model comes from a committed command (`experiments/round2/`); no hand-edited artifacts.

### 2.2 Valid candidates (Fig. 2)
- The reorder failure under scarce data: answers where two items share a slot.
- Restricting the argmin to states the invariants allow: the spec's own knowledge, applied at the network's
  step; infinite energy outside the set.
- The gain is all in where the network may answer; retraining over the restricted set adds nothing.
- This is paper 1's "illegal states cost infinite energy" (Theorem 1) carried from the circuit to the learned
  term; it is also the first wiring primitive (a block's output port is typed by its invariant).

### 2.3 Drags as searches [have]
- `closest`, `between`, `vary`, `each`, `floating`, `far`; futures derived from the spec, never listed by hand
  (tests: equal to Dragology's hand-written lists for every demo).
- The learned term in drags is 0 on finite specs (network agrees with the spec) and small on continuous ones;
  the network's job in a drag is tie-breaking and holes, not geometry.

### 2.4 The enumeration wall (Fig. 3)
- Training cost ~ cells; the trainer scores every state as a candidate.
- Where it fails: full fifteen, real sokoban, card piles, any list of lists.
- Two exits: restrict candidates to a gesture's futures (engineering), or stop building one network per
  machine (this paper).

## 3. Machines from blocks

### 3.1 Blocks
- A block = a spec + its network + its ports: state fields it owns, actions it accepts (input ports), events it
  emits (output ports), fields it reads from other blocks (read ports).
- Exactness of a block: on its own cells, network step = oracle step, no violation (the round-2 result).
- A block's contract is its invariant and ensures; valid-candidate restriction makes the contract hold by
  construction at the output.

### 3.2 Three couplings and their cost (Fig. 5)
- Independent: energies add; argmin splits; cost = sum. Scheduler is three meeting blocks.
- Event-wired: a step of A emits an action to B (carry). Cost = sum; the machine's state space is the product
  but is never enumerated. Odometer.
- Read-wired: B's rule reads A's fields. B trains on its own fields × the fields it reads: local product,
  grows with coupling. Sokoban's box reads the player and the cell beyond; kanban is fully coupled and does
  not split.
- Rule of thumb for designers: keep read ports few; move coupling into events where possible.

### 3.3 Exactness composes (claim, with proof sketch)
- Theorem (informal): if every block is exact on its cells and the wiring is a correct routing of actions and
  events, then the machine's step equals the composed spec's step, and no invariant of any block is broken.
- Proof sketch by induction on the event cascade; termination needs an acyclic or bounded event graph per tick.
- What it does not say: nothing about liveness or global properties that span blocks (temporal properties,
  ROADMAP 9); those need the composed oracle.

### 3.4 The composed spec
- `import`/`uses` (ROADMAP 6): a machine spec names blocks, instances, wires (`on digit0.wrap -> digit1.inc`),
  and reads.
- The composed oracle is derived, like every oracle: it runs the machine against itself, exhaustively when
  small, by sampling when large.
- Block reuse: one network for every instance of a block type.

## 4. Block architecture (the ML)

### 4.1 The energy block today
- JEPA prediction error + equivariant factored head (paper 1), trained with InfoNCE over the candidate set,
  hinge on spec-wrong negatives; valid-candidate argmin; ~70–145k parameters; GPU minutes.
- Continuous blocks: contrast against near misses; step by descent from seeds; scored within 1% of range.
- Hole blocks: behaviour by example (hanoi's solver): 81/81 learned alongside 729 written cells.

### 4.2 Heads by field type (the equivariance programme)
- Ordinal fields: the shift kernel (paper 1) makes "+1" the same function everywhere.
- List fields: a permutation-equivariant head (attention or DeepSets over items) so "move i to j" is one rule,
  not n² cells; what reorder's residual errors at 85–90% held out say about the need.
- Positions in 2-D: translation-equivariant heads; fifteen and sokoban as tests.
- The type system of the spec chooses the head: a rule's symmetry group is a property of its field types.
- Prediction: with the right head, cells-to-train drops from the product to something near the sum of domain
  sizes; falsifier: no reduction in cells needed for 100% on reorder or fifteen.

### 4.3 Candidate restriction by futures
- Score only a gesture's futures (spec-derived) rather than every state; turns the enumeration wall into a
  per-action cost.
- Keeps training exhaustive over cells that matter; evaluation unchanged.

### 4.4 Compiled vs trained per block
- Paper 1's compiler gives exact weights for explicit transitions; round 2 trained everything. Plan: compile
  the explicit core of each block, train holes and continuous parts, as in paper 1, now per block.
- Energy as a type (ROADMAP 67): two blocks are compatible at a port if their energies agree on the shared
  fields' valid states; a check at wiring time.

### 4.5 Sizes and budgets
- Why blocks must stay small: enumeration, training time, and the runtime frame budget.
- Per-block parameter and latency targets: ≤ 150k params; ≤ 1 ms per finite step, ≤ 40 ms per continuous step
  on CPU; what a chain of ten costs per frame.

## 5. Runtime

### 5.1 The tick
- Loop per frame: inputs → bindings → actions → per-block argmin → event cascade → spec checks → render.
- Determinism: a tick is a pure function of (machine state, inputs); replayable, testable.
- Event cascade bounded per tick; a block's emitted events are themselves spec'd (`emits:`).

### 5.2 Bindings (the input model)
- A `bindings:` section of the spec: key, modifier, mouse button, wheel, double-click → action with params from
  the pointer or the focused/selected item.
- Dragology precedents: `alt-to-copy`, `drag-to-copy`, `with-pointer-delay`.
- Modifier semantics in drags: Shift constrains an axis (a `vary` with a fixed field), Alt copies (a different
  action behind the same gesture).

### 5.3 Selection, focus, menus, undo
- Selection and focus as blocks wired to the board: 2ⁿ selection states kept out of the board's spec.
- Context menu as a mode field; Esc as an action.
- Undo as a runtime wrapper over any machine (a history of states), never learned: cheap and exact by
  construction.
- Marquee selection: a new drag kind whose futures are selection sets from the rectangle; not in Dragology.
- Text stays out until there is a text type (ROADMAP 7 family).

### 5.4 Drags inside a running machine (Fig. 7)
- A drag is a search over futures from the current state; during a drag the block is detached from the clock,
  on release the cascade resumes from the dropped state.
- Live spec checking as the visible signature: each block lights green/red per tick.

### 5.5 Continuous motion and hand-offs (Fig. 9)
- Error accumulates inside continuous blocks (observed: 0 → 0.59 → 1.41 on the continuous slider).
- Design rule: hand-offs are discrete events; a continuous block's output is snapped to the spec at the
  hand-off; a chain cannot wander.

### 5.6 Rendering
- The lab draws points; a machine needs shapes (ROADMAP 12): rectangles, lines, angles, z-order; render is
  derived from state, never learned.
- The lab as the runtime: one page, one tick loop, every block a network served from the same process.

## 6. The demos as experiments (each: setup, prediction, falsifier, acceptance test)

### 6.1 Odometer (Fig. 4)
- One digit block (10 states, `inc`, emits `wrap`); 3 and 6 copies; carry wiring.
- Prediction: exact on all 1,000 states of 3 digits by exhaustive check; 10⁶ states at 6 digits by sampling
  with zero violations; falsifier: any carry error.

### 6.2 The self-running chain (Fig. 6)
- toggle → counter → tracks → reorder (or similar): a clock drives the first block; each hand-off is an event.
- Prediction: runs for 10⁴ ticks with zero violations; the chain's step time < 16 ms; falsifier: a violation or
  a dropped frame budget.

### 6.3 Grab while running (Fig. 7)
- Drag any element mid-run; measure that downstream blocks reach the state the composed oracle predicts from
  the dropped state within k ticks.

### 6.4 The desktop board (Fig. 8)
- Cards on a grid; focus (Tab, arrows), selection (click, Ctrl, Shift, marquee), move (arrows, drag), Alt-drag
  copies, Delete, Ctrl+Z, right-click menu.
- Replayed traces: recorded input sequences; prediction: 100% agreement with the composed spec over 1,000
  random traces of length 50; falsifier: any divergence.

### 6.5 The rolling ball (Fig. 9)
- Track blocks chained; continuous motion inside, events between; drift with vs without snapping.

## 7. Evaluation protocol for machines
- Exhaustive conformance when the machine's reachable states ≤ 10⁴; sampled trajectories otherwise.
- k-step rollout conformance (ROADMAP 34), not single steps.
- Replayed input traces as the unit of test for desktop interfaces.
- Hand-off drift for continuous chains.
- Reporting rules from round 1's review: seeds, both scoring modes (network alone vs with the spec's mask),
  committed scripts, append-only results, no hand edits.

## 8. Related work
- Programs in weights: Tracr, Cajal, ink `emit-circuit`, Qwenputer and qwenGPU (general machines in LMs; sizes
  and costs compared); ROADMAP 20 as the meeting point.
- Synchronous and reactive languages: Esterel, Lustre, statecharts (Harel), the Elm architecture; what the
  tick and the event cascade borrow.
- Declarative dragging (Dragology): futures, operators, and the demos we rebuilt; what we add (derived
  futures, spec checking) and what we lack (chaining, snap radius).
- Neural interfaces and world models: NeuralOS, GameNGen, Genie-style models; pixels vs state; our networks
  are the behaviour, not the picture.
- Energy-based inference and JEPA; equivariant networks (DeepSets, set transformers, CNNs) for the heads.
- Modular and compositional networks: mixture of experts, neural module networks; why exact blocks differ.

## 9. Limitations, and what we do not know
- Everything in §3–6 is a plan with predictions; none of it is built.
- Exactness composes only with correct wiring; wiring bugs are code bugs again (the composed oracle catches
  them, but it is code).
- Read-coupled machines do not split; the head programme (§4.2) is a bet.
- Continuous blocks are 99.6–100%, not exact; the snapping rule contains, not removes, the error.
- No text, no rendering beyond points, no chained drags; Dragology's largest demos remain out of reach.
- Single machine, single user, single clock; nothing about networks of machines or concurrency.

## 10. Conclusion
- One network per interface is exact at ~10⁵ parameters; the way to large systems is not larger networks but
  machines of small exact ones, with a clock, events, bindings and live checking; the demos that will show it.
