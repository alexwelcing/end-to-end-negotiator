//! Machines: blocks wired by events (paper 2 §3 and §5; docs/machine-room.md).
//!
//! A `machine:` document names block instances (each a spec with its own network), the external
//! actions the outside world may apply (`inputs`), wires from one block's events to another's
//! actions, `derived` values over all blocks, and machine-level `ensures`. A tick applies a list of
//! external actions; after each, the event queue drains in order, every delivery stepping its
//! target block in the machine state current at that moment. Events are evaluated from the specs'
//! `emits` clauses, never produced by a network. A block a pointer holds is *detached*: deliveries
//! addressed to it are held and returned, to be delivered first on a later tick.

use crate::expr::{Expr, Scope};
use crate::oracle::Oracle;
use crate::spec::{ActionInst, Spec, State};
use crate::value::Value;
use anyhow::{bail, ensure, Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_yaml::{Mapping, Value as Yaml};
use std::collections::{HashSet, VecDeque};
use std::path::Path;

/// Reachable states up to which `check` is exhaustive; beyond it, random walks.
pub const EXHAUSTIVE_LIMIT: usize = 10_000;
const DEFAULT_MAX_EVENTS: usize = 64;

/// One wire: an event of a source block delivered as an action to a target block. The action's
/// parameters are expressions over the whole machine (`inst.field`, `old.inst.field`, derived
/// names), evaluated when the event is raised, in the action's declared parameter order.
#[derive(Debug, Clone)]
pub struct Wire {
    pub from_block: usize,
    pub event: String,
    pub to_block: usize,
    pub action: String,
    pub params: Vec<(String, Expr)>,
}

#[derive(Debug, Clone)]
pub struct MachineEnsures {
    /// `(block, action)` this property is checked after, or every tick.
    pub after: Option<(usize, String)>,
    pub holds: Expr,
}

#[derive(Debug, Clone)]
pub struct MachineSpec {
    pub name: String,
    /// `(instance name, type name, spec)`; instances of one type share one network.
    pub blocks: Vec<(String, String, Spec)>,
    /// External action instances the outside world may apply.
    pub inputs: Vec<(usize, ActionInst)>,
    pub wires: Vec<Wire>,
    /// Named values over all blocks (`value: "d0.d + 10 * d1.d"`), usable in `ensures`.
    pub derived: Vec<(String, Expr)>,
    pub ensures: Vec<MachineEnsures>,
    /// Deliveries one tick may process before it is declared non-terminating.
    pub max_events: usize,
    /// Where each block sits in the machine's one picture (`place:`), per block; `None` for a block the
    /// document does not place (the page then gives it a panel).
    pub place: Vec<Option<Place>>,
    /// The machine's own backdrop (`scene:`), in machine coordinates.
    pub scene: Option<crate::render::Scene>,
}

/// A block's placement in the machine's picture: a block point `(bx, by)` sits at
/// `(x + scale * bx, y + scale * by)` in machine coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Place {
    pub x: f64,
    pub y: f64,
    pub scale: f64,
}

/// The machine's state: one state per block, in block order.
pub type MachineState = Vec<State>;

/// A pending block step: an event's delivery, or an external action.
#[derive(Debug, Clone, PartialEq)]
pub struct Delivery {
    pub block: usize,
    pub action: ActionInst,
    /// The wire that raised it, as `(source block, event)`; `None` for an external action.
    pub via: Option<(usize, String)>,
}

/// What one tick is asked to do.
#[derive(Debug, Clone, Default)]
pub struct TickInput<'i> {
    /// External actions, applied in order, each followed by its event cascade.
    pub external: &'i [(usize, ActionInst)],
    /// Deliveries held over from earlier ticks (their blocks were detached), delivered first.
    pub pending: Vec<Delivery>,
    /// Blocks a pointer holds this tick: deliveries to them are held, not applied.
    pub detached: &'i [usize],
}

#[derive(Debug, Clone, Default)]
pub struct TickOutput {
    pub state: MachineState,
    pub trace: TickTrace,
    /// Deliveries held for detached blocks, in order; pass them as the next tick's `pending`.
    pub pending: Vec<Delivery>,
}

/// What `MachineSpec::check` found and how far it looked.
#[derive(Debug, Clone, Default)]
pub struct CheckReport {
    pub problems: Vec<String>,
    /// Every reachable state was visited; otherwise random walks were run instead.
    pub exhaustive: bool,
    pub states_visited: usize,
    pub ticks: usize,
}

/// Whether a `.nerve` document describes a machine (has a `machine:` key) rather than one block.
pub fn is_machine_document(text: &str) -> bool {
    serde_yaml::from_str::<Yaml>(text)
        .ok()
        .and_then(|y| y.as_mapping().map(|m| get(m, "machine").is_some()))
        .unwrap_or(false)
}

impl MachineSpec {
    /// Loads `path`; block types resolve to `<type>.nerve` next to the document, then in a sibling
    /// `specs/` directory, or to an explicit `.nerve` path relative to the document.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        Self::parse(&text, &|name: &str| {
            let candidates = if name.ends_with(".nerve") {
                vec![dir.join(name)]
            } else {
                vec![dir.join(format!("{name}.nerve")), dir.join("..").join("specs").join(format!("{name}.nerve"))]
            };
            for c in &candidates {
                if c.exists() {
                    return Spec::load(c);
                }
            }
            bail!("block type {name:?}: no spec at {}", candidates.iter().map(|c| c.display().to_string()).collect::<Vec<_>>().join(" or "))
        })
        .with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse(text: &str, resolve: &dyn Fn(&str) -> Result<Spec>) -> Result<Self> {
        let doc: Yaml = serde_yaml::from_str(text)?;
        let Yaml::Mapping(doc) = doc else { bail!("machine must be a mapping") };
        ensure!(get(&doc, "nerve").and_then(Yaml::as_i64) == Some(1), "machine must declare `nerve: 1`");
        let name = get(&doc, "machine").map(as_str).transpose()?.context("machine needs `machine: <name>`")?;

        let mut blocks = Vec::new();
        for (inst, ty) in entries(get(&doc, "blocks"))? {
            ensure!(!inst.contains('.') && !inst.contains("__"), "block name {inst:?} may not contain '.' or '__'");
            let ty = as_str(&ty)?;
            let spec = resolve(&ty).with_context(|| format!("block {inst}: {ty}"))?;
            blocks.push((inst, ty, spec));
        }
        ensure!(!blocks.is_empty(), "machine {name} has no blocks");
        let names: Vec<&str> = blocks.iter().map(|(n, _, _)| n.as_str()).collect();
        let block_idx = |inst: &str| names.iter().position(|n| *n == inst);
        let split = |s: &str| -> Result<(usize, String)> {
            let (inst, rest) = s.split_once('.').with_context(|| format!("{s:?} must be `block.name`"))?;
            Ok((block_idx(inst).with_context(|| format!("{s:?}: unknown block {inst:?}"))?, rest.to_string()))
        };

        let mut inputs = Vec::new();
        match get(&doc, "inputs") {
            None | Some(Yaml::Null) => {
                for (b, (_, _, spec)) in blocks.iter().enumerate() {
                    inputs.extend(spec.action_instances()?.into_iter().map(|a| (b, a)));
                }
            }
            Some(Yaml::Sequence(items)) => {
                for it in items {
                    let (b, action) = split(&as_str(it)?)?;
                    let spec = &blocks[b].2;
                    let act = spec.action(&action).with_context(|| format!("input {}.{action}: no such action", names[b]))?;
                    inputs.extend(act.instances()?.into_iter().map(|a| (b, a)));
                }
            }
            Some(other) => bail!("`inputs` must be a list, got {other:?}"),
        }

        let mut wires = Vec::new();
        for w in get(&doc, "wires").and_then(Yaml::as_sequence).cloned().unwrap_or_default() {
            let Yaml::Mapping(wm) = &w else { bail!("each wire must be a mapping with `on` and `do`") };
            let (from_block, event) = split(&as_str(get(wm, "on").context("wire needs `on: block.event`")?)?)?;
            let (to_block, action) = split(&as_str(get(wm, "do").context("wire needs `do: block.action`")?)?)?;
            let src = &blocks[from_block].2;
            ensure!(src.events().contains(&event.as_str()), "wire on {}.{event}: block {} emits no such event", names[from_block], names[from_block]);
            let act = blocks[to_block].2.action(&action).with_context(|| format!("wire do {}.{action}: no such action", names[to_block]))?;
            let given = entries(get(wm, "with"))?;
            for (k, _) in &given {
                ensure!(act.param(k).is_some(), "wire do {}.{action}: no parameter {k:?}", names[to_block]);
            }
            // In the action's declared order, so a delivery is an ordinary action instance.
            let mut params = Vec::new();
            for p in &act.params {
                let (_, v) = given.iter().find(|(k, _)| *k == p.name).with_context(|| {
                    format!("wire do {}.{action}: parameter {} needs a `with:` expression", names[to_block], p.name)
                })?;
                params.push((p.name.clone(), Expr::compile(&flatten(&expr_src(v)?, &names))?));
            }
            wires.push(Wire { from_block, event, to_block, action, params });
        }

        let mut derived = Vec::new();
        for (k, v) in entries(get(&doc, "derived"))? {
            ensure!(block_idx(&k).is_none(), "derived {k:?} has the name of a block");
            derived.push((k, Expr::compile(&flatten(&expr_src(&v)?, &names))?));
        }
        let mut ensures = Vec::new();
        for e in get(&doc, "ensures").and_then(Yaml::as_sequence).cloned().unwrap_or_default() {
            let Yaml::Mapping(em) = &e else { bail!("ensures entries must be mappings") };
            let after = match get(em, "after") {
                None | Some(Yaml::Null) => None,
                Some(a) => {
                    let (b, action) = split(&as_str(a)?)?;
                    ensure!(blocks[b].2.action(&action).is_some(), "ensures after {}.{action}: no such action", names[b]);
                    Some((b, action))
                }
            };
            let holds = Expr::compile(&flatten(&expr_src(get(em, "holds").context("ensures needs `holds`")?)?, &names))?;
            ensures.push(MachineEnsures { after, holds });
        }
        let max_events = get(&doc, "max_events").and_then(Yaml::as_u64).map_or(DEFAULT_MAX_EVENTS, |n| n as usize);

        // One scene: `place:` puts blocks (which must have a scene of their own) in machine coordinates,
        // and the machine's own `scene:` is the backdrop behind them.
        let mut place = vec![None; blocks.len()];
        for (inst, pv) in entries(get(&doc, "place"))? {
            let b = block_idx(&inst).with_context(|| format!("place {inst:?}: unknown block"))?;
            let Yaml::Mapping(pm) = &pv else { bail!("place {inst}: needs a mapping with x, y and scale") };
            let num = |k: &str, dflt: f64| -> Result<f64> {
                match get(pm, k) {
                    None | Some(Yaml::Null) => Ok(dflt),
                    Some(v) => v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)).with_context(|| format!("place {inst}.{k}: must be a number")),
                }
            };
            let p = Place { x: num("x", 0.0)?, y: num("y", 0.0)?, scale: num("scale", 1.0)? };
            ensure!(p.scale > 0.0, "place {inst}: scale must be positive");
            ensure!(blocks[b].2.scene.is_some(), "place {inst}: block type {} has no scene to place", blocks[b].1);
            place[b] = Some(p);
        }
        let scene = get(&doc, "scene").map(crate::render::Scene::parse).transpose().context("scene")?;
        if place.iter().any(Option::is_some) {
            let missing: Vec<&str> = blocks.iter().zip(&place).filter(|(_, p)| p.is_none()).map(|((n, _, _), _)| n.as_str()).collect();
            ensure!(missing.is_empty(), "place: every block or none; not placed: {}", missing.join(", "));
        }
        ensure!(scene.is_none() || place.iter().any(Option::is_some), "a machine scene needs placed blocks");
        Ok(Self { name, blocks, inputs, wires, derived, ensures, max_events, place, scene })
    }

    /// Whether the document composes one picture (any block placed).
    pub fn has_picture(&self) -> bool {
        self.place.iter().any(Option::is_some)
    }

    /// A block's point in machine coordinates, when the block is placed.
    pub fn to_machine(&self, block: usize, (x, y): (f64, f64)) -> Option<(f64, f64)> {
        self.place[block].map(|p| (p.x + p.scale * x, p.y + p.scale * y))
    }

    /// A block's port in machine coordinates: the port of that name, else the centre of its scene box.
    pub fn port(&self, block: usize, name: &str) -> Option<(f64, f64)> {
        let spec = &self.blocks[block].2;
        let sc = spec.scene.as_ref()?;
        let p = spec.ports.iter().find(|(n, _)| n == name).map(|(_, p)| *p).unwrap_or(((sc.x0 + sc.x1) / 2.0, (sc.y0 + sc.y1) / 2.0));
        self.to_machine(block, p)
    }

    /// Where a wire runs in the machine's picture: from the source's port named after the event to the
    /// target's port named after the action, when both blocks are placed.
    pub fn wire_path(&self, w: &Wire) -> Option<((f64, f64), (f64, f64))> {
        Some((self.port(w.from_block, &w.event)?, self.port(w.to_block, &w.action)?))
    }

    /// The box of the machine's picture, `(x0, y0, x1, y1)`: its scene's, else the union of the placed
    /// blocks' boxes.
    pub fn picture_box(&self) -> Option<(f64, f64, f64, f64)> {
        if let Some(sc) = &self.scene {
            return Some((sc.x0, sc.y0, sc.x1, sc.y1));
        }
        let mut out: Option<(f64, f64, f64, f64)> = None;
        for (b, (_, _, spec)) in self.blocks.iter().enumerate() {
            let (Some(sc), Some(p)) = (spec.scene.as_ref(), self.place[b]) else { continue };
            let (x0, y0) = (p.x + p.scale * sc.x0, p.y + p.scale * sc.y0);
            let (x1, y1) = (p.x + p.scale * sc.x1, p.y + p.scale * sc.y1);
            out = Some(out.map_or((x0, y0, x1, y1), |(a, b, c, d)| (a.min(x0), b.min(y0), c.max(x1), d.max(y1))));
        }
        out
    }

    pub fn initial(&self) -> MachineState {
        self.blocks.iter().map(|(_, _, s)| s.initial.clone()).collect()
    }

    pub fn block_index(&self, inst: &str) -> Option<usize> {
        self.blocks.iter().position(|(n, _, _)| n == inst)
    }

    /// Number of machine states the block state spaces allow (the product), for sizing a check.
    pub fn state_space_size(&self) -> Result<usize> {
        self.blocks.iter().map(|(_, _, s)| s.state_space_size()).product()
    }

    /// A readable key for a machine state, for traces and sets.
    pub fn state_key(&self, s: &MachineState) -> String {
        self.blocks.iter().zip(s).map(|((n, _, sp), st)| format!("{n}={}", sp.state_key(st))).collect::<Vec<_>>().join(" ")
    }

    /// Runs the machine, every block stepped by its oracle, against its own `ensures` and the blocks'
    /// invariants: exhaustively over the reachable states when there are at most `EXHAUSTIVE_LIMIT`,
    /// otherwise over `walks` random walks of length `horizon`.
    pub fn check(&self, walks: usize, horizon: usize) -> Result<CheckReport> {
        let m = Machine::oracle(self);
        let mut problems = Vec::new();
        let mut ticks = 0usize;
        let mut note = |b: usize, a: &ActionInst, s: &MachineState, trace: &TickTrace| {
            for v in &trace.violations {
                if problems.len() < 50 {
                    problems.push(format!("{}.{} from {}: {v}", self.blocks[b].0, a.name, self.state_key(s)));
                }
            }
        };
        let mut seen: HashSet<String> = HashSet::new();
        let mut queue = VecDeque::from([self.initial()]);
        seen.insert(self.state_key(&self.initial()));
        let mut exhaustive = true;
        'bfs: while let Some(s) = queue.pop_front() {
            for (b, a) in &self.inputs {
                let (next, trace) = m.tick(&s, std::slice::from_ref(&(*b, a.clone())))?;
                ticks += 1;
                note(*b, a, &s, &trace);
                if seen.insert(self.state_key(&next)) {
                    if seen.len() > EXHAUSTIVE_LIMIT {
                        exhaustive = false;
                        break 'bfs;
                    }
                    queue.push_back(next);
                }
            }
        }
        if !exhaustive {
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            for _ in 0..walks {
                let mut s = self.initial();
                for _ in 0..horizon {
                    let (b, a) = self.inputs.choose(&mut rng).context("machine has no inputs")?;
                    let (next, trace) = m.tick(&s, std::slice::from_ref(&(*b, a.clone())))?;
                    ticks += 1;
                    note(*b, a, &s, &trace);
                    s = next;
                }
            }
        }
        Ok(CheckReport { problems, exhaustive, states_visited: seen.len(), ticks })
    }
}

/// Steps one block: its oracle, or its trained network.
pub trait BlockStep {
    fn step(&self, s: &State, a: &ActionInst) -> Result<State>;
}

impl BlockStep for Oracle<'_> {
    fn step(&self, s: &State, a: &ActionInst) -> Result<State> {
        Oracle::step(self, s, a)
    }
}

impl BlockStep for crate::train::Model {
    fn step(&self, s: &State, a: &ActionInst) -> Result<State> {
        crate::train::Model::step(self, s, a)
    }
}

impl BlockStep for crate::continuous::ContinuousModel {
    fn step(&self, s: &State, a: &ActionInst) -> Result<State> {
        crate::continuous::ContinuousModel::step(self, s, a)
    }
}

impl<T: BlockStep + ?Sized> BlockStep for &T {
    fn step(&self, s: &State, a: &ActionInst) -> Result<State> {
        (**self).step(s, a)
    }
}

impl<T: BlockStep + ?Sized> BlockStep for std::sync::Arc<T> {
    fn step(&self, s: &State, a: &ActionInst) -> Result<State> {
        (**self).step(s, a)
    }
}

/// One block step inside a tick, for traces and the lab.
#[derive(Debug, Clone)]
pub struct StepRecord {
    pub block: usize,
    pub action: ActionInst,
    /// The wire that delivered this action, or `None` for an external action.
    pub via: Option<(usize, String)>,
    pub before: State,
    pub after: State,
    pub events: Vec<String>,
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TickTrace {
    pub steps: Vec<StepRecord>,
    /// Block violations and machine `ensures` failures, in order.
    pub violations: Vec<String>,
}

/// A machine ready to run: a stepper per block, plus each block's oracle for events and checks.
pub struct Machine<'a> {
    pub spec: &'a MachineSpec,
    steppers: Vec<Box<dyn BlockStep + 'a>>,
    oracles: Vec<Oracle<'a>>,
}

impl<'a> Machine<'a> {
    /// Every block stepped by its oracle: the composed spec itself.
    pub fn oracle(spec: &'a MachineSpec) -> Self {
        let steppers = spec.blocks.iter().map(|(_, _, s)| Box::new(Oracle::new(s)) as Box<dyn BlockStep + 'a>).collect();
        Self::with_steppers(spec, steppers)
    }

    /// Blocks stepped by the given steppers (networks for the composed network), one per block.
    pub fn with_steppers(spec: &'a MachineSpec, steppers: Vec<Box<dyn BlockStep + 'a>>) -> Self {
        assert_eq!(steppers.len(), spec.blocks.len(), "one stepper per block");
        let oracles = spec.blocks.iter().map(|(_, _, s)| Oracle::new(s)).collect();
        Self { spec, steppers, oracles }
    }

    /// One quiescent tick: the external actions in order, each followed by its event cascade.
    pub fn tick(&self, state: &MachineState, external: &[(usize, ActionInst)]) -> Result<(MachineState, TickTrace)> {
        let out = self.tick_with(state, TickInput { external, pending: Vec::new(), detached: &[] })?;
        Ok((out.state, out.trace))
    }

    /// One tick: held deliveries first, then the external actions in order, each delivery followed
    /// by its cascade. Deliveries to detached blocks are held and returned. Machine `ensures` with an
    /// `after:` are checked only on quiescent ticks (nothing detached, nothing pending), with `old`
    /// the state before the tick; the others are checked every tick.
    pub fn tick_with(&self, state: &MachineState, input: TickInput<'_>) -> Result<TickOutput> {
        let spec = self.spec;
        let mut cur = state.clone();
        let mut trace = TickTrace::default();
        let mut held = Vec::new();
        let mut processed = 0usize;
        let quiescent = input.detached.is_empty() && input.pending.is_empty();
        let mut queue: VecDeque<Delivery> = input.pending.into_iter().collect();
        let mut external = input.external.iter();
        loop {
            let Some(d) = queue.pop_front().or_else(|| external.next().map(|(b, a)| Delivery { block: *b, action: a.clone(), via: None })) else {
                break;
            };
            if input.detached.contains(&d.block) {
                held.push(d);
                continue;
            }
            processed += 1;
            ensure!(processed <= spec.max_events, "tick exceeded {} deliveries: the wiring does not terminate", spec.max_events);
            let Delivery { block, action, via } = d;
            let before = cur[block].clone();
            let after = self.steppers[block].step(&before, &action)?;
            let events = self.oracles[block].events(&before, &action, &after)?;
            let violations: Vec<String> =
                self.oracles[block].violations(&before, Some(&action), &after, crate::continuous::TOLERANCE).into_iter().map(|v| format!("{}: {v}", spec.blocks[block].0)).collect();
            trace.violations.extend(violations.iter().cloned());
            let prev = cur.clone();
            cur[block] = after.clone();
            for e in &events {
                for w in spec.wires.iter().filter(|w| w.from_block == block && w.event == *e) {
                    let scope = MachineScope { spec, cur: &cur, old: Some(&prev) };
                    let params = w
                        .params
                        .iter()
                        .map(|(k, x)| {
                            let v = x.eval(&scope).with_context(|| format!("wire {}.{e} with {k}", spec.blocks[block].0))?;
                            let p = spec.blocks[w.to_block].2.action(&w.action).and_then(|a| a.param(k)).context("wire parameter vanished")?;
                            if let Ok(domain) = p.domain() {
                                ensure!(domain.contains(&v), "wire {}.{e} with {k}: {v:?} is not a value of {}.{}'s {k}", spec.blocks[block].0, spec.blocks[w.to_block].0, w.action);
                            }
                            Ok((k.clone(), v))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    queue.push_back(Delivery { block: w.to_block, action: ActionInst { name: w.action.clone(), params }, via: Some((block, e.clone())) });
                }
            }
            trace.steps.push(StepRecord { block, action, via, before, after, events, violations });
        }
        let scope = MachineScope { spec, cur: &cur, old: Some(state) };
        for e in &spec.ensures {
            let applies = match &e.after {
                None => true,
                Some((b, name)) => quiescent && input.external.iter().any(|(xb, xa)| xb == b && xa.name == *name),
            };
            if applies && !e.holds.eval(&scope).with_context(|| format!("machine ensures {}", e.holds.source()))?.truthy() {
                trace.violations.push(format!("machine ensures: {}", e.holds.source()));
            }
        }
        Ok(TickOutput { state: cur, trace, pending: held })
    }

    /// A derived value of the machine in `state`.
    pub fn derived(&self, state: &MachineState, name: &str) -> Result<Option<Value>> {
        let scope = MachineScope { spec: self.spec, cur: state, old: None };
        Ok(scope.get(name))
    }

    /// Every derived value in `state`, in declaration order.
    pub fn derived_all(&self, state: &MachineState) -> Result<Vec<(String, Value)>> {
        let scope = MachineScope { spec: self.spec, cur: state, old: None };
        self.spec.derived.iter().map(|(n, e)| Ok((n.clone(), e.eval(&scope).with_context(|| format!("derived {n}"))?))).collect()
    }
}

/// Resolves `block__field` and derived names over a machine state.
struct MachineScope<'a> {
    spec: &'a MachineSpec,
    cur: &'a MachineState,
    old: Option<&'a MachineState>,
}

impl MachineScope<'_> {
    fn lookup(&self, name: &str, states: &MachineState) -> Option<Value> {
        if let Some((_, e)) = self.spec.derived.iter().find(|(n, _)| n == name) {
            let inner = MachineScope { spec: self.spec, cur: states, old: None };
            return e.eval(&inner).ok();
        }
        let (inst, field) = name.split_once("__")?;
        let b = self.spec.block_index(inst)?;
        let i = self.spec.blocks[b].2.field_index(field)?;
        Some(states[b][i].clone())
    }
}

impl Scope for MachineScope<'_> {
    fn get(&self, name: &str) -> Option<Value> {
        self.lookup(name, self.cur)
    }

    fn old(&self, name: &str) -> Option<Value> {
        self.lookup(name, self.old.unwrap_or(self.cur))
    }
}

/// Rewrites `block.field` to `block__field` (and `old.block.field` to `old.block__field`) so the
/// expression language, which allows attribute access only on `old`, can name block fields.
pub fn flatten(src: &str, blocks: &[&str]) -> String {
    let chars: Vec<char> = src.chars().collect();
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < chars.len() {
        let at_start = i == 0 || !is_ident(chars[i - 1]) || (i >= 4 && chars[i - 4..i].iter().collect::<String>() == "old.");
        let mut replaced = false;
        if at_start && (chars[i].is_alphabetic() || chars[i] == '_') {
            for b in blocks {
                let n = b.chars().count();
                let matches = chars[i..].iter().take(n).collect::<String>() == *b
                    && chars.get(i + n) == Some(&'.')
                    && chars.get(i + n + 1).is_some_and(|c| c.is_alphabetic() || *c == '_');
                if matches {
                    out.push_str(b);
                    out.push_str("__");
                    i += n + 1;
                    replaced = true;
                    break;
                }
            }
        }
        if !replaced {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn get<'a>(m: &'a Mapping, k: &str) -> Option<&'a Yaml> {
    m.get(Yaml::String(k.to_string()))
}

fn as_str(y: &Yaml) -> Result<String> {
    match y {
        Yaml::String(s) => Ok(s.clone()),
        Yaml::Number(n) => Ok(n.to_string()),
        other => bail!("expected a string, got {other:?}"),
    }
}

fn expr_src(y: &Yaml) -> Result<String> {
    Ok(match y {
        Yaml::Number(n) if n.is_f64() => crate::value::float_repr(n.as_f64().unwrap_or(0.0)),
        Yaml::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        other => as_str(other)?,
    })
}

fn entries(y: Option<&Yaml>) -> Result<Vec<(String, Yaml)>> {
    match y {
        None | Some(Yaml::Null) => Ok(Vec::new()),
        Some(Yaml::Mapping(m)) => m.iter().map(|(k, v)| Ok((as_str(k)?, v.clone()))).collect(),
        Some(other) => bail!("expected a mapping, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{train, TrainConfig};
    use candle_core::Device;
    use rand::Rng;

    fn machine(name: &str) -> MachineSpec {
        MachineSpec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../machines/{name}.nerve"))).unwrap()
    }

    fn odometer() -> MachineSpec {
        machine("odometer")
    }

    fn value(m: &Machine, s: &MachineState) -> i64 {
        m.derived(s, "value").unwrap().unwrap().as_f64().unwrap() as i64
    }

    fn train_digit(spec: &Spec) -> crate::train::Model {
        let cfg = TrainConfig { epochs: 80, d: 16, hidden: 32, holdout: 0.0, log_every: 1000, ..Default::default() };
        let (net, report) = train(spec, &cfg, &Device::Cpu).unwrap();
        assert_eq!(report["all"]["exact"], 1.0, "the digit block itself must be exact: {report}");
        net
    }

    #[test]
    fn flatten_rewrites_block_fields_and_leaves_the_rest() {
        let b = ["d0", "d1"];
        assert_eq!(flatten("d0.d + 10 * d1.d", &b), "d0__d + 10 * d1__d");
        assert_eq!(flatten("value == (old.value + 1) % 1000", &b), "value == (old.value + 1) % 1000");
        assert_eq!(flatten("old.d0.d == 9 and xd0.d", &b), "old.d0__d == 9 and xd0.d");
    }

    #[test]
    fn the_odometer_parses_into_blocks_inputs_and_wires() {
        let m = odometer();
        assert_eq!(m.blocks.iter().map(|(n, t, _)| format!("{n}:{t}")).collect::<Vec<_>>(), vec!["d0:digit", "d1:digit", "d2:digit"]);
        assert_eq!(m.inputs.len(), 4);
        assert_eq!(m.wires.len(), 2);
        assert_eq!((m.wires[0].from_block, m.wires[0].event.as_str(), m.wires[0].to_block, m.wires[0].action.as_str()), (0, "wrap", 1, "inc"));
        assert_eq!(m.state_space_size().unwrap(), 1000);
        assert_eq!(m.ensures.len(), 2);
        // A wire from an event the block does not emit is refused at parse time.
        let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../machines/odometer.nerve")).unwrap();
        let digit = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/digit.nerve")).unwrap();
        let bad = text.replace("on: d0.wrap", "on: d0.overflow");
        assert!(MachineSpec::parse(&bad, &|_| Ok(digit.clone())).unwrap_err().to_string().contains("emits no such event"));
    }

    #[test]
    fn a_carry_cascades_through_the_wires() {
        let spec = odometer();
        let m = Machine::oracle(&spec);
        let inc = (0, ActionInst::new("inc"));
        let s: MachineState = vec![vec![Value::Int(9)], vec![Value::Int(9)], vec![Value::Int(3)]];
        assert_eq!(value(&m, &s), 399);
        let (next, trace) = m.tick(&s, std::slice::from_ref(&inc)).unwrap();
        assert_eq!(value(&m, &next), 400);
        assert!(trace.violations.is_empty(), "{:?}", trace.violations);
        // Three steps: the external inc, then two deliveries along the carry wires.
        assert_eq!(trace.steps.iter().map(|r| (r.block, r.via.clone())).collect::<Vec<_>>(),
                   vec![(0, None), (1, Some((0, "wrap".into()))), (2, Some((1, "wrap".into())))]);
        assert_eq!(trace.steps[0].events, vec!["wrap".to_string()]);
        assert!(trace.steps[2].events.is_empty());
        // 999 wraps to 0 and the machine's own ensures (mod 1000) holds.
        let top: MachineState = vec![vec![Value::Int(9)]; 3];
        let (zero, trace) = m.tick(&top, std::slice::from_ref(&inc)).unwrap();
        assert_eq!((value(&m, &zero), trace.violations.len()), (0, 0));
    }

    #[test]
    fn the_composed_spec_checks_itself_exhaustively() {
        let spec = odometer();
        let report = spec.check(0, 0).unwrap();
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert!(report.exhaustive);
        assert_eq!((report.states_visited, report.ticks), (1000, 4000));
        // Break a wire's direction and the machine's ensures catches it over the reachable states.
        let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../machines/odometer.nerve")).unwrap();
        let digit = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/digit.nerve")).unwrap();
        let bad = MachineSpec::parse(&text.replace("do: d2.inc", "do: d2.reset"), &|_| Ok(digit.clone())).unwrap();
        let problems = bad.check(0, 0).unwrap().problems;
        assert!(problems.iter().any(|p| p.contains("machine ensures")), "{problems:?}");
    }

    #[test]
    fn a_six_digit_odometer_is_checked_by_sampling_and_agrees_with_arithmetic() -> Result<()> {
        // Paper 2, Prediction 1, second half: a million states is past the exhaustive limit, so the
        // check walks; the digit network, used six times, agrees with integer arithmetic on sampled
        // increments from random states.
        let spec = machine("odometer6");
        assert_eq!(spec.state_space_size()?, 1_000_000);
        let report = spec.check(20, 200)?;
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert!(!report.exhaustive);
        let net = train_digit(&spec.blocks[0].2);
        let networks = Machine::with_steppers(&spec, (0..6).map(|_| Box::new(&net) as Box<dyn BlockStep>).collect());
        let samples: usize = std::env::var("NERVE_ODOMETER_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let inc = (0, ActionInst::new("inc"));
        let mut carries = 0;
        for _ in 0..samples {
            // Bias towards states with trailing nines so carries are exercised far more than 1 in 10.
            let trailing = rng.gen_range(0..6usize);
            let mut digits: Vec<i64> = (0..6).map(|_| rng.gen_range(0..10)).collect();
            for d in digits.iter_mut().take(trailing) {
                *d = 9;
            }
            let s: MachineState = digits.iter().map(|d| vec![Value::Int(*d)]).collect();
            let n: i64 = digits.iter().enumerate().map(|(i, d)| d * 10i64.pow(i as u32)).sum();
            let (next, trace) = networks.tick(&s, std::slice::from_ref(&inc))?;
            assert!(trace.violations.is_empty(), "{:?}", trace.violations);
            carries += trace.steps.len() - 1;
            assert_eq!(value(&networks, &next), (n + 1) % 1_000_000, "from {}", spec.state_key(&s));
        }
        assert!(carries > samples, "carries should be exercised: {carries} over {samples} ticks");
        Ok(())
    }

    #[test]
    fn exactness_composes_an_odometer_of_digit_networks() -> Result<()> {
        // Paper 2, Prediction 1: one digit network trained on its whole spec, used three times,
        // equals integer arithmetic on all 1,000 states, with no violation anywhere.
        let spec = odometer();
        let net = train_digit(&spec.blocks[0].2);
        let networks = Machine::with_steppers(&spec, (0..3).map(|_| Box::new(&net) as Box<dyn BlockStep>).collect());
        let oracle = Machine::oracle(&spec);
        let mut checked = 0;
        for a in 0..10 {
            for b in 0..10 {
                for c in 0..10 {
                    let s: MachineState = vec![vec![Value::Int(a)], vec![Value::Int(b)], vec![Value::Int(c)]];
                    for (blk, act) in &spec.inputs {
                        let (n1, t1) = networks.tick(&s, std::slice::from_ref(&(*blk, act.clone())))?;
                        let (n2, _) = oracle.tick(&s, std::slice::from_ref(&(*blk, act.clone())))?;
                        assert_eq!(n1, n2, "{} from {}", act.name, spec.state_key(&s));
                        assert!(t1.violations.is_empty(), "{:?}", t1.violations);
                        if *blk == 0 && act.name == "inc" {
                            assert_eq!(value(&networks, &n1), (value(&oracle, &s) + 1) % 1000);
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 4000);
        Ok(())
    }

    /// The chain at a moment where one tick runs the whole cascade: digit at 9, toggle off, list
    /// one rotation from home with the switch on b, so the clock's advance flips the toggle on,
    /// increments the digit to 0 (wrap), rotates the list home, turns the switch to g, and nothing more.
    fn chain_on_the_brink(spec: &MachineSpec) -> MachineState {
        let mut s = spec.initial();
        let at = |n: &str| spec.block_index(n).unwrap();
        s[at("digit")] = vec![Value::Int(9)];
        // list: e at the front, then a b c d (one rotation moves e to the back: home)
        s[at("list")] = spec.blocks[at("list")].2.state_from_json(&serde_json::json!({"pa": 1, "pb": 2, "pc": 3, "pd": 4, "pe": 0})).unwrap();
        s[at("switch")] = spec.blocks[at("switch")].2.state_from_json(&serde_json::json!({"name": "b", "clockwise_only": false})).unwrap();
        s
    }

    #[test]
    fn wire_parameters_are_computed_over_the_machine_when_the_event_is_raised() -> Result<()> {
        let spec = machine("chain");
        let m = Machine::oracle(&spec);
        let s = chain_on_the_brink(&spec);
        let advance = (spec.block_index("clock").unwrap(), ActionInst::new("advance"));
        let (next, trace) = m.tick(&s, std::slice::from_ref(&advance))?;
        assert!(trace.violations.is_empty(), "{:?}", trace.violations);
        let names: Vec<String> = trace.steps.iter().map(|r| format!("{}.{}", spec.blocks[r.block].0, r.action.name)).collect();
        assert_eq!(names, vec!["clock.advance", "toggle.flip", "digit.inc", "list.move", "switch.set"]);
        // The list's wire chose the item at the front (e) and sent it to the back.
        let mv = &trace.steps[3].action;
        assert_eq!((mv.param("item"), mv.param("to")), (Some(&Value::Str("e".into())), Some(&Value::Int(4))));
        assert_eq!(m.derived(&next, "home")?, Some(Value::Bool(true)));
        // The switch's wire turned it one place clockwise, b -> g, so it did not come home: no reset.
        assert_eq!(trace.steps[4].action.param("to"), Some(&Value::Str("g".into())));
        assert!(trace.steps[4].events.is_empty());
        assert_eq!(m.derived(&next, "count")?, Some(Value::Int(0)));
        Ok(())
    }

    #[test]
    fn detached_blocks_hold_their_deliveries_until_they_are_back() -> Result<()> {
        let spec = machine("chain");
        let m = Machine::oracle(&spec);
        let s = chain_on_the_brink(&spec);
        let list = spec.block_index("list").unwrap();
        let advance = [(spec.block_index("clock").unwrap(), ActionInst::new("advance"))];
        // With the list grabbed, the cascade stops at its door: the move is held, the switch does not turn.
        let out = m.tick_with(&s, TickInput { external: &advance, pending: Vec::new(), detached: &[list] })?;
        assert_eq!(out.trace.steps.len(), 3);
        assert_eq!(out.state[list], s[list]);
        assert_eq!(out.pending.len(), 1);
        assert_eq!((out.pending[0].block, out.pending[0].action.name.as_str()), (list, "move"));
        assert_eq!(out.pending[0].via, Some((spec.block_index("digit").unwrap(), "wrap".to_string())));
        assert!(out.trace.violations.is_empty(), "{:?}", out.trace.violations);
        // Released: the held move is delivered first and its own cascade (list home -> switch) resumes.
        let resumed = m.tick_with(&out.state, TickInput { external: &[], pending: out.pending, detached: &[] })?;
        assert_eq!(resumed.trace.steps.iter().map(|r| spec.blocks[r.block].0.clone()).collect::<Vec<_>>(), vec!["list", "switch"]);
        assert!(resumed.pending.is_empty());
        assert_eq!(m.derived(&resumed.state, "home")?, Some(Value::Bool(true)));
        // The same two ticks, un-grabbed, land on the same machine state.
        let (direct, _) = m.tick(&s, &advance)?;
        assert_eq!(resumed.state, direct);
        Ok(())
    }

    #[test]
    fn the_chain_checks_itself_by_sampling() -> Result<()> {
        let spec = machine("chain");
        assert_eq!(spec.state_space_size()?, 10 * 2 * 10 * 3125 * 6);
        assert_eq!(spec.inputs.len(), 1 + 25 + 3);
        let report = spec.check(30, 300)?;
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert!(!report.exhaustive);
        Ok(())
    }
}
