//! The machine room's API: machines of blocks, each block stepped by its trained network, every
//! step checked by its oracle, the whole tick compared with the composed oracle.
//!
//! A tick is a pure function of the machine state and the frame's inputs (paper 2 §5.1), so the
//! page holds the state and the server holds nothing between requests, as for single interfaces.
//! The HTTP routes live in `serve::http`; these methods answer them and the WebAssembly build alike.

use crate::machine::{BlockStep, Delivery, Machine, MachineSpec, MachineState, TickInput};
use crate::oracle::Oracle;
use crate::serve::{spec_summary, App, Entry, LearnedEnergy};
use crate::spec::{ActionInst, Spec, State};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value as J};
use std::sync::Arc;

/// The long side of a machine backdrop's raster: drawn once by the oracle, so it can be sharper than
/// the render networks' pictures.
const BACKDROP_MAX: usize = 768;

/// A machine the lab serves, with its document for display.
pub struct MachineEntry {
    pub spec: MachineSpec,
    pub source: String,
}

/// A block stepped by its trained network: the argmin of the network's energies over the
/// block's candidate successors, exactly as `Model::step` picks it.
pub struct NetworkStepper(pub Arc<dyn LearnedEnergy>);

impl BlockStep for NetworkStepper {
    fn step(&self, s: &State, a: &ActionInst) -> Result<State> {
        if let Some(next) = self.0.step(s, a)? {
            return Ok(next);
        }
        let energies = self.0.step_energies(s, a)?.context("this block's network cannot enumerate its successors")?;
        energies.into_iter().min_by(|x, y| x.1.total_cmp(&y.1)).map(|(st, _)| st).context("the block has no candidate successor")
    }
}

/// Agreement between two machine states: exact for finite blocks, within 1% of range for continuous ones.
pub fn agree(m: &MachineSpec, a: &MachineState, b: &MachineState) -> bool {
    m.blocks.iter().zip(a.iter().zip(b)).all(|((_, _, spec), (x, y))| if spec.finite() { x == y } else { crate::continuous::close(spec, x, y, crate::continuous::TOLERANCE) })
}

/// The steppers for a machine: each block's type network when the lab has one, else its oracle.
/// The second value says, per block, whether a network drives it.
pub fn steppers<'a>(app: &'a App, m: &'a MachineSpec) -> (Vec<Box<dyn BlockStep + 'a>>, Vec<bool>) {
    let mut out: Vec<Box<dyn BlockStep + 'a>> = Vec::new();
    let mut learned = Vec::new();
    for (_, ty, spec) in &m.blocks {
        match app.entry(ty).ok().and_then(|e| e.model.clone()) {
            Some(model) => {
                out.push(Box::new(NetworkStepper(model)));
                learned.push(true);
            }
            None => {
                out.push(Box::new(Oracle::new(spec)));
                learned.push(false);
            }
        }
    }
    (out, learned)
}

/// The steppers for a machine with some blocks broken: those run corrupted copies of their networks.
pub fn steppers_with_broken<'a>(app: &'a App, m: &'a MachineSpec, broken: &[usize]) -> (Vec<Box<dyn BlockStep + 'a>>, Vec<bool>) {
    let (mut out, learned) = steppers(app, m);
    for &b in broken {
        if let Some(c) = app.corrupted(&m.blocks[b].1) {
            out[b] = Box::new(NetworkStepper(c));
        }
    }
    (out, learned)
}

impl App {
    /// What a key does from this state: the bound action with its payload evaluated, a verb, or `null`.
    pub fn machine_binding(&self, name: &str, req: &BindingReq) -> Result<J> {
        let e = self.machine(name)?;
        let m = &e.spec;
        let s = state_from_json(m, &req.state)?;
        Ok(match m.binding(&req.key, &s)? {
            None => J::Null,
            Some(crate::machine::Bound::Verb(v)) => json!({"verb": v}),
            Some(crate::machine::Bound::Action { block, action }) => json!({"block": m.blocks[block].0, "action": action.name, "params": action.params_json()}),
        })
    }
}

/// The box every rendered element of a finite block can occupy over its valid states, so the page
/// can give each block a panel that fits whatever it does.
fn extent(spec: &Spec) -> Option<J> {
    if spec.render.is_empty() || !spec.finite() {
        return None;
    }
    let states = spec.state_space().ok()?;
    if states.len() > 20_000 {
        return None;
    }
    let oracle = Oracle::new(spec);
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for s in states.iter().filter(|s| !oracle.breaks_invariant(s)) {
        for r in &spec.render {
            let (x, y) = crate::drag::element_position(spec, &r.name, s).ok()?;
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
    }
    x0.is_finite().then(|| json!({"x0": x0, "y0": y0, "x1": x1, "y1": y1}))
}

fn state_from_json(m: &MachineSpec, j: &J) -> Result<MachineState> {
    m.blocks
        .iter()
        .map(|(inst, _, spec)| spec.state_from_json(j.get(inst).with_context(|| format!("state has no block {inst:?}"))?))
        .collect()
}

fn state_json(m: &MachineSpec, s: &MachineState) -> J {
    J::Object(m.blocks.iter().zip(s).map(|((inst, _, spec), st)| (inst.clone(), spec.state_json(st))).collect())
}

fn positions_json(m: &MachineSpec, s: &MachineState) -> Result<J> {
    let mut out = serde_json::Map::new();
    for ((inst, _, spec), st) in m.blocks.iter().zip(s) {
        out.insert(inst.clone(), App::positions(spec, st)?);
    }
    Ok(J::Object(out))
}

fn derived_json(machine: &Machine, s: &MachineState) -> Result<J> {
    Ok(J::Object(machine.derived_all(s)?.into_iter().map(|(n, v)| (n, v.to_json())).collect()))
}

fn delivery_json(m: &MachineSpec, d: &Delivery) -> J {
    json!({
        "block": m.blocks[d.block].0,
        "action": d.action.name,
        "params": d.action.params_json(),
        "via": d.via.as_ref().map(|(b, e)| json!({"block": m.blocks[*b].0, "event": e})),
    })
}

fn block_action(m: &MachineSpec, block: &str, action: &str, params: &J) -> Result<(usize, ActionInst)> {
    let b = m.block_index(block).with_context(|| format!("unknown block {block:?}"))?;
    let a = m.blocks[b].2.action_from_json(action, if params.is_null() { &J::Null } else { params })?;
    Ok((b, a))
}

#[derive(Deserialize)]
pub struct ExternalReq {
    pub block: String,
    pub action: String,
    #[serde(default)]
    pub params: J,
}

#[derive(Deserialize)]
pub struct ViaReq {
    pub block: String,
    pub event: String,
}

#[derive(Deserialize)]
pub struct DeliveryReq {
    pub block: String,
    pub action: String,
    #[serde(default)]
    pub params: J,
    pub via: Option<ViaReq>,
}

#[derive(Deserialize)]
pub struct TickReq {
    pub state: J,
    #[serde(default)]
    pub external: Vec<ExternalReq>,
    /// Deliveries an earlier tick held for detached blocks, delivered first.
    #[serde(default)]
    pub pending: Vec<DeliveryReq>,
    /// Blocks a pointer holds: deliveries to them are held and returned.
    #[serde(default)]
    pub detached: Vec<String>,
    /// Blocks a visitor broke: their networks are replaced by corrupted copies, and the checks must
    /// catch what the copies get wrong.
    #[serde(default)]
    pub broken: Vec<String>,
}

#[derive(Deserialize)]
pub struct BindingReq {
    pub state: J,
    pub key: String,
}

impl App {
    /// Every machine as the machine room lists it: its blocks (as the lab lists their types, plus
    /// instance, type, whether a network drives it and the box its elements can occupy), inputs,
    /// wires, derived names, ensures and initial state.
    pub fn machines_json(&self) -> J {
        let mut out = Vec::new();
        for e in self.machines() {
            let m = &e.spec;
            let (_, learned) = steppers(self, m);
            let blocks: Vec<J> = m
                .blocks
                .iter()
                .zip(&learned)
                .map(|((inst, ty, spec), net)| {
                    let mut summary = match self.entry(ty) {
                        Ok(entry) => spec_summary(entry),
                        Err(_) => spec_summary(&Entry { spec: spec.clone(), model: None, picture: None, source: String::new() }),
                    };
                    summary["instance"] = json!(inst);
                    summary["type"] = json!(ty);
                    summary["network"] = json!(net);
                    summary["extent"] = extent(spec).unwrap_or(J::Null);
                    summary
                })
                .collect();
            out.push(json!({
                "name": m.name,
                "source": e.source,
                "blocks": blocks,
                "inputs": m.inputs.iter().map(|(b, a)| json!({"block": m.blocks[*b].0, "action": a.name, "params": a.params_json()})).collect::<Vec<_>>(),
                "wires": m.wires.iter().map(|w| json!({
                    "from": m.blocks[w.from_block].0, "event": w.event,
                    "to": m.blocks[w.to_block].0, "action": w.action,
                    "with": J::Object(w.params.iter().map(|(k, x)| (k.clone(), json!(x.source()))).collect()),
                    // Where the wire runs in the machine's picture, port to port, when both ends are placed.
                    "path": m.wire_path(w).map(|(a, b)| json!({"from": [a.0, a.1], "to": [b.0, b.1]})),
                })).collect::<Vec<_>>(),
                // One scene: each placed block's transform, the backdrop's box, and the picture's box.
                "place": J::Object(m.blocks.iter().zip(&m.place).filter_map(|((inst, _, _), p)| p.map(|p| (inst.clone(), json!({"x": p.x, "y": p.y, "scale": p.scale})))).collect()),
                "scene": m.scene.as_ref().map(|sc| json!({"box": [sc.x0, sc.y0, sc.x1, sc.y1]})),
                "picture_box": m.picture_box().map(|(x0, y0, x1, y1)| json!([x0, y0, x1, y1])),
                "derived": m.derived.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
                "ensures": m.ensures.iter().map(|e| json!({
                    "after": e.after.as_ref().map(|(b, a)| format!("{}.{a}", m.blocks[*b].0)),
                    "holds": e.holds.source(),
                })).collect::<Vec<_>>(),
                "max_events": m.max_events,
                "bindings": m.bindings.iter().map(|b| json!({"key": b.key, "does": m.binding_label(b)})).collect::<Vec<_>>(),
                "states": m.state_space_size().unwrap_or(0),
                "initial": state_json(m, &m.initial()),
            }));
        }
        J::Array(out)
    }

    /// Element positions and derived values of a machine state.
    pub fn machine_view(&self, name: &str, state: &J) -> Result<J> {
        let e = self.machine(name)?;
        let s = state_from_json(&e.spec, state)?;
        let machine = Machine::oracle(&e.spec);
        Ok(json!({"positions": positions_json(&e.spec, &s)?, "derived": derived_json(&machine, &s)?}))
    }

    /// The machine's backdrop: its own scene, rasterised by the render oracle. It has no state behind
    /// it, so there is nothing for a network to learn; the blocks' pictures are the networks' and sit
    /// on top of it where `place:` says.
    pub fn machine_backdrop(&self, name: &str) -> Result<J> {
        let e = self.machine(name)?;
        let sc = e.spec.scene.as_ref().with_context(|| format!("machine {name} has no scene"))?;
        let (w, h) = sc.raster_size_up_to(BACKDROP_MAX);
        let px = crate::render::rasterize_scene(sc, w, h)?;
        Ok(json!({"w": w, "h": h, "view": crate::render::raster_view(sc, w, h), "rgb": crate::render::base64(&crate::render::to_rgb8(&px, w, h))}))
    }

    /// One tick of the machine by its networks, and the same tick by the composed oracle for comparison.
    pub fn machine_tick(&self, name: &str, req: &TickReq) -> Result<J> {
        let e = self.machine(name)?;
        let m = &e.spec;
        let state = state_from_json(m, &req.state)?;
        let external: Vec<(usize, ActionInst)> = req.external.iter().map(|x| block_action(m, &x.block, &x.action, &x.params)).collect::<Result<_>>()?;
        let pending: Vec<Delivery> = req
            .pending
            .iter()
            .map(|d| {
                let (block, action) = block_action(m, &d.block, &d.action, &d.params)?;
                let via = d
                    .via
                    .as_ref()
                    .map(|v| Ok::<_, anyhow::Error>((m.block_index(&v.block).with_context(|| format!("unknown block {:?}", v.block))?, v.event.clone())))
                    .transpose()?;
                Ok(Delivery { block, action, via })
            })
            .collect::<Result<_>>()?;
        let detached: Vec<usize> = req.detached.iter().map(|b| m.block_index(b).with_context(|| format!("unknown block {b:?}"))).collect::<Result<_>>()?;
        let broken: Vec<usize> = req.broken.iter().map(|b| m.block_index(b).with_context(|| format!("unknown block {b:?}"))).collect::<Result<_>>()?;

        let (steps, learned) = steppers_with_broken(self, m, &broken);
        let networks = Machine::with_steppers(m, steps);
        let oracle = Machine::oracle(m);
        let input = || TickInput { external: &external, pending: pending.clone(), detached: &detached };
        let t0 = now_ms();
        let out = networks.tick_with(&state, input())?;
        let ms = now_ms() - t0;
        let truth = oracle.tick_with(&state, input())?;

        let steps_json: Vec<J> = out
            .trace
            .steps
            .iter()
            .map(|r| {
                let spec: &Spec = &m.blocks[r.block].2;
                // Per step, the oracle's answer from the same state the network saw.
                let truth_step = Oracle::new(spec).step(&r.before, &r.action)?;
                let agrees = if spec.finite() { r.after == truth_step } else { crate::continuous::close(spec, &r.after, &truth_step, crate::continuous::TOLERANCE) };
                Ok(json!({
                    "block": m.blocks[r.block].0,
                    "action": r.action.name,
                    "params": r.action.params_json(),
                    "via": r.via.as_ref().map(|(b, e)| json!({"block": m.blocks[*b].0, "event": e})),
                    "before": spec.state_json(&r.before),
                    "after": spec.state_json(&r.after),
                    "oracle": spec.state_json(&truth_step),
                    "agrees": agrees,
                    "continuous": !spec.finite(),
                    "events": r.events,
                    "violations": r.violations,
                    "network": learned[r.block],
                }))
            })
            .collect::<Result<_>>()?;
        Ok(json!({
            "state": state_json(m, &out.state),
            "positions": positions_json(m, &out.state)?,
            "derived": derived_json(&networks, &out.state)?,
            "pending": out.pending.iter().map(|d| delivery_json(m, d)).collect::<Vec<_>>(),
            "steps": steps_json,
            "violations": out.trace.violations,
            "oracle": state_json(m, &truth.state),
            "agrees": agree(m, &out.state, &truth.state),
            "tolerance": m.blocks.iter().any(|(_, _, s)| !s.finite()),
            "ms": ms,
            "networks": J::Object(m.blocks.iter().zip(&learned).map(|((inst, _, _), l)| (inst.clone(), json!(l))).collect()),
        }))
    }
}

/// Milliseconds on a monotonic clock where there is one; the browser build measures nothing.
#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    0.0
}
