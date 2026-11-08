//! Paper 2, Predictions 2 and 3, run on what the lab ships: the served networks in `models/served`
//! driving `machines/chain.nerve`, compared with the composed oracle.
//!
//! Prediction 2: the chain runs for 10^4 ticks with no violation at any block and no disagreement
//! with the composed oracle, under 16 ms per tick on a CPU.
//! Prediction 3: a scripted pointer grabs a list item or the switch's knob at random ticks, drags it
//! and releases; the machine's state 1, 10 and 100 ticks after release equals the composed oracle's
//! prediction from the dropped state, on every one of 1,000 grabs.

use anyhow::{Context, Result};
use candle_core::Device;
use nerve::drag::{element_position, DragSession, EnergyFn};
use nerve::machine::{BlockStep, Delivery, Machine, MachineSpec, TickInput};
use nerve::machine_api::NetworkStepper;
use nerve::serve::LearnedEnergy;
use nerve::spec::ActionInst;
use nerve::train::load_learned;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// The chain and one served network per block, loaded on the CPU.
fn chain() -> Result<(MachineSpec, Vec<Arc<dyn LearnedEnergy>>)> {
    let spec = MachineSpec::load(&root().join("machines/chain.nerve"))?;
    let models = spec
        .blocks
        .iter()
        .map(|(inst, ty, s)| {
            let dir = root().join("models/served").join(ty);
            load_learned(s.clone(), &dir, &Device::Cpu).with_context(|| format!("block {inst}: no served network for {ty} in {}", dir.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((spec, models))
}

fn by_networks<'a>(spec: &'a MachineSpec, models: &[Arc<dyn LearnedEnergy>]) -> Machine<'a> {
    Machine::with_steppers(spec, models.iter().map(|m| Box::new(NetworkStepper(m.clone())) as Box<dyn BlockStep>).collect())
}

fn advance(spec: &MachineSpec) -> (usize, ActionInst) {
    (spec.block_index("clock").expect("the chain has a clock"), ActionInst::new("advance"))
}

#[test]
fn prediction_2_the_chain_runs_ten_thousand_ticks_exactly() -> Result<()> {
    let (spec, models) = chain()?;
    let networks = by_networks(&spec, &models);
    let oracle = Machine::oracle(&spec);
    let ticks = env_or("NERVE_CHAIN_TICKS", 10_000);
    let clock = advance(&spec);
    let mut s = spec.initial();
    let mut fired: HashSet<(usize, String)> = HashSet::new();
    let mut steps = 0usize;
    // The networks cache energies per (state, action), and the chain driven by its clock alone
    // repeats every 300 ticks, so the first 300 ticks are the uncached cost: report both.
    const FIRST: usize = 300;
    let (mut first_time, mut rest_time) = (0.0, 0.0);
    for t in 0..ticks {
        let t0 = Instant::now();
        let (next, trace) = networks.tick(&s, std::slice::from_ref(&clock))?;
        let dt = t0.elapsed().as_secs_f64();
        if t < FIRST {
            first_time += dt;
        } else {
            rest_time += dt;
        }
        let (truth, _) = oracle.tick(&s, std::slice::from_ref(&clock))?;
        assert!(trace.violations.is_empty(), "tick {t} from {}: {:?}", spec.state_key(&s), trace.violations);
        assert_eq!(next, truth, "tick {t}: the networks disagree with the composed oracle from {}", spec.state_key(&s));
        steps += trace.steps.len();
        fired.extend(trace.steps.iter().filter_map(|r| r.via.clone()));
        s = next;
    }
    let first_ms = 1000.0 * first_time / FIRST.min(ticks) as f64;
    let rest_ms = if ticks > FIRST { 1000.0 * rest_time / (ticks - FIRST) as f64 } else { 0.0 };
    println!(
        "prediction 2: {ticks} ticks, {steps} block steps, {} wires fired; {first_ms:.3} ms per tick over the first {FIRST} (energies uncached), {rest_ms:.3} ms per tick after",
        fired.len()
    );
    assert_eq!(fired.len(), spec.wires.len(), "every wire should fire within {ticks} ticks");
    if cfg!(not(debug_assertions)) {
        assert!(first_ms.max(rest_ms) < 16.0, "{first_ms:.2} / {rest_ms:.2} ms per tick is more than a frame");
    }
    Ok(())
}

/// A scripted grab: detach the block, let the machine run a few ticks around it, drag one of the
/// block's elements along a straight path to a landing spot the spec allows, and drop.
struct Grab {
    block: usize,
    gesture: String,
    target: String,
    detached_ticks: usize,
}

#[test]
fn prediction_3_grabs_mid_run_agree_with_the_composed_oracle() -> Result<()> {
    let (spec, models) = chain()?;
    let networks = by_networks(&spec, &models);
    let oracle = Machine::oracle(&spec);
    let grabs = env_or("NERVE_CHAIN_GRABS", 1_000);
    let clock = advance(&spec);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let grabbable: Vec<usize> = ["list", "switch"].iter().map(|n| spec.block_index(n).expect("chain block")).collect();
    let mut s = spec.initial();
    let mut pending: Vec<Delivery> = Vec::new();
    let mut held_total = 0usize;
    let mut changed_total = 0usize;
    for g in 0..grabs {
        // Run normally for a while.
        for _ in 0..rng.gen_range(0..30) {
            let (next, trace) = networks.tick(&s, std::slice::from_ref(&clock))?;
            assert!(trace.violations.is_empty(), "{:?}", trace.violations);
            s = next;
        }
        let block = *grabbable.choose(&mut rng).unwrap();
        let bspec = &spec.blocks[block].2;
        let future = bspec.futures.choose(&mut rng).unwrap();
        // Up to 24 ticks held, so the list (a delivery every 20 ticks) is often grabbed with one queued.
        let grab = Grab { block, gesture: future.name.clone(), target: future.grab.clone().unwrap(), detached_ticks: rng.gen_range(0..25) };
        // The pointer holds the block: the machine keeps ticking, deliveries for the block queue.
        let detached = [grab.block];
        for _ in 0..grab.detached_ticks {
            let out = networks.tick_with(&s, TickInput { external: std::slice::from_ref(&clock), pending: std::mem::take(&mut pending), detached: &detached })?;
            assert!(out.trace.violations.is_empty(), "{:?}", out.trace.violations);
            assert!(out.trace.steps.iter().all(|r| r.block != grab.block), "a held block must not step");
            s = out.state;
            pending = out.pending;
        }
        held_total += pending.len();
        // The drag itself, scored by the block's network as in the lab, along a straight path.
        let model = models[grab.block].clone();
        let energy: Box<EnergyFn> = Box::new(move |a, act, c| model.energy(a, act, c));
        let f = bspec.future(&grab.gesture).unwrap();
        let mut sess = DragSession::new(bspec, f, s[grab.block].clone(), Some(&*energy))?;
        let cands = sess.candidates().to_vec();
        let target = cands.choose(&mut rng).unwrap();
        let from = element_position(bspec, &grab.target, &s[grab.block])?;
        let to = element_position(bspec, &grab.target, &target.state)?;
        for i in 1..=4 {
            let k = i as f64 / 4.0;
            sess.frame((from.0 + (to.0 - from.0) * k, from.1 + (to.1 - from.1) * k))?;
        }
        let drop = sess.release()?;
        if drop != s[grab.block] {
            changed_total += 1;
        }
        let mut dropped = s.clone();
        dropped[grab.block] = drop;
        // From the dropped state, with the held deliveries first: networks and oracle, 100 ticks each.
        let mut n = dropped.clone();
        let mut o = dropped.clone();
        let mut pn = pending.clone();
        let mut po = std::mem::take(&mut pending);
        for k in 1..=100 {
            let on = networks.tick_with(&n, TickInput { external: std::slice::from_ref(&clock), pending: std::mem::take(&mut pn), detached: &[] })?;
            let oo = oracle.tick_with(&o, TickInput { external: std::slice::from_ref(&clock), pending: std::mem::take(&mut po), detached: &[] })?;
            assert!(on.trace.violations.is_empty(), "grab {g}, {k} ticks after release: {:?}", on.trace.violations);
            assert!(on.pending.is_empty() && oo.pending.is_empty());
            n = on.state;
            o = oo.state;
            if k == 1 || k == 10 || k == 100 {
                assert_eq!(n, o, "grab {g} ({}.{}, {} held): the networks disagree with the composed oracle {k} ticks after release", spec.blocks[grab.block].0, grab.gesture, held_total);
            }
        }
        s = n;
    }
    println!("prediction 3: {grabs} grabs, {changed_total} changed the block, {held_total} deliveries held while grabbed; agreement at k = 1, 10, 100 on every grab");
    assert!(held_total > 0, "some grabs should have held deliveries");
    assert!(changed_total > grabs / 4, "most drags should move something: {changed_total} of {grabs}");
    Ok(())
}
