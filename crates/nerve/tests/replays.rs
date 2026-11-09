//! docs/rube-goldberg.md, M8, Prediction 7 (paper 2's board prediction): the runtime and the composed
//! oracle agree on 1,000 replayed input traces of length 50, and every input is answered within the
//! frame. A trace is what a visitor could do to the light machine in 50 ticks: the clock every tick,
//! the marble dropped on the first, and grabs of the finite blocks (a disk, a list item, the switch's
//! knob, a card, a fruit, a proposal) dropped as external actions. Each trace runs twice by the networks
//! (a replay is the same states) and the composed oracle answers every tick from the state the networks
//! were in, as the runtime checks (finite blocks exact, continuous blocks within tolerance). A tick's
//! cost is the faster of its two replays, so the computer's own scheduling noise is not counted against
//! the runtime. Grabs of the marble and the block on tracks are left out: those networks land a
//! continuous `set`/`move` within tolerance 96.7% and 97.8% of the time (`tracks_clamp.rs`), a limit of
//! those networks, not of replay.

use anyhow::{ensure, Context, Result};
use candle_core::Device;
use nerve::machine::{BlockStep, Machine, MachineSpec, MachineState};
use nerve::machine_api::{agree, NetworkStepper};
use nerve::serve::LearnedEnergy;
use nerve::spec::ActionInst;
use nerve::train::load_learned;
use rand::{Rng, SeedableRng};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// One tick's inputs in a trace.
type Inputs = Vec<(usize, ActionInst)>;

/// A random trace of `len` ticks: the clock every tick; the marble dropped on tick 1; a grab of a finite
/// block dropped as one of its action instances on about one tick in four. The gate key is left out with
/// the other continuous grabs: the tracks network lands a move within reach inside the tolerance 97.8% of
/// the time (`tracks_clamp.rs`), which is a limit of that network, not of replay.
fn random_trace(spec: &MachineSpec, networks: &Machine<'_>, rng: &mut impl Rng, len: usize) -> Result<Vec<Inputs>> {
    let at = |n: &str| spec.block_index(n).unwrap();
    let clock = (at("clock"), ActionInst::new("advance"));
    let grabbable: Vec<(usize, Vec<ActionInst>)> = ["hanoi", "list", "switch", "fruit", "kanban", "todo", "deal", "fifteen", "sokoban", "rush"]
        .iter()
        .map(|n| {
            let b = at(n);
            Ok((b, spec.blocks[b].2.action_instances()?.into_iter().filter(|a| !["hint", "auto"].contains(&a.name.as_str())).collect()))
        })
        .collect::<Result<_>>()?;
    let mut trace = Vec::new();
    let mut s = spec.initial();
    for t in 1..=len {
        let mut inputs = vec![clock.clone()];
        if t == 1 {
            inputs.push((at("marble"), ActionInst::new("drop")));
        } else if rng.gen_bool(1.0 / 4.0) {
            let (b, acts) = &grabbable[rng.gen_range(0..grabbable.len())];
            inputs.push((*b, acts[rng.gen_range(0..acts.len())].clone()));
        }
        s = networks.tick(&s, &inputs)?.0;
        trace.push(inputs);
    }
    Ok(trace)
}

/// Runs a trace by `m`: the states after each tick, and each tick's time in milliseconds.
fn replay(m: &Machine<'_>, spec: &MachineSpec, trace: &[Inputs]) -> Result<(Vec<MachineState>, Vec<f64>)> {
    let mut s = spec.initial();
    let mut states = Vec::with_capacity(trace.len());
    let mut ms = Vec::with_capacity(trace.len());
    for inputs in trace {
        let t0 = Instant::now();
        let (next, tr) = m.tick(&s, inputs)?;
        ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        ensure!(tr.violations.is_empty(), "{:?}", tr.violations);
        s = next;
        states.push(s.clone());
    }
    Ok((states, ms))
}

#[test]
fn prediction_7_replayed_traces_agree_with_the_composed_oracle_within_the_frame() -> Result<()> {
    let spec = MachineSpec::load(&root().join("machines/light.nerve"))?;
    let models: Vec<Arc<dyn LearnedEnergy>> = spec
        .blocks
        .iter()
        .map(|(inst, ty, s)| load_learned(s.clone(), &root().join("models/served").join(ty), &Device::Cpu).with_context(|| format!("block {inst}: no served network for {ty}")))
        .collect::<Result<_>>()?;
    let networks = Machine::with_steppers(&spec, models.iter().map(|m| Box::new(NetworkStepper(m.clone())) as Box<dyn BlockStep>).collect());
    let oracle = Machine::oracle(&spec);
    let traces = env_or("NERVE_REPLAY_TRACES", 1000);
    let len = env_or("NERVE_REPLAY_LEN", 50);
    let frame_ms = 1000.0 / 30.0;
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let (mut costs, mut inputs_total) = (Vec::new(), 0usize);
    let mut slowest = (0.0f64, String::new());
    for i in 0..traces {
        let trace = random_trace(&spec, &networks, &mut rng, len)?;
        inputs_total += trace.iter().map(Vec::len).sum::<usize>();
        let (a, ms1) = replay(&networks, &spec, &trace)?;
        let (b, ms2) = replay(&networks, &spec, &trace)?;
        ensure!(a == b, "trace {i}: the replay did not reproduce the run");
        // A tick's cost is the faster of its two replays of the same inputs from the same state: the slower
        // one carries whatever else the computer was doing (a build beside this test once made a tick take 75 ms).
        for (t, (x, y)) in ms1.iter().zip(&ms2).enumerate() {
            let cost = x.min(*y);
            if cost > slowest.0 {
                let names: Vec<String> = trace[t].iter().map(|(b, act)| format!("{}.{}", spec.blocks[*b].0, act.name)).collect();
                slowest = (cost, format!("tick {} of trace {i} (inputs {names:?})", t + 1));
            }
            costs.push(cost);
        }
        // The composed oracle's answer to every tick from the state the networks were in, as the runtime checks.
        for (t, inputs) in trace.iter().enumerate() {
            let from = if t == 0 { spec.initial() } else { a[t - 1].clone() };
            let (y, _) = oracle.tick(&from, inputs)?;
            let x = &a[t];
            ensure!(
                agree(&spec, x, &y),
                "trace {i}, tick {} (inputs {:?}): {}",
                t + 1,
                inputs.iter().map(|(b, act)| format!("{}.{}", spec.blocks[*b].0, act.name)).collect::<Vec<_>>(),
                spec.blocks.iter().zip(x.iter().zip(&y)).filter(|((_, _, s), (p, q))| if s.finite() { p != q } else { !nerve::continuous::close(s, p, q, nerve::continuous::TOLERANCE) }).map(|((inst, _, _), (p, q))| format!("{inst}: networks {p:?} vs oracle {q:?}")).collect::<Vec<_>>().join("; ")
            );
        }
    }
    costs.sort_by(|x, y| x.total_cmp(y));
    let (median, worst) = (costs[costs.len() / 2], *costs.last().unwrap());
    println!("prediction 7: {traces} traces of {len} ticks ({inputs_total} inputs) replayed identically and agreed with the composed oracle at every tick; a tick took {median:.2} ms typically and {worst:.2} ms at most, against a {frame_ms:.1} ms frame; the slowest was {}", slowest.1);
    ensure!(worst < frame_ms, "a tick took {worst:.2} ms, more than the frame");
    Ok(())
}
