//! Paper 2, Prediction 4, and docs/rube-goldberg.md, Prediction 5, on the served networks: the ball
//! machine (three rails in a loop), with the snapping hand-off (`machines/ball.nerve`) and without it
//! (`machines/ball_drift.nerve`), run by the rail network's amortised step head against the composed
//! oracle; and the head against the energy's descent.
//!
//! Prediction 4: without snapping, position error grows with the number of hand-offs and exceeds 1%
//! of range within ten; with snapping it is bounded by one rail's within-episode error at every count.
//! Prediction 5: the step head agrees with descent within tolerance on at least 99% of steps.

use anyhow::{Context, Result};
use candle_core::Device;
use nerve::continuous::{close, ContinuousModel, TOLERANCE};
use nerve::machine::{BlockStep, Machine, MachineSpec, MachineState};
use nerve::machine_api::NetworkStepper;
use nerve::serve::LearnedEnergy;
use nerve::spec::{ActionInst, Spec};
use nerve::train::load_learned;
use nerve::value::Value;
use rand::{Rng, SeedableRng};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn load(machine: &str) -> Result<(MachineSpec, Vec<Arc<dyn LearnedEnergy>>)> {
    let spec = MachineSpec::load(&root().join(format!("machines/{machine}.nerve")))?;
    let models = spec
        .blocks
        .iter()
        .map(|(inst, ty, s)| load_learned(s.clone(), &root().join("models/served").join(ty), &Device::Cpu).with_context(|| format!("block {inst}: no served network for {ty}")))
        .collect::<Result<Vec<_>>>()?;
    Ok((spec, models))
}

fn by_networks<'a>(spec: &'a MachineSpec, models: &[Arc<dyn LearnedEnergy>]) -> Machine<'a> {
    Machine::with_steppers(spec, models.iter().map(|m| Box::new(NetworkStepper(m.clone())) as Box<dyn BlockStep>).collect())
}

/// Position of the ball in the machine: on whichever rail it rolls, as a fraction of that rail's range,
/// plus the rail index, so a hand-off is a jump in the index and the error is comparable across rails.
fn ball(spec: &MachineSpec, s: &MachineState) -> Option<(usize, f64)> {
    for (b, (inst, _, _)) in spec.blocks.iter().enumerate() {
        if inst.starts_with('r') && s[b][1] == Value::Bool(true) {
            return Some((b, s[b][0].as_f64().unwrap_or(0.0)));
        }
    }
    None
}

/// What one run measures: the position error (fraction of the rail's range) after each hand-off the
/// oracle makes, and the within-episode error at every tick both balls spent on the same rail.
struct Drift {
    hand_offs: Vec<f64>,
    within: Vec<f64>,
}

/// Runs the machine by its networks and by the composed oracle in lockstep from a ball dropped at a
/// random offset.
fn drift(spec: &MachineSpec, models: &[Arc<dyn LearnedEnergy>], hand_offs: usize, seed: u64) -> Result<Drift> {
    let networks = by_networks(spec, models);
    let oracle = Machine::oracle(spec);
    let clock = (spec.block_index("clock").unwrap(), ActionInst::new("advance"));
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Drop the ball at a random offset on the first rail, so runs differ: the machine is deterministic,
    // and every later overshoot depends on where the ball started.
    let drop = (spec.block_index("r1").unwrap(), ActionInst { name: "enter_at".into(), params: vec![("to".into(), Value::Float(rng.gen_range(0.0..8.0)))] });
    let mut n = networks.tick(&spec.initial(), std::slice::from_ref(&drop))?.0;
    let mut o = oracle.tick(&spec.initial(), std::slice::from_ref(&drop))?.0;
    let mut out = Drift { hand_offs: Vec::new(), within: Vec::new() };
    let mut last_rail = ball(spec, &o).map(|b| b.0);
    for _ in 0..hand_offs * 40 {
        let (n2, trace) = networks.tick(&n, std::slice::from_ref(&clock))?;
        assert!(trace.violations.is_empty(), "{:?}", trace.violations);
        let (o2, _) = oracle.tick(&o, std::slice::from_ref(&clock))?;
        n = n2;
        o = o2;
        let now = ball(spec, &o);
        let mine = ball(spec, &n);
        if let (Some((rail, op)), Some((nr, np))) = (now, mine) {
            if rail == nr {
                out.within.push((np - op).abs() / 248.0);
            }
        }
        if let (Some((rail, op)), Some(prev)) = (now, last_rail) {
            if rail != prev {
                // The oracle just handed the ball to the next rail: how far off is the networks' ball?
                let e = match mine {
                    Some((nr, np)) if nr == rail => (np - op).abs() / 248.0,
                    Some((nr, np)) => (np + 240.0 * ((nr as f64 - rail as f64).abs()) - op).abs() / 248.0,
                    None => 1.0,
                };
                out.hand_offs.push(e);
                if out.hand_offs.len() == hand_offs {
                    break;
                }
            }
        }
        last_rail = now.map(|b| b.0);
    }
    Ok(out)
}

fn stats(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let max = xs.iter().cloned().fold(0.0, f64::max);
    (mean, max)
}

#[test]
fn prediction_4_snapping_bounds_the_drift_across_hand_offs() -> Result<()> {
    let runs = env_or("NERVE_BALL_RUNS", 200);
    let (snap, m1) = load("ball")?;
    let (free, m2) = load("ball_drift")?;
    let at = |spec: &MachineSpec, models: &[Arc<dyn LearnedEnergy>]| -> Result<([Vec<f64>; 3], Vec<f64>)> {
        let mut out = [Vec::new(), Vec::new(), Vec::new()];
        let mut within = Vec::new();
        for r in 0..runs {
            let d = drift(spec, models, 100, r as u64)?;
            for (k, slot) in [1usize, 10, 100].iter().zip(out.iter_mut()) {
                if let Some(v) = d.hand_offs.get(k - 1) {
                    slot.push(*v);
                }
            }
            within.extend(d.within);
        }
        Ok((out, within))
    };
    let (s, sw) = at(&snap, &m1)?;
    let (f, _) = at(&free, &m2)?;
    for (k, (a, b)) in [1, 10, 100].iter().zip(s.iter().zip(&f)) {
        let (sm, sx) = stats(a);
        let (fm, fx) = stats(b);
        println!("prediction 4: after {k:>3} hand-offs, error with snapping mean {:.4} max {:.4}; without mean {:.4} max {:.4} ({} runs)", sm, sx, fm, fx, a.len());
    }
    let (wm, wx) = stats(&sw);
    println!("prediction 4: within an episode (both balls on the same rail, {} ticks), the error is mean {wm:.4} max {wx:.4}; one tick of travel is {:.4}", sw.len(), 8.0 / 248.0);
    let (s1, _) = stats(&s[0]);
    let (s100, s100x) = stats(&s[2]);
    let (f10, _) = stats(&f[1]);
    let (f100, _) = stats(&f[2]);
    // The claim: with the rule the error after 100 hand-offs is no larger than after one and is bounded by
    // the within-episode error (its worst tick, plus the tolerance); without it the error passes 1% within
    // ten hand-offs and keeps growing.
    assert!(s100 <= s1 + TOLERANCE, "with snapping the error should not grow with hand-offs: {s100:.4} after 100 vs {s1:.4} after 1");
    assert!(s100x <= wx + TOLERANCE, "with snapping the error at a hand-off should stay within one rail's own drift: {s100x:.4} vs {wx:.4}");
    assert!(f10 > TOLERANCE && f100 > f10, "without snapping the error should pass 1% within ten hand-offs and keep growing: {f10:.4} after 10, {f100:.4} after 100");
    Ok(())
}

#[test]
fn prediction_5_the_step_head_agrees_with_descent() -> Result<()> {
    let spec = Spec::load(&root().join("specs/rail.nerve"))?;
    let model = ContinuousModel::load(spec.clone(), &root().join("models/served/rail"), &Device::Cpu)?;
    assert!(model.head_trained, "the served rail network needs a step head");
    let oracle = nerve::oracle::Oracle::new(&spec);
    // On the machine's own ticks: the ball machine runs by its networks from a drop, and every rail step it
    // takes is answered again by the head alone and by descent. A step whose head answer the spec rejects
    // is one `step` would have answered by descent instead (the fallback), counted here.
    let ticks = env_or("NERVE_HEAD_TICKS", 400);
    let (ball, models) = load("ball")?;
    let networks = by_networks(&ball, &models);
    let clock = (ball.block_index("clock").unwrap(), ActionInst::new("advance"));
    let drop = (ball.block_index("r1").unwrap(), ActionInst::new("enter"));
    let mut s = networks.tick(&ball.initial(), std::slice::from_ref(&drop))?.0;
    let (mut n, mut agree, mut exact, mut fallbacks) = (0, 0, 0, 0);
    let mut head_time = 0.0;
    for _ in 0..ticks {
        let (s2, trace) = networks.tick(&s, std::slice::from_ref(&clock))?;
        for st in trace.steps.iter().filter(|st| ball.blocks[st.block].1 == "rail") {
            let t0 = std::time::Instant::now();
            let fast = model.step_head(&st.before, &st.action)?;
            head_time += t0.elapsed().as_secs_f64();
            let slow = model.step_descent(&st.before, &st.action)?;
            let truth = oracle.step(&st.before, &st.action)?;
            n += 1;
            agree += usize::from(close(&spec, &fast, &slow, TOLERANCE));
            exact += usize::from(close(&spec, &fast, &truth, TOLERANCE));
            fallbacks += usize::from(!oracle.violations(&st.before, Some(&st.action), &fast, TOLERANCE).is_empty());
        }
        s = s2;
    }
    println!("prediction 5: on {n} rail steps of {ticks} ticks, the head agrees with descent on {agree}/{n}, is within tolerance of the oracle on {exact}/{n}, and the spec rejected {fallbacks} answers (answered by descent); {:.3} ms per head step", 1000.0 * head_time / n.max(1) as f64);
    // And on steps the machine never takes: uniformly random states and parameters (reported, not claimed).
    let m = env_or("NERVE_HEAD_SAMPLES", 500);
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let mut random_agree = 0;
    for _ in 0..m {
        let s = vec![Value::Float(rng.gen_range(0.0..240.0)), Value::Bool(rng.gen_bool(0.7))];
        let a = match rng.gen_range(0..4) {
            0 => ActionInst::new("enter"),
            1 => ActionInst::new("advance"),
            2 => ActionInst { name: "set".into(), params: vec![("to".into(), Value::Float(rng.gen_range(0.0..248.0)))] },
            _ => ActionInst { name: "enter_at".into(), params: vec![("to".into(), Value::Float(rng.gen_range(0.0..8.0)))] },
        };
        random_agree += usize::from(close(&spec, &model.step_head(&s, &a)?, &model.step_descent(&s, &a)?, TOLERANCE));
    }
    println!("prediction 5: on {m} uniformly random steps, the head agrees with descent on {random_agree}/{m}");
    assert!(agree as f64 >= 0.99 * n as f64, "{agree}/{n}");
    Ok(())
}
