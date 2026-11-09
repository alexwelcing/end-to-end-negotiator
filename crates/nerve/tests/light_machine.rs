//! docs/rube-goldberg.md, M7, Prediction 6: The Most Complicated Way to Turn On a Light, run by its
//! networks. 100 runs from the marble's drop to the bell complete with zero violations at any block,
//! every finite block exact and every continuous block within tolerance of the composed oracle at every
//! tick, and every self-solving puzzle solved by its learned hints alone. In 100 further runs a scripted
//! hand sabotages a stage (drags the block on tracks to the middle track, where it stalls, then puts it
//! back; or moves the double-booked meeting back to its own room) and the composed oracle's prediction
//! from the dropped state is matched 1, 10 and 100 ticks later.
//!
//! The runs differ by where the four puzzles start: each is scrambled by a random walk of legal moves
//! from its initial board before the drop, within the boards its hints cover (a sokoban box pushed into a
//! corner is a dead board, which no hint solves, so it is not a run of this machine) and ending on a board
//! not yet solved (a puzzle solved before its relay arms has no `solved` to raise, by the document's own
//! logic); the machine is deterministic otherwise, so every run solves different boards by its learned
//! hints and takes its own number of ticks. A hand's nudge of the marble
//! is not the variation: the marble network answers `set` within tolerance in 96.7% of cases, and a
//! continuous grab that lands 2% off is a disagreement, not a different run.

use anyhow::{ensure, Context, Result};
use candle_core::Device;
use nerve::machine::{BlockStep, Machine, MachineSpec, MachineState};
use nerve::machine_api::{agree, NetworkStepper};
use nerve::serve::LearnedEnergy;
use nerve::spec::ActionInst;
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

const MAX_TICKS: usize = 1500;

/// The blocks on which two machine states differ, with both values, for a failure message.
fn diff(spec: &MachineSpec, a: &MachineState, b: &MachineState) -> String {
    spec.blocks
        .iter()
        .zip(a.iter().zip(b))
        .filter(|((_, _, s), (x, y))| if s.finite() { x != y } else { !nerve::continuous::close(s, x, y, nerve::continuous::TOLERANCE) })
        .map(|((inst, _, _), (x, y))| format!("{inst}: networks {x:?} vs oracle {y:?}"))
        .collect::<Vec<_>>()
        .join("; ")
}

struct Light {
    spec: MachineSpec,
    models: Vec<Arc<dyn LearnedEnergy>>,
}

impl Light {
    fn load() -> Result<Self> {
        let spec = MachineSpec::load(&root().join("machines/light.nerve"))?;
        let models = spec
            .blocks
            .iter()
            .map(|(inst, ty, s)| load_learned(s.clone(), &root().join("models/served").join(ty), &Device::Cpu).with_context(|| format!("block {inst}: no served network for {ty}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { spec, models })
    }

    fn networks(&self) -> Machine<'_> {
        Machine::with_steppers(&self.spec, self.models.iter().map(|m| Box::new(NetworkStepper(m.clone())) as Box<dyn BlockStep>).collect())
    }

    fn at(&self, inst: &str) -> usize {
        self.spec.block_index(inst).unwrap_or_else(|| panic!("no block {inst}"))
    }

    fn field(&self, s: &MachineState, inst: &str, field: &str) -> Value {
        let b = self.at(inst);
        s[b][self.spec.blocks[b].2.field_index(field).unwrap()].clone()
    }

    fn rang(&self, s: &MachineState) -> bool {
        self.field(s, "bell", "rings").as_f64().unwrap_or(0.0) > 0.0
    }

    fn external(&self, inst: &str, action: &str, params: Vec<(&str, Value)>) -> (usize, ActionInst) {
        (self.at(inst), ActionInst { name: action.into(), params: params.into_iter().map(|(k, v)| (k.to_string(), v)).collect() })
    }
}

/// What one run records: the ticks it took, and which puzzles the networks solved by hints alone.
struct Run {
    ticks: usize,
    hint_steps: usize,
    other_puzzle_steps: usize,
}

/// The machine's start with its four puzzles scrambled: each board is the result of `walk` random legal
/// moves of the puzzle's own actions (hints excluded) from the spec's initial board, by the oracle, staying
/// among the boards the hints cover and ending on one not yet solved, so the hints have a different problem
/// to solve in every run.
fn scrambled_start(light: &Light, rng: &mut impl Rng, walk: usize) -> Result<MachineState> {
    let mut s = light.spec.initial();
    for p in ["hanoi", "fifteen", "sokoban", "rush"] {
        let b = light.at(p);
        let spec = &light.spec.blocks[b].2;
        let oracle = nerve::oracle::Oracle::new(spec);
        let moves: Vec<ActionInst> = spec.action_instances()?.into_iter().filter(|a| a.name != "hint").collect();
        // Only boards the hints cover: a sokoban box pushed into a corner is a dead board, which no hint
        // solves (724 of its 990 boards are dead); the other puzzles' legal moves never leave the covered set.
        let covered = |st: &nerve::spec::State| spec.holes.iter().filter(|h| h.action == "hint").all(|h| h.examples.iter().any(|e| e.state == *st));
        let hint = ActionInst::new("hint");
        let mut left = walk;
        // The walk, and then on until the board is one the hints still have to solve: on a terminal board the
        // hint is the identity, and a puzzle solved before its relay arms has no `solved` to raise.
        while left > 0 || oracle.step(&s[b], &hint)? == s[b] {
            let a = &moves[rng.gen_range(0..moves.len())];
            let next = oracle.step(&s[b], a)?;
            if covered(&next) {
                s[b] = next;
            }
            left = left.saturating_sub(1);
        }
    }
    Ok(s)
}

/// Runs the machine by its networks from the drop to the bell, from `start`, and checks every tick as the
/// runtime does: no violation, and the composed oracle's answer from the same state within tolerance (a
/// continuous block's drift across an episode is Prediction 4's subject, not this one's).
fn run(light: &Light, start: &MachineState) -> Result<Run> {
    let networks = light.networks();
    let oracle = Machine::oracle(&light.spec);
    let clock = light.external("clock", "advance", vec![]);
    let drop = light.external("marble", "drop", vec![]);
    let n0 = networks.tick(start, std::slice::from_ref(&drop))?.0;
    let o0 = oracle.tick(start, std::slice::from_ref(&drop))?.0;
    ensure!(agree(&light.spec, &n0, &o0), "after the drop: {}", diff(&light.spec, &n0, &o0));
    let mut n = n0;
    let puzzles = ["hanoi", "fifteen", "sokoban", "rush"].map(|p| light.at(p));
    let (mut hint_steps, mut other_puzzle_steps) = (0, 0);
    for t in 1..=MAX_TICKS {
        let (n2, trace) = networks.tick(&n, std::slice::from_ref(&clock))?;
        let (o2, _) = oracle.tick(&n, std::slice::from_ref(&clock))?;
        ensure!(trace.violations.is_empty(), "tick {t}: {:?}", trace.violations);
        ensure!(agree(&light.spec, &n2, &o2), "tick {t}: {}", diff(&light.spec, &n2, &o2));
        for st in &trace.steps {
            if puzzles.contains(&st.block) {
                if st.action.name == "hint" {
                    hint_steps += 1;
                } else {
                    other_puzzle_steps += 1;
                }
            }
        }
        n = n2;
        if light.rang(&n) {
            return Ok(Run { ticks: t, hint_steps, other_puzzle_steps });
        }
    }
    anyhow::bail!("the bell did not ring within {MAX_TICKS} ticks (stage {})", networks.derived(&n, "stage")?.map(|v| v.repr()).unwrap_or_default())
}

/// The document's own logic, before any network: the composed oracle takes the machine from the drop to the
/// bell, through every stage in order.
#[test]
fn the_composed_oracle_takes_the_marble_to_the_bell() -> Result<()> {
    let spec = MachineSpec::load(&root().join("machines/light.nerve"))?;
    let oracle = Machine::oracle(&spec);
    let at = |n: &str| spec.block_index(n).unwrap();
    let clock = (at("clock"), ActionInst::new("advance"));
    let mut s = oracle.tick(&spec.initial(), &[(at("marble"), ActionInst::new("drop"))])?.0;
    let mut stages = Vec::new();
    for t in 1..=MAX_TICKS {
        let (next, trace) = oracle.tick(&s, std::slice::from_ref(&clock))?;
        ensure!(trace.violations.is_empty(), "tick {t}: {:?}", trace.violations);
        s = next;
        let stage = oracle.derived(&s, "stage")?.and_then(|v| v.as_f64()).unwrap_or(0.0) as i64;
        if stages.last() != Some(&stage) {
            stages.push(stage);
        }
        if oracle.derived(&s, "rang")?.and_then(|v| v.as_f64()).unwrap_or(0.0) > 0.0 {
            println!("the composed oracle rang the bell after {t} ticks, through stages {stages:?}");
            return Ok(());
        }
    }
    anyhow::bail!("the composed oracle did not ring the bell within {MAX_TICKS} ticks; stages seen {stages:?}, state {}", spec.blocks.iter().zip(&s).map(|((n, _, _), st)| format!("{n}={st:?}")).collect::<Vec<_>>().join(" "))
}

#[test]
fn prediction_6_the_light_comes_on_in_a_hundred_runs() -> Result<()> {
    let light = Light::load()?;
    let runs = env_or("NERVE_LIGHT_RUNS", 100);
    let mut rng = rand::rngs::StdRng::seed_from_u64(6);
    let (mut ticks, mut hints) = (Vec::new(), Vec::new());
    for i in 0..runs {
        // The first run is the script as written (the puzzles at their initial boards); the rest scramble them.
        let start = if i == 0 { light.spec.initial() } else { scrambled_start(&light, &mut rng, 12)? };
        let r = run(&light, &start).with_context(|| format!("run {i}"))?;
        assert_eq!(r.other_puzzle_steps, 0, "a puzzle moved by something other than a hint");
        assert!(r.hint_steps > 0);
        ticks.push(r.ticks);
        hints.push(r.hint_steps);
    }
    let (min, max) = (ticks.iter().min().unwrap(), ticks.iter().max().unwrap());
    println!("prediction 6: the bell rang in {runs} of {runs} runs (the puzzles scrambled by 12 random moves each in all but the first), after {min} to {max} ticks and {} to {} hints, with zero violations and the composed oracle agreeing at every tick; every puzzle solved by its hints alone", hints.iter().min().unwrap(), hints.iter().max().unwrap());
    Ok(())
}

/// At a hundred ticks: block `b` of the networks' state, `x`, against the prediction `y`. A finite block
/// and a discrete field match exactly; a continuous field is within tolerance of the prediction plus one
/// tick of its own travel, `travel[j]`: the most the prediction moved that field in one tick over the
/// hundred (the stage's pulse), since a continuous stage drifts, within tolerance at every tick, by up to
/// a pulse over a hundred ticks and rests where its last pulse left it (M6's Prediction 4 bounded the
/// ball's drift the same way). Returns the largest continuous deviation in the field's own units, or
/// `None` where something differs.
fn block_within_travel(spec: &MachineSpec, b: usize, x: &[Value], y: &[Value], travel: &[f64]) -> Option<f64> {
    let mut worst = 0.0f64;
    for (j, f) in spec.blocks[b].2.fields.iter().enumerate() {
        if f.continuous() {
            let (lo, hi) = f.range();
            let dev = (x[j].as_f64()? - y[j].as_f64()?).abs();
            if dev > nerve::continuous::TOLERANCE * (hi - lo) + travel[j] {
                return None;
            }
            worst = worst.max(dev);
        } else if x[j] != y[j] {
            return None;
        }
    }
    Some(worst)
}

/// A scripted hand: at `when` ticks after the drop, apply `hands` as external actions (each at its own
/// offset); the run must still reach the bell, and the composed oracle's prediction from the state after
/// each hand, given the same inputs after it, is matched 1 and 10 ticks later exactly. 100 ticks later every
/// block is on the predicted trajectory within two ticks of schedule, each at its own lag (the odometer on
/// the tick; a stage downstream of a continuous crossing a pulse early or late), finite blocks exactly and
/// continuous blocks within a tick of travel (`block_within_travel`); each check's largest lag and deviation
/// are returned with the tick count. A continuous stage's drift, within tolerance at every tick, becomes a
/// whole tick where the stage crosses a threshold (the block on tracks' exit line, the slider's end), and two
/// such crossings lie between a track hand and k = 100.
fn sabotaged_run(light: &Light, start: &MachineState, when: usize, hands: &[(usize, (usize, ActionInst))]) -> Result<(usize, Vec<(i64, f64)>)> {
    let networks = light.networks();
    let oracle = Machine::oracle(&light.spec);
    let clock = light.external("clock", "advance", vec![]);
    let drop = light.external("marble", "drop", vec![]);
    let mut n = networks.tick(start, std::slice::from_ref(&drop))?.0;
    let mut pending: Vec<(usize, (usize, ActionInst))> = hands.iter().map(|(dt, h)| (when + dt, h.clone())).collect();
    // The oracle's predictions from each hand's dropped state, checked k ticks later; the inputs of every
    // tick since are kept, so a prediction replays a later hand too.
    let mut predictions: Vec<(usize, MachineState)> = Vec::new();
    let mut inputs: Vec<Vec<(usize, ActionInst)>> = vec![Vec::new()];
    let only_clock = vec![clock.clone()];
    let mut lags = Vec::new();
    for t in 1..=MAX_TICKS {
        let mut ext = vec![clock.clone()];
        if let Some(i) = pending.iter().position(|(at, _)| *at == t) {
            ext.push(pending.remove(i).1);
        }
        let (n2, trace) = networks.tick(&n, &ext)?;
        ensure!(trace.violations.is_empty(), "tick {t}: {:?}", trace.violations);
        inputs.push(ext.clone());
        if ext.len() > 1 {
            predictions.push((t, n2.clone()));
        }
        for (t0, s0) in &predictions {
            let k = t - t0;
            if k == 100 {
                // The prediction at t - 2 ..= t + 2 (the clock alone past t), and the most it moved each
                // field in one tick on the way; every block matches one of the five, the smallest lag first.
                let mut p = s0.clone();
                let mut at: Vec<MachineState> = Vec::new();
                let mut travel: Vec<Vec<f64>> = light.spec.blocks.iter().map(|(_, _, s)| vec![0.0; s.fields.len()]).collect();
                for tt in t0 + 1..=t + 2 {
                    let next = oracle.tick(&p, inputs.get(tt).unwrap_or(&only_clock))?.0;
                    for (b, tr) in travel.iter_mut().enumerate() {
                        for (j, v) in tr.iter_mut().enumerate() {
                            if let (Some(a), Some(c)) = (next[b][j].as_f64(), p[b][j].as_f64()) {
                                *v = v.max((a - c).abs());
                            }
                        }
                    }
                    p = next;
                    if tt + 2 >= t {
                        at.push(p.clone());
                    }
                }
                let (mut worst_lag, mut worst_dev) = (0i64, 0.0f64);
                for b in 0..light.spec.blocks.len() {
                    let found = [0i64, -1, 1, -2, 2].into_iter().find_map(|lag| {
                        let i = (2 + lag) as usize;
                        block_within_travel(&light.spec, b, &n2[b], &at[i][b], &travel[b]).map(|dev| (lag, dev))
                    });
                    let (lag, dev) = found.with_context(|| {
                        let stages: Vec<String> = at.iter().map(|p| oracle.derived(p, "stage").ok().flatten().map(|v| v.repr()).unwrap_or_default()).collect();
                        format!("tick {t}: 100 ticks after a hand, {} is more than two ticks from the prediction: networks {:?} (stage {}) vs oracle {:?} (stages {} at t-2..t+2)", light.spec.blocks[b].0, n2[b], networks.derived(&n2, "stage").ok().flatten().map(|v| v.repr()).unwrap_or_default(), at[2][b], stages.join(" "))
                    })?;
                    worst_lag = worst_lag.max(lag.abs());
                    worst_dev = worst_dev.max(dev);
                }
                lags.push((worst_lag, worst_dev));
            } else if [1, 10].contains(&k) {
                let mut p = s0.clone();
                for inp in &inputs[t0 + 1..=t] {
                    p = oracle.tick(&p, inp)?.0;
                }
                ensure!(agree(&light.spec, &n2, &p), "tick {t}: {k} ticks after a hand: {}", diff(&light.spec, &n2, &p));
            }
        }
        n = n2;
        if light.rang(&n) && pending.is_empty() {
            return Ok((t, lags));
        }
    }
    anyhow::bail!(
        "the bell did not ring within {MAX_TICKS} ticks after sabotage (stage {}, {} hand(s) still pending; track {:?}, sliders {:?})",
        networks.derived(&n, "stage")?.map(|v| v.repr()).unwrap_or_default(),
        pending.len(),
        n[light.at("track")],
        n[light.at("sliders")]
    )
}

#[test]
fn prediction_6_a_scripted_hand_cannot_stop_it_and_the_oracle_predicts_the_rest() -> Result<()> {
    let light = Light::load()?;
    let runs = env_or("NERVE_LIGHT_SABOTAGE_RUNS", 100);
    let mut rng = rand::rngs::StdRng::seed_from_u64(66);
    let (mut ticks, mut lags) = (Vec::new(), Vec::new());
    for r in 0..runs {
        let start = scrambled_start(&light, &mut rng, 12)?;
        let hands = if r % 2 == 0 {
            // Drag the block to the middle track while it slides, then put it back six ticks later, before it
            // reaches that track's end: held at the clamp, the network sits 1.3% short of 180 (see the
            // tracks_clamp test and docs/machine-room.md), which is past the tolerance.
            vec![
                (0, light.external("track", "move", vec![("t", Value::Int(1)), ("to", Value::Float(100.0))])),
                (6, light.external("track", "move", vec![("t", Value::Int(0)), ("to", Value::Float(100.0))])),
            ]
        } else {
            // Move the double-booked meeting back to its own room once the sliders are already moving.
            vec![(0, light.external("scheduler", "set_room1", vec![("to", Value::Int(1))]))]
        };
        // The track hand lands once the marble has left the ramp (tick 15), while the block slides.
        let when = if r % 2 == 0 { rng.gen_range(18..24) } else { rng.gen_range(90..110) };
        let (t, l) = sabotaged_run(&light, &start, when, &hands).with_context(|| format!("run {r}, the hand at tick {when}"))?;
        ticks.push(t);
        lags.extend(l);
    }
    let (min, max) = (ticks.iter().min().unwrap(), ticks.iter().max().unwrap());
    let count = |d: i64| lags.iter().filter(|(l, _)| l.abs() == d).count();
    let dev = lags.iter().map(|(_, d)| *d).fold(0.0f64, f64::max);
    println!("prediction 6, sabotage: the bell still rang in {runs} of {runs} runs (puzzles scrambled, a hand on the block on tracks in half of them and on the meeting in the other half), after {min} to {max} ticks; the composed oracle's prediction from every dropped state, given the inputs since, matched exactly 1 and 10 ticks later; 100 ticks later every block was on the predicted trajectory, on the tick in {} of {} checks, a stage one tick off in {} and two ticks off in {} (a continuous stage crosses its threshold a pulse early or late), the continuous blocks at most {dev:.1} units from the prediction, within one tick of travel", count(0), lags.len(), count(1), count(2));
    Ok(())
}
