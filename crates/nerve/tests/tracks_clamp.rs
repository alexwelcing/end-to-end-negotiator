//! The block on tracks under a machine's pulses (docs/rube-goldberg.md, stage 2): `move(t, to: pos + 7)` every
//! tick, as the light machine's relay asks. On the top track the pulses stop when `exit` fires (the relay
//! disarms), on the middle track they run until the block nears that track's end at 180. Every answer must
//! stay within tolerance of the spec and break no rule. Pulsing on at the clamp is measured but not claimed:
//! held against 180 the served network sits about 1.3% short, past the tolerance, which is why the machine's
//! sabotage hand puts the block back before it gets there.

use anyhow::Result;
use candle_core::Device;
use nerve::continuous::{close, TOLERANCE};
use nerve::oracle::Oracle;
use nerve::spec::{ActionInst, Spec};
use nerve::train::load_learned;
use nerve::value::Value;
use rand::{Rng, SeedableRng};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn the_served_tracks_network_slides_to_the_end_and_holds_at_the_clamp() -> Result<()> {
    let spec = Spec::load(&root().join("specs/tracks.nerve"))?;
    // `NERVE_TRACKS_DIR` tries another checkpoint; the served one by default.
    let dir = std::env::var("NERVE_TRACKS_DIR").map(PathBuf::from).unwrap_or_else(|_| root().join("models/served/tracks"));
    let model = load_learned(spec.clone(), &dir, &Device::Cpu)?;
    let oracle = Oracle::new(&spec);
    let (mut off, mut steps, mut exits) = (0, 0, 0);
    let pulse = |s: &Vec<Value>, track: i64| {
        let end = if track == 1 { 180.0 } else { 240.0 };
        ActionInst { name: "move".into(), params: vec![("t".into(), Value::Int(track)), ("to".into(), Value::Float((s[1].as_f64().unwrap() + 7.0).min(end)))] }
    };
    let mut held_short = None;
    for track in [0i64, 1] {
        let mut s = vec![Value::Int(track), Value::Float(if track == 1 { 60.0 } else { 0.0 })];
        for step in 0..60 {
            let pos = s[1].as_f64().unwrap();
            // The relay's pulses end with the top track's `exit`; on the middle track the hand is back before the clamp.
            if track == 1 && pos + 7.0 >= 180.0 {
                break;
            }
            let a = pulse(&s, track);
            let truth = oracle.step(&s, &a)?;
            let next = model.step(&s, &a)?.expect("the tracks network steps");
            let bad = oracle.violations(&s, Some(&a), &next, TOLERANCE);
            assert!(bad.is_empty(), "track {track}, step {step}: {bad:?}");
            steps += 1;
            if !close(&spec, &next, &truth, TOLERANCE) {
                off += 1;
                println!("track {track}, step {step}: network {next:?}, oracle {truth:?}");
            }
            let exited = oracle.events(&s, &a, &next)?.iter().any(|e| e == "exit");
            exits += usize::from(exited);
            s = next;
            if exited {
                break;
            }
        }
        if track == 1 {
            // Measured, not claimed: pulsing on against the clamp.
            let mut t = s.clone();
            for _ in 0..6 {
                t = model.step(&t, &pulse(&t, 1))?.expect("steps");
            }
            held_short = Some(180.0 - t[1].as_f64().unwrap());
        }
    }
    println!("tracks under pulses: {off} of {steps} steps outside tolerance, exit fired {exits} time(s); held against the middle track's clamp the network sits {:.1} units short of 180", held_short.unwrap_or(0.0));
    assert_eq!(off, 0);
    assert_eq!(exits, 1, "the top track's exit fires once, as the block enters the last stretch");
    Ok(())
}

/// A hand's moves within reach (the gate key, a drop on a track): `move(t, to)` from random positions to
/// random targets inside the destination track's reach, as the light machine's `g` binding asks. How often
/// the served network lands within tolerance is measured (97.8% on 2026-11-08) and floored at 95% here;
/// it is the reason replayed traces (Prediction 7) grab the finite blocks and not this one.
#[test]
fn moves_within_reach_land_within_tolerance() -> Result<()> {
    let spec = Spec::load(&root().join("specs/tracks.nerve"))?;
    let dir = std::env::var("NERVE_TRACKS_DIR").map(PathBuf::from).unwrap_or_else(|_| root().join("models/served/tracks"));
    let model = load_learned(spec.clone(), &dir, &Device::Cpu)?;
    let oracle = Oracle::new(&spec);
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    let n = 400;
    let mut hits = 0;
    for _ in 0..n {
        let track = rng.gen_range(0..3i64);
        let (lo, hi) = if track == 1 { (60.0, 180.0) } else { (0.0, 240.0) };
        let s = vec![Value::Int(track), Value::Float(rng.gen_range(lo..hi))];
        let t = (track + rng.gen_range(-1..=1i64)).clamp(0, 2);
        let (tlo, thi) = if t == 1 { (60.0, 180.0) } else { (0.0, 240.0) };
        let a = ActionInst { name: "move".into(), params: vec![("t".into(), Value::Int(t)), ("to".into(), Value::Float(rng.gen_range(tlo..thi)))] };
        let truth = oracle.step(&s, &a)?;
        let next = model.step(&s, &a)?.expect("steps");
        assert!(oracle.violations(&s, Some(&a), &next, TOLERANCE).is_empty(), "{s:?} {a:?} -> {next:?}");
        hits += usize::from(close(&spec, &next, &truth, TOLERANCE));
    }
    println!("tracks moves within reach: {hits} of {n} within tolerance ({:.1}%)", 100.0 * hits as f64 / n as f64);
    assert!(hits * 100 >= n * 95, "{hits}/{n}");
    Ok(())
}
