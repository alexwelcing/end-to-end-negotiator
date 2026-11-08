//! docs/rube-goldberg.md, M6: one scene. The ball machine's rails sit end to end in machine coordinates,
//! so r1's exit port is r2's entry port, and the picture the page composites is the rails' render networks'
//! rasters placed where the document says. Over 100 runs by the served networks, at every hand-off the ball's
//! picture (found in the rail's raster, not read from the state) is at the exit port of the rail it leaves
//! on the last tick it is drawn there, and at the entry port of the next rail on the next tick.

use anyhow::{Context, Result};
use candle_core::Device;
use nerve::machine::{BlockStep, Machine, MachineSpec, MachineState};
use nerve::machine_api::NetworkStepper;
use nerve::picture::Picture;
use nerve::render::{color, raster_transform};
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

/// Where the ball is drawn in a raster: the centroid of the pixels in the ball's fill colour, in pixels,
/// or `None` when the raster shows no ball.
fn ball_pixels(px: &[f32], w: usize, h: usize, fill: [f32; 3]) -> Option<(f64, f64)> {
    let n = w * h;
    let (mut sx, mut sy, mut c) = (0.0, 0.0, 0usize);
    for k in 0..n {
        let d = (0..3).map(|ch| (px[ch * n + k] - fill[ch]).abs()).sum::<f32>() / 3.0;
        if d < 0.12 {
            sx += (k % w) as f64 + 0.5;
            sy += (k / w) as f64 + 0.5;
            c += 1;
        }
    }
    (c >= 4).then(|| (sx / c as f64, sy / c as f64))
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

#[test]
fn the_ball_leaves_at_the_exit_port_and_arrives_at_the_entry_port() -> Result<()> {
    let m = MachineSpec::load(&root().join("machines/ball.nerve"))?;
    assert!(m.has_picture(), "machines/ball.nerve places its blocks");
    let rails: Vec<usize> = (0..m.blocks.len()).filter(|&b| m.blocks[b].1 == "rail").collect();
    // The layout's promise: each rail's exit port is the next rail's entry port, in machine coordinates.
    for pair in rails.windows(2) {
        let (a, b) = (m.port(pair[0], "exit").unwrap(), m.port(pair[1], "enter").unwrap());
        assert!(dist(a, b) < 1e-9, "{} exits at {a:?}, {} enters at {b:?}", m.blocks[pair[0]].0, m.blocks[pair[1]].0);
    }
    let models: Vec<Arc<dyn LearnedEnergy>> = m
        .blocks
        .iter()
        .map(|(inst, ty, s)| load_learned(s.clone(), &root().join("models/served").join(ty), &Device::Cpu).with_context(|| format!("block {inst}: no served network for {ty}")))
        .collect::<Result<_>>()?;
    let rail_spec = Spec::load(&root().join("specs/rail.nerve"))?;
    let picture = Picture::load(rail_spec.clone(), &root().join("models/pictures/rail"), &Device::Cpu)?;
    let scene = rail_spec.scene.as_ref().context("the rail has a scene")?;
    let (w, h) = scene.raster_size();
    let (s, ox, oy) = raster_transform(scene, w, h);
    let fill = color("f2")?;
    let networks = Machine::with_steppers(&m, models.iter().map(|x| Box::new(NetworkStepper(x.clone())) as Box<dyn BlockStep>).collect());
    let clock = (m.block_index("clock").unwrap(), ActionInst::new("advance"));
    // The ball as drawn: which rail's raster shows it, and where, in machine coordinates.
    let drawn = |st: &MachineState| -> Result<Option<(usize, (f64, f64))>> {
        for &b in &rails {
            let px = picture.draw(&st[b])?;
            if let Some((cx, cy)) = ball_pixels(&px, w, h, fill) {
                return Ok(Some((b, m.to_machine(b, (ox + cx / s, oy + cy / s)).unwrap())));
            }
        }
        Ok(None)
    };
    let runs = env_or("NERVE_SCENE_RUNS", 100);
    let step = 8.0; // the rail's travel per tick, in its own units
    let tol = 6.0; // machine units: a raster pixel is about two units, and the network draws the ball a little off
    let (mut hand_offs, mut worst_exit, mut worst_entry) = (0usize, 0.0f64, 0.0f64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    for _ in 0..runs {
        let drop = (m.block_index("r1").unwrap(), ActionInst { name: "enter_at".into(), params: vec![("to".into(), Value::Float(rng.gen_range(0.0..8.0)))] });
        let mut st = networks.tick(&m.initial(), std::slice::from_ref(&drop))?.0;
        let mut prev = drawn(&st)?;
        let mut seen = 0;
        for _ in 0..130 {
            let (next, trace) = networks.tick(&st, std::slice::from_ref(&clock))?;
            assert!(trace.violations.is_empty(), "{:?}", trace.violations);
            st = next;
            let now = drawn(&st)?;
            if let (Some((pb, ppos)), Some((nb, npos))) = (prev, now) {
                if pb != nb {
                    let (exit, entry) = (m.port(pb, "exit").unwrap(), m.port(nb, "enter").unwrap());
                    let scale = m.place[pb].unwrap().scale;
                    let (de, dn) = (dist(ppos, exit), dist(npos, entry));
                    assert!(de <= step * scale + tol, "{} left its rail {de:.1} units from the exit port", m.blocks[pb].0);
                    assert!(dn <= tol, "{} shows the ball {dn:.1} units from its entry port", m.blocks[nb].0);
                    (worst_exit, worst_entry) = (worst_exit.max(de), worst_entry.max(dn));
                    hand_offs += 1;
                    seen += 1;
                    if seen == 3 {
                        break;
                    }
                }
            }
            prev = now;
        }
        assert_eq!(seen, 3, "a run should go round all three rails");
    }
    println!(
        "one scene: {hand_offs} hand-offs over {runs} runs; the ball's picture left a rail at most {worst_exit:.1} units short of its exit port (one tick of travel is {step}) and appeared at most {worst_entry:.1} units from the next rail's entry port"
    );
    Ok(())
}
