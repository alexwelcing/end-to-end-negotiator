//! `nerve` command-line tool.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use nerve::oracle::{effective_holdout, holdout_mask, Oracle};
use nerve::spec::Spec;
use rand::SeedableRng;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(about = "Nerve: interfaces as spec-checked energy networks")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Type-check a spec and run it against its own invariants.
    Check { spec: PathBuf },
    /// Rescore a trained finite model on the hold-out it was trained with.
    Eval {
        spec: PathBuf,
        /// Directory holding `model.json` and `model.safetensors`.
        model: PathBuf,
        /// Answer only with states the invariants allow, whatever the model was trained with.
        #[arg(long, conflicts_with = "all_states")]
        valid_only: bool,
        /// Answer from every enumerable state: the network on its own, without the spec's invariants.
        #[arg(long)]
        all_states: bool,
        /// Print one JSON line (for an experiment log) instead of a summary.
        #[arg(long)]
        json: bool,
    },
    /// Print every (state, action) transition as JSON lines (finite specs).
    Table {
        spec: PathBuf,
        #[arg(long, default_value_t = 0.15)]
        holdout: f64,
        #[arg(long, default_value = "nerve")]
        salt: String,
    },
    /// Serve every spec with a render block as an interactive drag lab.
    Serve {
        /// Directory of `.nerve` specs.
        #[arg(long, default_value = "specs")]
        specs: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8787")]
        addr: String,
        /// Weight of the learned energy relative to the pixel gap.
        #[arg(long, default_value_t = 1.0)]
        lambda: f64,
        /// Directory of trained models (`<dir>/<spec name>/model.json`), as written by `nerve train`.
        /// The committed set lives in `models/served`; a spec with no model there runs on the spec alone.
        #[arg(long, default_value = "models/served")]
        models: Option<PathBuf>,
        /// Run loaded models on the GPU (needs a `--features cuda` build).
        #[arg(long)]
        cuda: bool,
        /// Directory of `machine:` documents, served at /machine with each block stepped by its type's network.
        #[arg(long, default_value = "machines")]
        machines: PathBuf,
        /// Directory of render networks (`<dir>/<spec name>/picture.json`), as written by `nerve train-picture`.
        #[arg(long, default_value = "models/pictures")]
        pictures: Option<PathBuf>,
        /// Directory of shaders (`<dir>/<spec name>/shader.json`), as written by `nerve train-shader`.
        #[arg(long, default_value = "models/shaders")]
        shaders: Option<PathBuf>,
    },
    /// Train the shader of a spec with a `scene:` (the picture as a function of the pixel, any resolution) and save it to `<out>/<spec name>/`.
    TrainShader {
        spec: PathBuf,
        #[arg(long, default_value = "models/shaders")]
        out: PathBuf,
        #[arg(long, default_value_t = 4000)]
        steps: usize,
        /// States per step.
        #[arg(long, default_value_t = 8)]
        batch: usize,
        /// Random pixels per state per step.
        #[arg(long, default_value_t = 2048)]
        pixels: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Random valid states to train on when the spec is not finite.
        #[arg(long, default_value_t = 256)]
        samples: usize,
        /// The training raster is the scene's raster times this.
        #[arg(long, default_value_t = 4)]
        scale: usize,
        #[arg(long, default_value_t = 128)]
        hidden: usize,
        #[arg(long, default_value_t = 4)]
        layers: usize,
        #[arg(long, default_value_t = 6)]
        octaves: usize,
        #[arg(long, default_value_t = 200)]
        log_every: usize,
        /// `mlp` (whole network per pixel) or `grid` (learned feature grids + a small per-pixel network).
        #[arg(long, default_value = "mlp")]
        kind: String,
        /// Grid rows per level, coarse to fine (grid kind).
        #[arg(long, value_delimiter = ',', default_value = "16,64")]
        grid_bases: Vec<usize>,
        #[arg(long, default_value_t = 8)]
        grid_channels: usize,
        #[arg(long)]
        cuda: bool,
    },
    /// Train the render network of a spec with a `scene:` on its render oracle and save it to `<out>/<spec name>/`.
    TrainPicture {
        spec: PathBuf,
        #[arg(long, default_value = "models/pictures")]
        out: PathBuf,
        #[arg(long, default_value_t = 400)]
        epochs: usize,
        #[arg(long, default_value_t = 32)]
        batch: usize,
        #[arg(long, default_value_t = 2e-3)]
        lr: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// States drawn for a spec whose state space cannot be enumerated.
        #[arg(long, default_value_t = 512)]
        samples: usize,
        #[arg(long, default_value_t = 50)]
        log_every: usize,
        #[arg(long)]
        cuda: bool,
    },
    /// Build the static site (the lab on WebAssembly) into `site/dist`.
    Site {
        #[arg(long, default_value = "site/dist")]
        out: PathBuf,
        /// Reuse the wasm package already in `--out`/pkg instead of rebuilding it.
        #[arg(long)]
        skip_wasm: bool,
    },
    /// Smoke-test a running lab: networks answer, the chain ticks in agreement, pictures draw.
    Smoke {
        #[arg(default_value = "http://127.0.0.1:8787")]
        url: String,
    },
    /// Score a trained model and append its rows to the experiment log.
    Record {
        spec: PathBuf,
        /// Directory holding `model.json` and `model.safetensors`.
        model: PathBuf,
        /// The split the run belongs to (`full`, `h15`, ...).
        tag: String,
        #[arg(long, default_value = "experiments/round2/results.jsonl")]
        log: PathBuf,
    },
    /// Build `models/served` from the experiment log: the best full-spec run per spec.
    ServeSet {
        #[arg(long, default_value = "experiments/round2/results.jsonl")]
        log: PathBuf,
        #[arg(long, default_value = "models/served")]
        served: PathBuf,
        #[arg(long, default_value = "experiments/round2/served.json")]
        manifest: PathBuf,
    },
    /// Print the experiment log as markdown tables (mean ± sd over seeds).
    Summarize {
        specs: Vec<String>,
        #[arg(long, default_value = "experiments/round2/results.jsonl")]
        log: PathBuf,
    },
    /// Write the generated specs (all of them, or the named ones) into `--out`.
    Gen {
        names: Vec<String>,
        #[arg(long, default_value = "specs")]
        out: PathBuf,
    },
    /// Dump the render oracle's rasters of a spec's training states at `scale` times the scene's raster:
    /// `<out>/meta.json` (states, w, h) and `<out>/rasters.f32` (CHW float32, one state after another).
    /// The GPU trainer in `nerve/render/` reads them and writes a shader the engine loads.
    Rasters {
        spec: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 4)]
        scale: usize,
        #[arg(long, default_value_t = 256)]
        samples: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// At most this many states (a seeded sample of a large finite state space); 0 for all.
        #[arg(long, default_value_t = 0)]
        max_states: usize,
    },
    /// Score a shader directory against the render oracle (mean pixel error at its training raster and the scene's).
    EvalShader {
        spec: PathBuf,
        #[arg(long, default_value = "models/shaders")]
        dir: PathBuf,
    },
    /// Write the render oracle's picture of a spec's initial state (or `--state` JSON) as a PPM file.
    Picture {
        spec: PathBuf,
        out: PathBuf,
        #[arg(long)]
        state: Option<String>,
    },
    /// Train the JEPA energy for a finite spec and save it to `<out>/<spec name>/`.
    Train {
        spec: PathBuf,
        #[arg(long, default_value = "runs/rs")]
        out: PathBuf,
        #[arg(long, default_value_t = 60)]
        epochs: usize,
        #[arg(long, default_value_t = 128)]
        batch: usize,
        #[arg(long, default_value_t = 3e-3)]
        lr: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 0.15)]
        holdout: f64,
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f64,
        /// Specs with continuous fields: sampled (state, action) pairs to train on.
        #[arg(long, default_value_t = 4000)]
        pairs: usize,
        /// Evaluate every N epochs (and at the last one).
        #[arg(long, default_value_t = 10)]
        log_every: usize,
        /// Weight of the equivariant factored head (0 turns it off, for ablations).
        #[arg(long, default_value_t = 10.0)]
        fact_weight: f64,
        /// Finite specs: score every enumerable state, including ones that break an invariant (for ablations).
        #[arg(long)]
        all_states: bool,
        /// Which hashed hold-out split: a different salt is a different set of unseen cells.
        #[arg(long, default_value = "nerve")]
        salt: String,
        /// Hold out every cell where this expression over the state (and action params) is true.
        #[arg(long)]
        holdout_rule: Option<String>,
        /// Train on the GPU (needs a `--features cuda` build).
        #[arg(long)]
        cuda: bool,
    },
}

fn device(cuda: bool) -> Result<candle_core::Device> {
    if !cuda {
        return Ok(candle_core::Device::Cpu);
    }
    anyhow::ensure!(candle_core::utils::cuda_is_available(), "--cuda needs a binary built with `--features cuda` (scripts/cargo-cuda.cmd)");
    Ok(candle_core::Device::new_cuda(0)?)
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Serve { specs, addr, lambda, models, cuda, machines, pictures, shaders } => serve(&specs, &addr, lambda, models.as_deref(), &machines, pictures.as_deref(), shaders.as_deref(), &device(cuda)?)?,
        Cmd::TrainShader { spec, out, steps, batch, pixels, lr, seed, samples, scale, hidden, layers, octaves, log_every, kind, grid_bases, grid_channels, cuda } => {
            let s = Spec::load(&spec)?;
            let cfg = nerve::shader::ShaderTrainConfig { steps, batch_states: batch, pixels, lr, seed, samples, scale, log_every };
            let mut sc = if kind == "grid" {
                nerve::shader::ShaderConfig::grid_for_spec(&s, scale, &grid_bases, grid_channels)?
            } else {
                let mut c = nerve::shader::ShaderConfig::for_spec(&s, scale)?;
                c.hidden = hidden;
                c.layers = layers;
                c.octaves = octaves;
                c
            };
            if kind == "grid" {
                sc.hidden = if hidden == 128 { 64 } else { hidden };
                sc.layers = if layers == 4 { 2 } else { layers };
                sc.octaves = if octaves == 6 { 4 } else { octaves };
            }
            let (p, report) = nerve::shader::train_with(&s, &cfg, sc, &device(cuda)?)?;
            let dir = out.join(&s.name);
            p.save(&dir, &cfg, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            println!("saved {}", dir.display());
        }
        Cmd::TrainPicture { spec, out, epochs, batch, lr, seed, samples, log_every, cuda } => {
            let s = Spec::load(&spec)?;
            let cfg = nerve::picture::PictureTrainConfig { epochs, batch, lr, seed, samples, log_every, ..Default::default() };
            let (p, report) = nerve::picture::train(&s, &cfg, &device(cuda)?)?;
            let dir = out.join(&s.name);
            p.save(&dir, &cfg, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            println!("saved {}", dir.display());
        }
        Cmd::Site { out, skip_wasm } => {
            let root = std::env::current_dir()?;
            nerve::site::SiteBuild { dist: root.join(out), root, skip_wasm }.build()?;
        }
        Cmd::Smoke { url } => nerve::smoke::run(&url)?,
        Cmd::Record { spec, model, tag, log } => {
            nerve::exp::record(&spec, &model, &tag, &log)?;
        }
        Cmd::ServeSet { log, served, manifest } => {
            let root = std::env::current_dir()?;
            nerve::exp::serve_set(&log, &root, &served, &manifest)?;
        }
        Cmd::Summarize { specs, log } => print!("{}", nerve::exp::summarize(&log, &specs)?),
        Cmd::Gen { names, out } => {
            for n in nerve::gen::write(&out, &names)? {
                println!("wrote {}", out.join(format!("{n}.nerve")).display());
            }
        }
        Cmd::Rasters { spec, out, scale, samples, seed, max_states } => {
            let s = Spec::load(&spec)?;
            let (w, h) = s.scene.as_ref().context("the spec has no scene")?.raster_size();
            let (w, h) = (w * scale.max(1), h * scale.max(1));
            let mut states = nerve::picture::training_states(&s, samples, seed)?;
            if max_states > 0 && states.len() > max_states {
                use rand::seq::SliceRandom;
                use rand::SeedableRng;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let keep: Vec<_> = {
                    let mut idx: Vec<usize> = (1..states.len()).collect();
                    idx.shuffle(&mut rng);
                    idx.truncate(max_states - 1);
                    idx.sort_unstable();
                    std::iter::once(0).chain(idx).map(|i| states[i].clone()).collect()
                };
                states = keep;
            }
            std::fs::create_dir_all(&out)?;
            let mut raw: Vec<u8> = Vec::with_capacity(states.len() * 3 * w * h * 4);
            for st in &states {
                for v in nerve::render::rasterize(&s, st, w, h)? {
                    raw.extend_from_slice(&v.to_le_bytes());
                }
            }
            std::fs::write(out.join("rasters.f32"), raw)?;
            // the engine's own state encoding, so the trainer never re-implements it
            let enc = nerve::encode::Encoder::new(&s)?;
            let mut xs: Vec<u8> = Vec::new();
            for st in &states {
                for v in enc.state(&s, st)? {
                    xs.extend_from_slice(&v.to_le_bytes());
                }
            }
            std::fs::write(out.join("x.f32"), xs)?;
            let states_json: Vec<serde_json::Value> = states.iter().map(|st| s.state_json(st)).collect();
            let meta = serde_json::json!({"spec": s.name, "w": w, "h": h, "scale": scale, "n": states.len(), "states": states_json,
                "view": nerve::render::raster_view(s.scene.as_ref().unwrap(), w, h), "state_dim": nerve::encode::Encoder::new(&s)?.state_dim});
            std::fs::write(out.join("meta.json"), serde_json::to_string(&meta)?)?;
            println!("wrote {} rasters of {w}x{h} to {}", states.len(), out.display());
        }
        Cmd::EvalShader { spec, dir } => {
            let s = Spec::load(&spec)?;
            let sh = nerve::shader::Shader::load(s.clone(), &dir.join(&s.name), &candle_core::Device::Cpu)?;
            let states = nerve::picture::training_states(&s, 256, 0)?;
            println!("{}", serde_json::to_string_pretty(&sh.evaluate(&states)?)?);
        }
        Cmd::Picture { spec, out, state } => {
            let s = Spec::load(&spec)?;
            let st = match state {
                Some(j) => s.state_from_json(&serde_json::from_str(&j)?)?,
                None => s.initial.clone(),
            };
            let (w, h) = s.scene.as_ref().context("the spec has no scene")?.raster_size();
            let px = nerve::render::rasterize(&s, &st, w, h)?;
            let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
            ppm.extend(nerve::render::to_rgb8(&px, w, h));
            std::fs::write(&out, ppm)?;
            println!("wrote {} ({w}x{h})", out.display());
        }
        Cmd::Train { spec, out, epochs, batch, lr, seed, holdout, weight_decay, pairs, log_every, fact_weight, all_states, salt, holdout_rule, cuda } => {
            let s = Spec::load(&spec)?;
            let cfg = nerve::train::TrainConfig { epochs, batch, lr, seed, holdout, weight_decay, cont_pairs: pairs, log_every, fact_weight, candidates: if all_states { nerve::train::Candidates::All } else { nerve::train::Candidates::Valid }, holdout_salt: salt, holdout_rule, ..Default::default() };
            let dir = out.join(&s.name);
            // Finite specs score every enumerable successor; continuous ones descend the energy.
            let report = if s.finite() {
                let (model, report) = nerve::train::train(&s, &cfg, &device(cuda)?)?;
                model.save(&dir, &cfg, &report)?;
                report
            } else {
                let (model, report) = nerve::continuous::train(&s, &cfg, &device(cuda)?)?;
                model.save(&dir, &cfg, &report)?;
                report
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
            println!("saved {}", dir.display());
        }
        Cmd::Eval { spec, model, valid_only, all_states, json } => {
            use nerve::train::Candidates;
            let s = Spec::load(&spec)?;
            let force = if valid_only { Some(Candidates::Valid) } else if all_states { Some(Candidates::All) } else { None };
            let m = nerve::train::Model::load_as(s.clone(), &model, &candle_core::Device::Cpu, force)?;
            // The same hold-out the model was trained with: fraction, salt and rule.
            let t: nerve::train::TrainConfig = serde_json::from_value(m.meta().context("checkpoint has no model.json")?["train"].clone())?;
            let r = nerve::train::evaluate(&m, &nerve::train::HoldOut::new(&s, &t)?)?;
            if json {
                let line = serde_json::json!({
                    "spec": s.name,
                    "model": model.display().to_string(),
                    "candidates": m.candidates,
                    "candidate_count": m.candidate_count(),
                    "seed": t.seed,
                    "holdout": t.holdout,
                    "salt": t.holdout_salt,
                    "rule": t.holdout_rule,
                    "epochs": t.epochs,
                    "fact_weight": t.fact_weight,
                    "heldout": r["heldout"],
                    "seen": r["seen"],
                    "all": r["all"],
                    "hole_examples": r["hole_examples"],
                });
                println!("{line}");
            } else {
                let (h, he) = (&r["heldout"], &r["hole_examples"]);
                let holes = if he.is_null() {
                    String::new()
                } else {
                    format!(", hole examples seen {} / unseen {}", he["seen"]["exact"], he["heldout"]["exact"])
                };
                println!(
                    "{} ({:?} candidates, {}): held-out {}/{} ({:.3}), violations {:.4}{holes}",
                    model.display(),
                    m.candidates,
                    m.candidate_count(),
                    (h["exact"].as_f64().unwrap_or(0.0) * h["n"].as_f64().unwrap_or(0.0)).round(),
                    h["n"],
                    h["exact"].as_f64().unwrap_or(0.0),
                    r["all"]["violation_rate"].as_f64().unwrap_or(0.0)
                );
            }
        }
        Cmd::Check { spec } => {
            let text = std::fs::read_to_string(&spec).with_context(|| format!("reading {}", spec.display()))?;
            if nerve::machine::is_machine_document(&text) {
                return check_machine(&spec);
            }
            let s = Spec::load(&spec)?;
            let problems = Oracle::new(&s).self_check(&mut rand::rngs::StdRng::seed_from_u64(0))?;
            println!(
                "{}: {} fields, {} actions, {} futures, {} holes, finite={}",
                s.name,
                s.fields.len(),
                s.actions.len(),
                s.futures.len(),
                s.holes.len(),
                s.finite()
            );
            for p in &problems {
                println!("  {p}");
            }
            anyhow::ensure!(problems.is_empty(), "{} problems", problems.len());
            println!("ok");
        }
        Cmd::Table { spec, holdout, salt } => {
            let s = Spec::load(&spec)?;
            let o = Oracle::new(&s);
            let frac = effective_holdout(&s, holdout)?;
            for st in s.state_space()? {
                for a in s.action_instances()? {
                    let next = o.step(&st, &a)?;
                    let line = serde_json::json!({
                        "s": s.state_key(&st),
                        "a": s.action_key(&a),
                        "next": s.state_key(&next),
                        "held": holdout_mask(&s, &st, &a, frac, &salt),
                        "viol": o.violations(&st, Some(&a), &next, 0.01).len(),
                    });
                    println!("{line}");
                }
            }
        }
    }
    Ok(())
}

/// `nerve check` on a `machine:` document: parse the wiring, then run the composed oracle against
/// the machine's `ensures` and every block's invariants.
fn check_machine(path: &Path) -> Result<()> {
    use nerve::machine::{MachineSpec, EXHAUSTIVE_LIMIT};
    const WALKS: usize = 200;
    const HORIZON: usize = 500;
    let m = MachineSpec::load(path)?;
    let blocks: Vec<String> = m.blocks.iter().map(|(inst, ty, _)| format!("{inst}:{ty}")).collect();
    println!(
        "machine {}: {} blocks [{}], {} inputs, {} wires, {} derived, {} ensures, {} states",
        m.name,
        m.blocks.len(),
        blocks.join(" "),
        m.inputs.len(),
        m.wires.len(),
        m.derived.len(),
        m.ensures.len(),
        m.state_space_size().map(|n| n.to_string()).unwrap_or_else(|_| "continuous".to_string())
    );
    let report = m.check(WALKS, HORIZON)?;
    for p in &report.problems {
        println!("  {p}");
    }
    anyhow::ensure!(report.problems.is_empty(), "{} problems", report.problems.len());
    if report.exhaustive {
        println!("ok: exhaustive over {} reachable states, {} ticks", report.states_visited, report.ticks);
    } else {
        println!("ok: more than {EXHAUSTIVE_LIMIT} reachable states; {WALKS} random walks of {HORIZON} steps, {} ticks", report.ticks);
    }
    Ok(())
}

fn serve(dir: &std::path::Path, addr: &str, lambda: f64, models: Option<&std::path::Path>, machines_dir: &std::path::Path, pictures: Option<&std::path::Path>, shaders: Option<&std::path::Path>, dev: &candle_core::Device) -> Result<()> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "nerve"))
        .collect();
    paths.sort();
    let mut entries = Vec::new();
    for p in paths {
        let source = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        let spec = Spec::parse(&source).with_context(|| format!("parsing {}", p.display()))?;
        let model: Option<std::sync::Arc<dyn nerve::serve::LearnedEnergy>> = match models.map(|m| m.join(&spec.name)) {
            Some(d) if d.join("model.json").exists() => Some(nerve::train::load_learned(spec.clone(), &d, dev)?),
            _ => None,
        };
        let picture = match pictures.map(|p| p.join(&spec.name)) {
            Some(d) if d.join(nerve::picture::META_FILE).exists() && spec.scene.is_some() => Some(std::sync::Arc::new(nerve::picture::Picture::load(spec.clone(), &d, dev)?)),
            _ => None,
        };
        let shader = match shaders.map(|p| p.join(&spec.name)) {
            Some(d) if d.join(nerve::shader::META_FILE).exists() && spec.scene.is_some() => match nerve::shader::Shader::load(spec.clone(), &d, dev) {
                Ok(s) => Some(std::sync::Arc::new(s)),
                Err(e) => {
                    eprintln!("  {:<22} shader not loaded: {e:#}", spec.name);
                    None
                }
            },
            _ => None,
        };
        println!("  {:<22} {} futures{}{}{}", spec.name, spec.futures.len(), if model.is_some() { ", trained model" } else { "" }, if picture.is_some() { ", picture" } else { "" }, if shader.is_some() { ", shader" } else { "" });
        entries.push(nerve::serve::Entry { spec, model, picture, shader, source });
    }
    let mut machines = Vec::new();
    if machines_dir.is_dir() {
        let mut mpaths: Vec<PathBuf> = std::fs::read_dir(machines_dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "nerve")).collect();
        mpaths.sort();
        for p in mpaths {
            let source = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
            let spec = nerve::machine::MachineSpec::load(&p)?;
            println!("  machine {:<14} {} blocks, {} wires", spec.name, spec.blocks.len(), spec.wires.len());
            machines.push(nerve::machine_api::MachineEntry { spec, source });
        }
    }
    let app = std::sync::Arc::new(nerve::serve::App::new(entries, lambda).with_machines(machines));
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("nerve drag lab on http://{addr}");
        axum::serve(listener, nerve::serve::router(app)).await?;
        anyhow::Ok(())
    })
}
