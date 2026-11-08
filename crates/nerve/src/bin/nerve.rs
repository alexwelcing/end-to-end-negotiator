//! `nerve` command-line tool.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use nerve::oracle::{effective_holdout, holdout_mask, Oracle};
use nerve::spec::Spec;
use rand::SeedableRng;
use std::path::PathBuf;

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
        Cmd::Serve { specs, addr, lambda, models, cuda } => serve(&specs, &addr, lambda, models.as_deref(), &device(cuda)?)?,
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

fn serve(dir: &std::path::Path, addr: &str, lambda: f64, models: Option<&std::path::Path>, dev: &candle_core::Device) -> Result<()> {
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
        println!("  {:<22} {} futures{}", spec.name, spec.futures.len(), if model.is_some() { ", trained model" } else { "" });
        entries.push(nerve::serve::Entry { spec, model, source });
    }
    let app = std::sync::Arc::new(nerve::serve::App::new(entries, lambda));
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("nerve drag lab on http://{addr}");
        axum::serve(listener, nerve::serve::router(app)).await?;
        anyhow::Ok(())
    })
}
