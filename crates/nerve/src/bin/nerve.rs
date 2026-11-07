//! `nerve` command-line tool.

use anyhow::Result;
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
        #[arg(long)]
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
        Cmd::Train { spec, out, epochs, batch, lr, seed, holdout, weight_decay, cuda } => {
            let s = Spec::load(&spec)?;
            let cfg = nerve::train::TrainConfig { epochs, batch, lr, seed, holdout, weight_decay, ..Default::default() };
            let (model, report) = nerve::train::train(&s, &cfg, &device(cuda)?)?;
            let dir = out.join(&s.name);
            model.save(&dir, &cfg, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            println!("saved {}", dir.display());
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
        let spec = Spec::load(&p)?;
        let model: Option<std::sync::Arc<dyn nerve::serve::LearnedEnergy>> = match models.map(|m| m.join(&spec.name)) {
            Some(d) if d.join("model.json").exists() => Some(std::sync::Arc::new(nerve::train::Model::load(spec.clone(), &d, dev)?)),
            _ => None,
        };
        println!("  {:<22} {} futures{}", spec.name, spec.futures.len(), if model.is_some() { ", trained model" } else { "" });
        entries.push(nerve::serve::Entry { spec, model });
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
