//! Fine-tunes Alice with REINFORCE against Bob, interleaving supervised steps.

use anyhow::Result;
use clap::Parser;
use negotiator::agent::{ModelAgent, RlConfig, Strategy};
use negotiator::cli::{fork_rng, RuntimeArgs};
use negotiator::data::WordCorpus;
use negotiator::dialog::{ContextGenerator, Dialog, DialogLogger};
use negotiator::engine::{Engine, TrainConfig};
use negotiator::model::DialogModel;
use std::path::PathBuf;
use std::rc::Rc;

#[derive(Parser, Debug)]
#[command(about = "Reinforcement learning via self-play", rename_all = "snake_case")]
struct Args {
    #[arg(long, default_value = "data/negotiate")]
    data: PathBuf,
    #[arg(long, default_value_t = 20)]
    unk_threshold: i64,
    #[arg(long, alias = "alice_model")]
    alice_model_file: PathBuf,
    #[arg(long, alias = "bob_model")]
    bob_model_file: PathBuf,
    #[arg(long, default_value = "rl_model")]
    output_model_file: PathBuf,
    #[arg(long)]
    context_file: PathBuf,
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    #[arg(long)]
    verbose: bool,
    /// Accepted for compatibility; unused by the reference implementation.
    #[arg(long, default_value_t = 6)]
    score_threshold: i64,
    #[arg(long)]
    log_file: Option<PathBuf>,
    #[arg(long)]
    smart_bob: bool,
    #[arg(long, default_value_t = 0.99)]
    gamma: f64,
    #[arg(long, default_value_t = 0.5)]
    eps: f64,
    #[arg(long)]
    nesterov: bool,
    #[arg(long, default_value_t = 0.0)]
    momentum: f64,
    #[arg(long, default_value_t = 0.1)]
    lr: f64,
    #[arg(long, default_value_t = 0.1)]
    clip: f64,
    #[arg(long, default_value_t = 0.1)]
    rl_lr: f64,
    #[arg(long, default_value_t = 0.1)]
    rl_clip: f64,
    #[arg(long)]
    ref_text: Option<PathBuf>,
    #[arg(long, default_value_t = 8)]
    bsz: usize,
    /// Run a supervised step every N dialogues (<= 0 disables).
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    sv_train_freq: i64,
    #[arg(long, default_value_t = 1)]
    nepoch: usize,
    #[arg(long, default_value_t = 20)]
    max_turns: usize,
    #[arg(long)]
    max_dialogs: Option<usize>,
    #[command(flatten)]
    runtime: RuntimeArgs,
}

fn main() -> Result<()> {
    negotiator::cli::run_with_big_stack(run)
}

fn run() -> Result<()> {
    let args = Args::parse();
    let (device, mut rng) = args.runtime.setup()?;

    let alice_model = Rc::new(DialogModel::load(&args.alice_model_file, &device)?);
    let rl = RlConfig {
        lr: args.rl_lr,
        momentum: args.momentum,
        nesterov: args.nesterov,
        clip: args.rl_clip,
        gamma: args.gamma,
        eps: args.eps,
    };
    let mut alice =
        ModelAgent::new(alice_model.clone(), "Alice", args.temperature, Strategy::Sample, fork_rng(&mut rng))?.with_rl(rl);
    let bob_strategy = if args.smart_bob { Strategy::rollout() } else { Strategy::Sample };
    let bob_model = Rc::new(DialogModel::load(&args.bob_model_file, &device)?);
    let mut bob = ModelAgent::new(bob_model, "Bob", args.temperature, bob_strategy, fork_rng(&mut rng))?;

    let mut dialog = Dialog::new(["Alice", "Bob"], args.ref_text.as_deref(), args.max_turns)?;
    let mut logger = DialogLogger::new(args.verbose, args.log_file.as_deref())?;
    let ctx_gen = ContextGenerator::from_file(&args.context_file)?;
    let corpus = WordCorpus::load(&args.data, args.unk_threshold, false)?;

    let train_cfg = TrainConfig {
        lr: args.lr,
        min_lr: 0.0,
        decay_rate: 1.0,
        decay_every: 1,
        momentum: args.momentum,
        nesterov: args.nesterov,
        clip: args.clip,
        bsz: args.bsz,
        max_epoch: 0,
        sel_weight: alice_model.cfg.sel_weight,
    };
    let mut engine = Engine::new(&alice_model, train_cfg, false)?;
    let (validset, valid_stats) = corpus.batches(&corpus.valid, args.bsz, &mut rng, &device)?;
    let (trainset, train_stats) = corpus.batches(&corpus.train, args.bsz, &mut rng, &device)?;

    let contexts = ctx_gen.epochs(args.nepoch, &mut rng);
    let limit = args.max_dialogs.unwrap_or(contexts.len());
    for (i, ctxs) in contexts.iter().take(limit).enumerate() {
        let n = i + 1;
        if args.sv_train_freq > 0 && n as i64 % args.sv_train_freq == 0 {
            engine.train_single(&trainset, &mut rng)?;
        }
        logger.dump(&"=".repeat(80), false)?;
        dialog.run([&mut alice, &mut bob], ctxs, &mut logger, &mut rng)?;
        logger.dump(&"=".repeat(80), false)?;
        logger.dump("", false)?;
        if n % 100 == 0 {
            logger.dump(&format!("{n}: {}", dialog.show_metrics()), true)?;
        }
    }

    for (name, set, stats) in [("train", &trainset, &train_stats), ("valid", &validset, &valid_stats)] {
        let (loss, sel) = engine.valid_pass(set, stats)?;
        logger.dump(&format!("final: {name}_loss {loss:.3} {name}_ppl {:.3}", loss.exp()), true)?;
        logger.dump(&format!("final: {name}_select_loss {sel:.3} {name}_select_ppl {:.3}", sel.exp()), true)?;
    }
    logger.dump(&format!("final: {}", dialog.show_metrics()), true)?;
    alice_model.save(&args.output_model_file)?;
    Ok(())
}
