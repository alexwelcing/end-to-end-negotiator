//! Two trained models negotiate against each other.

use anyhow::Result;
use clap::Parser;
use negotiator::agent::ModelAgent;
use negotiator::cli::{fork_rng, RolloutArgs, RuntimeArgs};
use negotiator::dialog::{ContextGenerator, Dialog, DialogLogger};
use negotiator::model::DialogModel;
use std::path::PathBuf;
use std::rc::Rc;

#[derive(Parser, Debug)]
#[command(about = "Self-play between two models", rename_all = "snake_case")]
struct Args {
    #[arg(long, alias = "alice_model")]
    alice_model_file: PathBuf,
    #[arg(long, alias = "bob_model")]
    bob_model_file: PathBuf,
    #[arg(long)]
    context_file: PathBuf,
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    /// Print conversations to stderr.
    #[arg(long)]
    verbose: bool,
    #[arg(long, default_value_t = 20)]
    max_turns: usize,
    #[arg(long)]
    log_file: Option<PathBuf>,
    /// Plan Alice's utterances with rollouts.
    #[arg(long)]
    smart_alice: bool,
    /// Plan Bob's utterances with rollouts.
    #[arg(long)]
    smart_bob: bool,
    #[command(flatten)]
    rollout: RolloutArgs,
    /// Stop after this many dialogues (default: one pass over the contexts).
    #[arg(long)]
    max_dialogs: Option<usize>,
    #[arg(long)]
    ref_text: Option<PathBuf>,
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
    let bob_model = Rc::new(DialogModel::load(&args.bob_model_file, &device)?);
    let alice_strategy = args.rollout.strategy(args.smart_alice);
    let bob_strategy = args.rollout.strategy(args.smart_bob);
    let mut alice = ModelAgent::new(alice_model, "Alice", args.temperature, alice_strategy, fork_rng(&mut rng))?;
    let mut bob = ModelAgent::new(bob_model, "Bob", args.temperature, bob_strategy, fork_rng(&mut rng))?;

    let mut dialog = Dialog::new(["Alice", "Bob"], args.ref_text.as_deref(), args.max_turns)?;
    let mut logger = DialogLogger::new(args.verbose, args.log_file.as_deref())?;
    let ctx_gen = ContextGenerator::from_file(&args.context_file)?;
    let limit = args.max_dialogs.unwrap_or(ctx_gen.len());

    for (n, ctxs) in ctx_gen.epochs(1, &mut rng).iter().take(limit).enumerate() {
        logger.dump(&"=".repeat(80), false)?;
        dialog.run([&mut alice, &mut bob], ctxs, &mut logger, &mut rng)?;
        logger.dump(&"=".repeat(80), false)?;
        logger.dump("", false)?;
        if (n + 1) % 100 == 0 {
            logger.dump(&format!("{}: {}", n + 1, dialog.show_metrics()), true)?;
        }
    }
    logger.dump(&format!("final: {}", dialog.show_metrics()), true)?;
    Ok(())
}
