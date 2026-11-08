//! Negotiate against a trained model from the terminal.

use anyhow::Result;
use clap::Parser;
use negotiator::agent::{HumanAgent, ModelAgent};
use negotiator::cli::{fork_rng, RolloutArgs, RuntimeArgs};
use negotiator::dialog::{ContextGenerator, Dialog, DialogLogger};
use negotiator::model::DialogModel;
use std::path::PathBuf;
use std::rc::Rc;

#[derive(Parser, Debug)]
#[command(about = "Chat with a negotiation model", rename_all = "snake_case")]
struct Args {
    #[arg(long)]
    model_file: PathBuf,
    #[arg(long)]
    context_file: PathBuf,
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    /// Plan the bot's utterances with rollouts.
    #[arg(long)]
    smart_ai: bool,
    #[command(flatten)]
    rollout: RolloutArgs,
    #[arg(long, default_value_t = 20)]
    max_turns: usize,
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
    let model = Rc::new(DialogModel::load(&args.model_file, &device)?);
    let strategy = args.rollout.strategy(args.smart_ai);
    let mut ai = ModelAgent::new(model, "Alice", args.temperature, strategy, fork_rng(&mut rng))?;
    let mut human = HumanAgent::new("Human");
    let mut dialog = Dialog::new(["Human", "Alice"], args.ref_text.as_deref(), args.max_turns)?;
    let mut logger = DialogLogger::new(true, None)?;
    let ctx_gen = ContextGenerator::from_file(&args.context_file)?;

    logger.dump("Welcome to our Chatroulette! Type <selection> to end the negotiation.", false)?;
    for id in 0.. {
        logger.dump(&"=".repeat(80), false)?;
        logger.dump(&format!("Dialog {id}"), false)?;
        logger.dump(&"-".repeat(80), false)?;
        let ctxs = ctx_gen.sample(&mut rng).clone();
        dialog.run([&mut human, &mut ai], &ctxs, &mut logger, &mut rng)?;
        logger.dump(&"=".repeat(80), false)?;
        logger.dump("", false)?;
    }
    Ok(())
}
