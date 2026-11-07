//! Supervised training of the dialogue model.

use anyhow::Result;
use clap::Parser;
use negotiator::cli::RuntimeArgs;
use negotiator::data::{Vocab, WordCorpus};
use negotiator::engine::{Engine, TrainConfig};
use negotiator::model::{DialogModel, ModelConfig};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Train a negotiation model", rename_all = "snake_case")]
struct Args {
    /// Location of the data corpus.
    #[arg(long, default_value = "data/negotiate")]
    data: PathBuf,
    #[arg(long, default_value_t = 256)]
    nembed_word: usize,
    #[arg(long, default_value_t = 64)]
    nembed_ctx: usize,
    #[arg(long, default_value_t = 256)]
    nhid_lang: usize,
    #[arg(long, default_value_t = 64)]
    nhid_ctx: usize,
    /// Accepted for compatibility; the reference model does not use it.
    #[arg(long, default_value_t = 64)]
    nhid_strat: usize,
    #[arg(long, default_value_t = 64)]
    nhid_attn: usize,
    #[arg(long, default_value_t = 64)]
    nhid_sel: usize,
    #[arg(long, default_value_t = 20.0)]
    lr: f64,
    #[arg(long, default_value_t = 1e-5)]
    min_lr: f64,
    #[arg(long, default_value_t = 9.0)]
    decay_rate: f64,
    #[arg(long, default_value_t = 1)]
    decay_every: usize,
    #[arg(long, default_value_t = 0.0)]
    momentum: f64,
    #[arg(long)]
    nesterov: bool,
    #[arg(long, default_value_t = 0.2)]
    clip: f64,
    #[arg(long, default_value_t = 0.5)]
    dropout: f32,
    #[arg(long, default_value_t = 0.1)]
    init_range: f64,
    #[arg(long, default_value_t = 30)]
    max_epoch: usize,
    #[arg(long, default_value_t = 25)]
    bsz: usize,
    /// Minimum word frequency to be in the dictionary.
    #[arg(long, default_value_t = 20)]
    unk_threshold: i64,
    #[arg(long, default_value_t = 1.0)]
    sel_weight: f64,
    /// Where to save the model (`.safetensors` weights + `.json` metadata).
    #[arg(long, default_value = "sv_model")]
    model_file: PathBuf,
    #[command(flatten)]
    runtime: RuntimeArgs,
}

fn main() -> Result<()> {
    negotiator::cli::run_with_big_stack(run)
}

fn run() -> Result<()> {
    let args = Args::parse();
    let (device, mut rng) = args.runtime.setup()?;
    let corpus = WordCorpus::load(&args.data, args.unk_threshold, true)?;
    let cfg = ModelConfig {
        nembed_word: args.nembed_word,
        nembed_ctx: args.nembed_ctx,
        nhid_lang: args.nhid_lang,
        nhid_ctx: args.nhid_ctx,
        nhid_attn: args.nhid_attn,
        nhid_sel: args.nhid_sel,
        dropout: args.dropout,
        init_range: args.init_range,
        output_length: corpus.output_length,
        sel_weight: args.sel_weight,
    };
    let vocab = Vocab {
        word: corpus.vocab.word.clone(),
        item: corpus.vocab.item.clone(),
        context: corpus.vocab.context.clone(),
    };
    let model = DialogModel::new(cfg, vocab, &device)?;
    println!("model: {} parameters, vocab {}", model.params.count(), model.vocab.word.len());
    let train_cfg = TrainConfig {
        lr: args.lr,
        min_lr: args.min_lr,
        decay_rate: args.decay_rate,
        decay_every: args.decay_every,
        momentum: args.momentum,
        nesterov: args.nesterov,
        clip: args.clip,
        bsz: args.bsz,
        max_epoch: args.max_epoch,
        sel_weight: args.sel_weight,
    };
    let losses = Engine::new(&model, train_cfg, true)?.train(&corpus, &mut rng)?;
    println!("final selectppl {:.3}", losses.valid_select.exp());
    model.save(&args.model_file)?;
    println!("saved {}", args.model_file.with_extension("safetensors").display());
    Ok(())
}
