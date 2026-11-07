//! Supervised training: joint language-model and selection losses.

use crate::data::{Batch, BatchStats, WordCorpus};
use crate::model::DialogModel;
use crate::nn::weighted_cross_entropy;
use crate::optim::Sgd;
use anyhow::Result;
use candle_core::Tensor;
use rand::seq::SliceRandom;
use rand::Rng;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub lr: f64,
    pub min_lr: f64,
    pub decay_rate: f64,
    pub decay_every: usize,
    pub momentum: f64,
    pub nesterov: bool,
    pub clip: f64,
    pub bsz: usize,
    pub max_epoch: usize,
    pub sel_weight: f64,
}

pub struct Losses {
    pub train: f64,
    pub valid: f64,
    pub valid_select: f64,
}

pub struct Engine<'m> {
    model: &'m DialogModel,
    cfg: TrainConfig,
    opt: Sgd,
    word_weights: Tensor,
    sel_weights: Tensor,
    verbose: bool,
}

impl<'m> Engine<'m> {
    pub fn new(model: &'m DialogModel, cfg: TrainConfig, verbose: bool) -> Result<Self> {
        let opt = Sgd::new(model.params.vars(), cfg.lr, cfg.momentum, cfg.nesterov);
        let word_weights = Tensor::ones(model.vocab.word.len(), candle_core::DType::F32, &model.device)?;
        // Disconnects and disagreements carry no supervision for the selection head.
        let item = &model.vocab.item;
        let sel: Vec<f32> = (0..item.len())
            .map(|i| if matches!(item.word(i), "<disconnect>" | "<disagree>") { 0.0 } else { 1.0 })
            .collect();
        let sel_weights = Tensor::from_vec(sel, item.len(), &model.device)?;
        Ok(Self { model, cfg, opt, word_weights, sel_weights, verbose })
    }

    /// Returns `(lm_loss, selection_loss)` for one batch.
    fn losses(&self, batch: &Batch, train: bool) -> Result<(Tensor, Tensor)> {
        let m = self.model;
        let ctx_h = m.forward_context(&batch.ctx)?;
        let lang_h = m.zero_hid(ctx_h.dim(0)?, m.cfg.nhid_lang)?;
        let (logits, lang_hs) = m.forward_lm(&batch.inpt, &lang_h, &ctx_h, train)?;
        let sel = m.forward_selection(&batch.inpt, &lang_hs, &ctx_h, train)?;
        Ok((
            weighted_cross_entropy(&logits, &batch.tgt, &self.word_weights)?,
            weighted_cross_entropy(&sel, &batch.sel_tgt, &self.sel_weights)?,
        ))
    }

    fn update(&mut self, batch: &Batch, train: bool) -> Result<f64> {
        let (lm, sel) = self.losses(batch, train)?;
        let loss = (lm + (sel * self.cfg.sel_weight)?)?;
        let grads = loss.backward()?;
        self.opt.step(&grads, self.cfg.clip)?;
        Ok(loss.to_scalar::<f32>()? as f64)
    }

    pub fn train_pass(&mut self, trainset: &[Batch]) -> Result<(f64, f64)> {
        let start = Instant::now();
        let mut total = 0.0;
        for (i, batch) in trainset.iter().enumerate() {
            total += self.update(batch, true)?;
            if self.verbose && (i + 1) % 100 == 0 {
                let rate = (i + 1) as f64 / start.elapsed().as_secs_f64();
                eprintln!("|   batch {:>4}/{} | loss {:.3} | {rate:.1} batch/s", i + 1, trainset.len(), total / (i + 1) as f64);
            }
        }
        Ok((total / trainset.len().max(1) as f64, start.elapsed().as_secs_f64()))
    }

    /// One supervised step on a random batch, without dropout; used to keep
    /// RL from drifting away from human language.
    pub fn train_single<R: Rng>(&mut self, trainset: &[Batch], rng: &mut R) -> Result<f64> {
        match trainset.choose(rng) {
            Some(batch) => self.update(batch, false),
            None => Ok(0.0),
        }
    }

    /// Returns `(lm_loss per non-pad word, mean selection loss)`.
    pub fn valid_pass(&self, validset: &[Batch], stats: &BatchStats) -> Result<(f64, f64)> {
        let (mut lm_total, mut sel_total) = (0.0, 0.0);
        for batch in validset {
            let (lm, sel) = self.losses(batch, false)?;
            lm_total += batch.tgt.dim(0)? as f64 * lm.to_scalar::<f32>()? as f64;
            sel_total += sel.to_scalar::<f32>()? as f64;
        }
        Ok((lm_total / stats.nonpadn.max(1) as f64, sel_total / validset.len().max(1) as f64))
    }

    fn epoch<R: Rng>(&mut self, corpus: &WordCorpus, epoch: usize, valid: &(Vec<Batch>, BatchStats), rng: &mut R) -> Result<Losses> {
        let (trainset, _) = corpus.batches(&corpus.train, self.cfg.bsz, rng, &self.model.device)?;
        let (train, secs) = self.train_pass(&trainset)?;
        let (valid_lm, valid_select) = self.valid_pass(&valid.0, &valid.1)?;
        if self.verbose {
            let lr = self.opt.lr;
            println!("| epoch {epoch:03} | trainloss {train:.3} | trainppl {:.3} | s/epoch {secs:.2} | lr {lr:.8}", train.exp());
            println!("| epoch {epoch:03} | validloss {valid_lm:.3} | validppl {:.3}", valid_lm.exp());
            println!("| epoch {epoch:03} | validselectloss {valid_select:.3} | validselectppl {:.3}", valid_select.exp());
        }
        Ok(Losses { train, valid: valid_lm, valid_select })
    }

    /// Trains for `max_epoch` epochs, keeps the parameters with the best
    /// validation selection loss, then anneals the learning rate from there.
    pub fn train<R: Rng>(&mut self, corpus: &WordCorpus, rng: &mut R) -> Result<Losses> {
        let valid = corpus.batches(&corpus.valid, self.cfg.bsz, rng, &self.model.device)?;
        let mut best = (f64::INFINITY, self.model.params.snapshot()?);
        let mut last = None;
        for epoch in 1..=self.cfg.max_epoch {
            let losses = self.epoch(corpus, epoch, &valid, rng)?;
            if losses.valid_select < best.0 {
                best = (losses.valid_select, self.model.params.snapshot()?);
            }
            last = Some(losses);
        }
        if self.verbose {
            println!("| start annealing | best validselectloss {:.3} | best validselectppl {:.3}", best.0, best.0.exp());
        }
        self.model.params.restore(&best.1)?;

        let mut lr = self.cfg.lr;
        let mut last_decay = 0;
        for epoch in self.cfg.max_epoch + 1..100 {
            if epoch - last_decay >= self.cfg.decay_every {
                last_decay = epoch;
                lr /= self.cfg.decay_rate;
                if lr < self.cfg.min_lr {
                    break;
                }
                self.opt = Sgd::new(self.model.params.vars(), lr, 0.0, false);
            }
            last = Some(self.epoch(corpus, epoch, &valid, rng)?);
        }
        last.ok_or_else(|| anyhow::anyhow!("no epochs were run"))
    }
}
