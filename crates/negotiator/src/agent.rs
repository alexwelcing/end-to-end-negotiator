//! Negotiating agents: model-driven (sampling, rollout planning, RL) and human.

use crate::domain::ObjectDivision;
use crate::model::DialogModel;
use crate::optim::Sgd;
use anyhow::{ensure, Result};
use candle_core::Tensor;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::Rng;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::rc::Rc;

pub trait Agent {
    fn name(&self) -> &str;
    fn is_human(&self) -> bool {
        false
    }
    fn feed_context(&mut self, ctx: &[String]) -> Result<()>;
    fn read(&mut self, words: &[String]) -> Result<()>;
    fn write(&mut self) -> Result<Vec<String>>;
    fn choose(&mut self) -> Result<Vec<String>>;
    fn update(&mut self, _agree: bool, _reward: i64) -> Result<()> {
        Ok(())
    }
}

/// How a model agent produces its next utterance.
#[derive(Debug, Clone)]
pub enum Strategy {
    /// Sample one utterance from the model.
    Sample,
    /// Sample candidates, roll each out to the end of the dialogue several
    /// times, and keep the one with the best expected score.
    Rollout { ncandidate: usize, nrollout: usize, rollout_len: usize },
    /// Sample many complete continuations in one batch on the GPU and group
    /// them by first utterance.
    BatchedRollout { bsz: usize, count_threshold: usize },
}

impl Strategy {
    pub fn rollout() -> Self {
        Self::Rollout { ncandidate: 10, nrollout: 5, rollout_len: 100 }
    }
}

pub struct RlConfig {
    pub lr: f64,
    pub momentum: f64,
    pub nesterov: bool,
    pub clip: f64,
    pub gamma: f64,
    pub eps: f64,
}

/// REINFORCE state: the agent's own optimizer plus this dialogue's actions.
struct Rl {
    cfg: RlConfig,
    opt: Sgd,
    all_rewards: Vec<f64>,
    logprobs: Vec<Tensor>,
}

struct Choice {
    choice: Vec<String>,
    logprob: Option<Tensor>,
    p_agree: f64,
}

pub struct ModelAgent {
    model: Rc<DialogModel>,
    name: String,
    domain: ObjectDivision,
    temperature: f64,
    strategy: Strategy,
    rl: Option<Rl>,
    rng: StdRng,
    stop: [u32; 2],
    eos: u32,
    eod: u32,
    context: Vec<String>,
    ctx_h: Option<Tensor>,
    lang_h: Option<Tensor>,
    /// Every token this agent has seen, aligned row-for-row with `lang_hs`.
    words: Vec<u32>,
    lang_hs: Vec<Tensor>,
}

impl ModelAgent {
    pub fn new(model: Rc<DialogModel>, name: &str, temperature: f64, strategy: Strategy, rng: StdRng) -> Result<Self> {
        let eos = model.word_idx("<eos>")?;
        let eod = model.word_idx("<selection>")?;
        Ok(Self {
            model,
            name: name.to_string(),
            domain: ObjectDivision,
            temperature,
            strategy,
            rl: None,
            rng,
            stop: [eos, eod],
            eos,
            eod,
            context: Vec::new(),
            ctx_h: None,
            lang_h: None,
            words: Vec::new(),
            lang_hs: Vec::new(),
        })
    }

    /// Turns this agent into a REINFORCE learner that updates its model.
    pub fn with_rl(mut self, cfg: RlConfig) -> Self {
        let opt = Sgd::new(self.model.params.vars(), cfg.lr, cfg.momentum, cfg.nesterov);
        self.rl = Some(Rl { cfg, opt, all_rewards: Vec::new(), logprobs: Vec::new() });
        self
    }

    fn state(&self) -> Result<(&Tensor, &Tensor)> {
        match (&self.ctx_h, &self.lang_h) {
            (Some(c), Some(l)) => Ok((c, l)),
            _ => anyhow::bail!("{}: feed_context must be called first", self.name),
        }
    }

    fn push_turn(&mut self, words: &[u32], lang_hs: Tensor, lang_h: Tensor) -> Result<()> {
        self.words.push(self.model.you_token());
        self.words.extend_from_slice(words);
        self.lang_hs.push(lang_hs);
        self.lang_h = Some(lang_h);
        self.check_aligned()
    }

    fn check_aligned(&self) -> Result<()> {
        let rows: usize = self.lang_hs.iter().map(|t| t.dim(0).unwrap_or(0)).sum();
        ensure!(rows == self.words.len(), "{}: {} states for {} words", self.name, rows, self.words.len());
        Ok(())
    }

    /// Scores every legal split given a (possibly hypothetical) dialogue.
    fn choose_from(&mut self, words: &[u32], lang_hs: &Tensor, sample: bool) -> Result<Choice> {
        let (ctx_h, _) = self.state()?;
        let m = &self.model;
        let logits = m.choice_logits(words, lang_hs, ctx_h)?; // (slots, n_items)
        let choices = self.domain.generate_choices(&self.context)?;
        let slots = self.domain.selection_length();
        let idx = (0..slots)
            .flat_map(|i| choices.iter().map(move |c| m.vocab.item.idx(&c[i]).map(|x| x as u32)))
            .collect::<Result<Vec<_>>>()?;
        let idx = Tensor::from_vec(idx, (slots, choices.len()), &m.device)?;
        let choice_logit = logits.gather(&idx, 1)?.sum(0)?;
        let logprob = candle_nn::ops::log_softmax(&choice_logit, 0)?;
        let probs: Vec<f32> = logprob.exp()?.to_vec1()?;
        let pick = if sample {
            WeightedIndex::new(&probs)?.sample(&mut self.rng)
        } else {
            probs.iter().enumerate().fold(0, |best, (i, p)| if *p > probs[best] { i } else { best })
        };
        Ok(Choice {
            choice: choices[pick][..slots].to_vec(),
            logprob: sample.then(|| logprob.get(pick)).transpose()?,
            p_agree: probs[pick] as f64,
        })
    }

    fn history_with(&self, extra_hs: &[&Tensor]) -> Result<Tensor> {
        let all: Vec<&Tensor> = self.lang_hs.iter().chain(extra_hs.iter().copied()).collect();
        Ok(Tensor::cat(&all, 0)?)
    }

    fn write_sample(&mut self) -> Result<Vec<u32>> {
        let (ctx_h, lang_h) = self.state()?;
        let (ctx_h, lang_h) = (ctx_h.clone(), lang_h.clone());
        let u = self.model.write(&lang_h, &ctx_h, 100, self.temperature, &self.stop, false, &mut self.rng)?;
        if let Some(rl) = &mut self.rl {
            rl.logprobs.extend(u.logprobs);
        }
        self.push_turn(&u.words, u.lang_hs, u.lang_h)?;
        Ok(u.words)
    }

    fn write_rollout(&mut self, ncandidate: usize, nrollout: usize, rollout_len: usize) -> Result<Vec<u32>> {
        let (ctx_h, lang_h) = self.state()?;
        let (ctx_h, lang_h) = (ctx_h.clone(), lang_h.clone());
        let mut best: Option<(f64, crate::model::Utterance)> = None;
        for _ in 0..ncandidate {
            let mv = self.model.write(&lang_h, &ctx_h, 100, self.temperature, &self.stop, false, &mut self.rng)?;
            let is_selection = mv.words == [self.eod];
            let mut words = self.words.clone();
            words.push(self.model.you_token());
            words.extend_from_slice(&mv.words);
            // Greedy choice is deterministic, so a finished dialogue needs one evaluation.
            let n = if is_selection { 1 } else { nrollout };
            let mut score = 0.0;
            for _ in 0..n {
                let (words, hs) = if is_selection {
                    (words.clone(), self.history_with(&[&mv.lang_hs])?)
                } else {
                    let r = self.model.write(&mv.lang_h, &ctx_h, rollout_len, self.temperature, &[self.eod], true, &mut self.rng)?;
                    let mut w = words.clone();
                    w.extend_from_slice(&r.words);
                    (w, self.history_with(&[&mv.lang_hs, &r.lang_hs])?)
                };
                let c = self.choose_from(&words, &hs, false)?;
                score += c.p_agree * self.domain.score(&self.context, &c.choice)? as f64;
            }
            score *= nrollout as f64 / n as f64;
            if best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, mv));
            }
        }
        let (_, mv) = best.ok_or_else(|| anyhow::anyhow!("ncandidate must be positive"))?;
        self.push_turn(&mv.words, mv.lang_hs, mv.lang_h)?;
        Ok(mv.words)
    }

    fn write_batched(&mut self, bsz: usize, count_threshold: usize) -> Result<Vec<u32>> {
        let (ctx_h, lang_h) = self.state()?;
        let (ctx_h, lang_h) = (ctx_h.clone(), lang_h.clone());
        let (outs, lang_hs) = self.model.write_batch(bsz, &lang_h, &ctx_h, self.temperature, 100, self.eod, &mut self.rng)?;

        let mut groups: HashMap<Vec<u32>, (usize, f64, Tensor, Tensor)> = HashMap::new();
        // First-seen order, so ties resolve deterministically like Python's dict.
        let mut order: Vec<Vec<u32>> = Vec::new();
        for (i, out) in outs.iter().enumerate() {
            // Unfinished dialogues cannot be scored.
            let Some(eod_pos) = out.iter().position(|&w| w == self.eod) else { continue };
            let first_len = out.iter().position(|&w| w == self.eos || w == self.eod).unwrap_or(out.len()) + 1;
            let hs = lang_hs.narrow(1, i, 1)?.squeeze(1)?;
            let mut words = self.words.clone();
            words.push(self.model.you_token());
            words.extend_from_slice(&out[..=eod_pos]);
            let dialog_hs = self.history_with(&[&hs.narrow(0, 1, eod_pos + 2)?])?;
            let c = self.choose_from(&words, &dialog_hs, false)?;
            let value = self.domain.score(&self.context, &c.choice)? as f64 * c.p_agree;
            let key = out[..first_len].to_vec();
            if !groups.contains_key(&key) {
                // States after `YOU:` and each word of the first utterance.
                let turn_hs = hs.narrow(0, 1, first_len + 1)?;
                let turn_h = hs.narrow(0, first_len + 1, 1)?;
                groups.insert(key.clone(), (0, 0.0, turn_hs, turn_h));
                order.push(key.clone());
            }
            if let Some(entry) = groups.get_mut(&key) {
                entry.0 += 1;
                entry.1 += value;
            }
        }

        for threshold in (0..=count_threshold).rev() {
            let mut best: Option<(&Vec<u32>, f64)> = None;
            for key in &order {
                let (count, total, _, _) = &groups[key];
                let mean = total / *count as f64;
                if *count >= threshold.max(1) && best.is_none_or(|(_, b)| mean > b) {
                    best = Some((key, mean));
                }
            }
            if let Some((sent, _)) = best {
                let (_, _, hs, h) = &groups[sent];
                let (sent, hs, h) = (sent.clone(), hs.clone(), h.clone());
                self.push_turn(&sent, hs, h)?;
                return Ok(sent);
            }
        }
        // No rollout reached a selection; fall back to plain sampling.
        self.write_sample()
    }

    fn rl_update(&mut self, agree: bool, reward: i64) -> Result<()> {
        let Some(rl) = &mut self.rl else { return Ok(()) };
        let reward = if agree { reward as f64 } else { 0.0 };
        rl.all_rewards.push(reward);
        let n = rl.all_rewards.len() as f64;
        let mean = rl.all_rewards.iter().sum::<f64>() / n;
        let std = (rl.all_rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n).sqrt();
        let r = (reward - mean) / std.max(1e-4);

        let logprobs = std::mem::take(&mut rl.logprobs);
        if logprobs.is_empty() {
            return Ok(());
        }
        // The last action gets the full reward; earlier ones are discounted.
        let k = logprobs.len();
        let weights: Vec<f32> = (0..k).map(|i| (r * rl.cfg.gamma.powi((k - 1 - i) as i32)) as f32).collect();
        let lp = Tensor::stack(&logprobs, 0)?.flatten_all()?;
        let w = Tensor::from_vec(weights, k, lp.device())?;
        let loss = (lp * w)?.sum_all()?.neg()?;
        let grads = loss.backward()?;
        rl.opt.step(&grads, rl.cfg.clip)?;
        Ok(())
    }
}

impl Agent for ModelAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn feed_context(&mut self, ctx: &[String]) -> Result<()> {
        let m = &self.model;
        self.context = ctx.to_vec();
        let ids = m.vocab.context.w2i(ctx)?;
        let ctx_t = Tensor::new(ids.as_slice(), &m.device)?.unsqueeze(1)?;
        self.ctx_h = Some(m.forward_context(&ctx_t)?);
        self.lang_h = Some(m.zero_hid(1, m.cfg.nhid_lang)?);
        self.words.clear();
        self.lang_hs.clear();
        if let Some(rl) = &mut self.rl {
            rl.logprobs.clear();
        }
        Ok(())
    }

    fn read(&mut self, words: &[String]) -> Result<()> {
        let ids = self.model.vocab.word.w2i(words)?;
        let (ctx_h, lang_h) = self.state()?;
        let (hs, h) = self.model.read(&ids, lang_h, ctx_h)?;
        self.words.push(self.model.them_token());
        self.words.extend_from_slice(&ids);
        self.lang_hs.push(hs);
        self.lang_h = Some(h);
        self.check_aligned()
    }

    fn write(&mut self) -> Result<Vec<String>> {
        let words = match self.strategy.clone() {
            Strategy::Sample => self.write_sample()?,
            Strategy::Rollout { ncandidate, nrollout, rollout_len } => {
                self.write_rollout(ncandidate, nrollout, rollout_len)?
            }
            Strategy::BatchedRollout { bsz, count_threshold } => self.write_batched(bsz, count_threshold)?,
        };
        Ok(self.model.vocab.word.i2w(&words))
    }

    fn choose(&mut self) -> Result<Vec<String>> {
        let words = self.words.clone();
        let hs = self.history_with(&[])?;
        let explore = self.rl.as_ref().is_some_and(|rl| self.rng.gen::<f64>() <= rl.cfg.eps);
        let c = self.choose_from(&words, &hs, explore)?;
        if let (Some(rl), Some(lp)) = (&mut self.rl, c.logprob) {
            rl.logprobs.push(lp);
        }
        Ok(c.choice)
    }

    fn update(&mut self, agree: bool, reward: i64) -> Result<()> {
        self.rl_update(agree, reward)
    }
}

/// A person negotiating through stdin.
pub struct HumanAgent {
    name: String,
    domain: ObjectDivision,
    ctx: Vec<String>,
}

impl HumanAgent {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), domain: ObjectDivision, ctx: Vec::new() }
    }

    fn prompt(&self, label: &str) -> Result<String> {
        print!("{label} : ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().lock().read_line(&mut line)? == 0 {
            anyhow::bail!("stdin closed");
        }
        Ok(line)
    }
}

impl Agent for HumanAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_human(&self) -> bool {
        true
    }

    fn feed_context(&mut self, ctx: &[String]) -> Result<()> {
        self.ctx = ctx.to_vec();
        Ok(())
    }

    fn read(&mut self, _words: &[String]) -> Result<()> {
        Ok(())
    }

    fn write(&mut self) -> Result<Vec<String>> {
        let line = self.prompt(&self.name)?;
        if line.trim() == "<selection>" {
            return Ok(vec![line.trim().to_string()]);
        }
        let mut words: Vec<String> = line.to_lowercase().split_whitespace().map(String::from).collect();
        words.push("<eos>".to_string());
        Ok(words)
    }

    fn choose(&mut self) -> Result<Vec<String>> {
        loop {
            let line = self.prompt(&format!("{} choice", self.name))?;
            match self.domain.parse_human_choice(&self.ctx, &line) {
                Ok(choice) => return Ok(choice),
                Err(e) => println!("Your choice is invalid ({e}). Try again."),
            }
        }
    }
}
