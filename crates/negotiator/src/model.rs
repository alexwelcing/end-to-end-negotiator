//! The end-to-end negotiation model: a context encoder, a GRU language model
//! shared between reading and writing, and an attention-based selection head.

use crate::data::{Dictionary, Vocab};
use crate::domain::{is_item_token, ObjectDivision};
use crate::nn::{dropout, embed, Gru, Linear, Params};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub nembed_word: usize,
    pub nembed_ctx: usize,
    pub nhid_lang: usize,
    pub nhid_ctx: usize,
    pub nhid_attn: usize,
    pub nhid_sel: usize,
    pub dropout: f32,
    pub init_range: f64,
    pub output_length: usize,
    /// Selection-loss weight the model was trained with; reused by RL's supervised steps.
    #[serde(default = "default_sel_weight")]
    pub sel_weight: f64,
}

fn default_sel_weight() -> f64 {
    1.0
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    config: ModelConfig,
    word_dict: Dictionary,
    item_dict: Dictionary,
    context_dict: Dictionary,
}

/// Result of generating one utterance.
pub struct Utterance {
    pub words: Vec<u32>,
    /// Log-probability of each sampled word; differentiable for RL.
    pub logprobs: Vec<Tensor>,
    /// Hidden state after reading the final word, `(1, H)`.
    pub lang_h: Tensor,
    /// One hidden state per token fed to the writer, `(n, H)`.
    pub lang_hs: Tensor,
}

pub struct DialogModel {
    pub cfg: ModelConfig,
    pub vocab: Vocab,
    pub device: Device,
    pub params: Params,
    word_emb: candle_core::Var,
    ctx_cnt: candle_core::Var,
    ctx_val: candle_core::Var,
    ctx_lin: Linear,
    reader: Gru,
    decoder: Linear,
    sel_fwd: Gru,
    sel_bwd: Gru,
    sel_enc: Linear,
    attn_hid: Linear,
    attn_out: Linear,
    sel_decoders: Vec<Linear>,
    /// `-999` for tokens the writer may not produce mid-dialogue, else `0`.
    special_mask: Tensor,
    you: u32,
    them: u32,
}

impl DialogModel {
    pub fn new(cfg: ModelConfig, vocab: Vocab, device: &Device) -> Result<Self> {
        let dev = device;
        let r = cfg.init_range;
        let domain = ObjectDivision;
        let mut p = Params::default();
        let nwords = vocab.word.len();

        let word_emb = p.uniform("word_encoder.weight", &[nwords, cfg.nembed_word], r, dev)?;
        let ctx_cnt = p.uniform("ctx_encoder.cnt_enc.weight", &[vocab.context.len(), cfg.nembed_ctx], r, dev)?;
        let ctx_val = p.uniform("ctx_encoder.val_enc.weight", &[vocab.context.len(), cfg.nembed_ctx], r, dev)?;
        let ctx_in = domain.input_length() * cfg.nembed_ctx;
        let ctx_lin = Linear::new(&mut p, "ctx_encoder.encoder", ctx_in, cfg.nhid_ctx, r, dev)?;
        let lang_in = cfg.nembed_word + cfg.nhid_ctx;
        let reader = Gru::new(&mut p, "reader", lang_in, cfg.nhid_lang, r, true, dev)?;
        let decoder = Linear::new(&mut p, "decoder", cfg.nhid_lang, cfg.nembed_word, r, dev)?;
        // The selection GRU keeps PyTorch's default init, as in the reference code.
        let sel_in = cfg.nhid_lang + cfg.nembed_word;
        let sel_r = 1.0 / (cfg.nhid_attn as f64).sqrt();
        let sel_fwd = Gru::new(&mut p, "sel_rnn.fwd", sel_in, cfg.nhid_attn, sel_r, false, dev)?;
        let sel_bwd = Gru::new(&mut p, "sel_rnn.bwd", sel_in, cfg.nhid_attn, sel_r, false, dev)?;
        let sel_enc = Linear::new(&mut p, "sel_encoder", 2 * cfg.nhid_attn + cfg.nhid_ctx, cfg.nhid_sel, r, dev)?;
        let attn_hid = Linear::new(&mut p, "attn.0", 2 * cfg.nhid_attn, cfg.nhid_attn, r, dev)?;
        let attn_out = Linear::new(&mut p, "attn.2", cfg.nhid_attn, 1, r, dev)?;
        let sel_decoders = (0..cfg.output_length)
            .map(|i| Linear::new(&mut p, &format!("sel_decoders.{i}"), cfg.nhid_sel, vocab.item.len(), r, dev))
            .collect::<Result<Vec<_>>>()?;

        let mask: Vec<f32> = (0..nwords)
            .map(|i| {
                let w = vocab.word.word(i);
                let special = is_item_token(w) || matches!(w, "<unk>" | "YOU:" | "THEM:" | "<pad>");
                if special { -999.0 } else { 0.0 }
            })
            .collect();
        let special_mask = Tensor::from_vec(mask, nwords, dev)?;
        let you = vocab.word.idx("YOU:")? as u32;
        let them = vocab.word.idx("THEM:")? as u32;

        Ok(Self {
            cfg,
            vocab,
            device: dev.clone(),
            params: p,
            word_emb,
            ctx_cnt,
            ctx_val,
            ctx_lin,
            reader,
            decoder,
            sel_fwd,
            sel_bwd,
            sel_enc,
            attn_hid,
            attn_out,
            sel_decoders,
            special_mask,
            you,
            them,
        })
    }

    fn paths(path: &Path) -> (PathBuf, PathBuf) {
        (path.with_extension("safetensors"), path.with_extension("json"))
    }

    /// Writes `<path>.safetensors` (weights) and `<path>.json` (config + vocab).
    pub fn save(&self, path: &Path) -> Result<()> {
        let (weights, meta) = Self::paths(path);
        let ckpt = Checkpoint {
            config: self.cfg.clone(),
            word_dict: self.vocab.word.clone(),
            item_dict: self.vocab.item.clone(),
            context_dict: self.vocab.context.clone(),
        };
        std::fs::write(&meta, serde_json::to_string_pretty(&ckpt)?)?;
        self.params.save(&weights)
    }

    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let (weights, meta) = Self::paths(path);
        let text = std::fs::read_to_string(&meta).with_context(|| format!("reading {}", meta.display()))?;
        let ckpt: Checkpoint = serde_json::from_str(&text)?;
        let vocab = Vocab {
            word: ckpt.word_dict.reindex(),
            item: ckpt.item_dict.reindex(),
            context: ckpt.context_dict.reindex(),
        };
        let model = Self::new(ckpt.config, vocab, device)?;
        model.params.load(&weights, device)?;
        Ok(model)
    }

    pub fn word_idx(&self, word: &str) -> Result<u32> {
        Ok(self.vocab.word.idx(word)? as u32)
    }

    pub fn you_token(&self) -> u32 {
        self.you
    }

    pub fn them_token(&self) -> u32 {
        self.them
    }

    pub fn ids(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(Tensor::new(ids, &self.device)?)
    }

    pub fn zero_hid(&self, bsz: usize, nhid: usize) -> Result<Tensor> {
        Ok(Tensor::zeros((bsz, nhid), DType::F32, &self.device)?)
    }

    /// Encodes `(2 * n_items, B)` context tokens into `(B, nhid_ctx)`.
    pub fn forward_context(&self, ctx: &Tensor) -> Result<Tensor> {
        let (n, b) = ctx.dims2()?;
        let cnt_idx: Vec<u32> = (0..n as u32).step_by(2).collect();
        let val_idx: Vec<u32> = (1..n as u32).step_by(2).collect();
        let cnt = embed(&self.ctx_cnt, &ctx.index_select(&self.ids(&cnt_idx)?, 0)?)?;
        let val = embed(&self.ctx_val, &ctx.index_select(&self.ids(&val_idx)?, 0)?)?;
        let h = (cnt * val)?.transpose(0, 1)?.contiguous()?.reshape((b, ()))?;
        self.ctx_lin.forward(&h.tanh()?)
    }

    /// Tied output projection: decoder then the transposed word embeddings.
    fn word_scores(&self, lang_h: &Tensor) -> Result<Tensor> {
        Ok(self.decoder.forward(lang_h)?.matmul(&self.word_emb.t()?)?)
    }

    fn writer_input(&self, words: &Tensor, ctx_h: &Tensor) -> Result<Tensor> {
        Ok(Tensor::cat(&[&embed(&self.word_emb, words)?, ctx_h], 1)?)
    }

    /// Teacher-forced LM over `inpt: (T, B)`. Returns logits `(T * B, V)` and
    /// hidden states `(T, B, H)`.
    pub fn forward_lm(&self, inpt: &Tensor, lang_h: &Tensor, ctx_h: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let (t, b) = inpt.dims2()?;
        let emb = embed(&self.word_emb, inpt)?;
        let ctx_rep = ctx_h.unsqueeze(0)?.broadcast_as((t, b, self.cfg.nhid_ctx))?;
        let x = dropout(&Tensor::cat(&[&emb, &ctx_rep.contiguous()?], 2)?, self.cfg.dropout, train)?;
        let (out, _) = self.reader.forward(&x, lang_h, false)?;
        let logits = self.word_scores(&out.reshape((t * b, self.cfg.nhid_lang))?)?;
        Ok((logits, out))
    }

    /// Selection logits `(output_length * B, n_items)` from words `(T, B)` and
    /// their LM hidden states `(T, B, H)`.
    pub fn forward_selection(&self, inpt: &Tensor, lang_hs: &Tensor, ctx_h: &Tensor, train: bool) -> Result<Tensor> {
        let (t, b) = inpt.dims2()?;
        let a = self.cfg.nhid_attn;
        let h = Tensor::cat(&[lang_hs, &embed(&self.word_emb, inpt)?], 2)?;
        let h = dropout(&h, self.cfg.dropout, train)?;
        let h0 = self.zero_hid(b, a)?;
        let (fwd, _) = self.sel_fwd.forward(&h, &h0, false)?;
        let (bwd, _) = self.sel_bwd.forward(&h, &h0, true)?;
        let h = Tensor::cat(&[fwd, bwd], 2)?.transpose(0, 1)?.contiguous()?; // (B, T, 2A)
        let flat = h.reshape((b * t, 2 * a))?;
        let logit = self.attn_out.forward(&self.attn_hid.forward(&flat)?.tanh()?)?.reshape((b, t))?;
        let prob = candle_nn::ops::softmax(&logit, 1)?.unsqueeze(2)?;
        let attn = h.broadcast_mul(&prob)?.sum(1)?; // (B, 2A)
        let h = dropout(&Tensor::cat(&[&attn, ctx_h], 1)?, self.cfg.dropout, train)?;
        let h = self.sel_enc.forward(&h)?.tanh()?;
        let outs = self.sel_decoders.iter().map(|d| d.forward(&h)).collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(&outs, 0)?)
    }

    /// Per-slot selection logits `(output_length, n_items)` for one dialogue.
    pub fn choice_logits(&self, words: &[u32], lang_hs: &Tensor, ctx_h: &Tensor) -> Result<Tensor> {
        ensure!(words.len() == lang_hs.dim(0)?, "{} words vs {} states", words.len(), lang_hs.dim(0)?);
        let inpt = self.ids(words)?.unsqueeze(1)?;
        self.forward_selection(&inpt, &lang_hs.unsqueeze(1)?, ctx_h, false)
    }

    /// Reads the partner's words, prefixed with `THEM:`. Returns one state per
    /// token read `(n + 1, H)` and the final state `(1, H)`.
    pub fn read(&self, words: &[u32], lang_h: &Tensor, ctx_h: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut ids = Vec::with_capacity(words.len() + 1);
        ids.push(self.them);
        ids.extend_from_slice(words);
        let x = self.writer_input(&self.ids(&ids)?, &ctx_h.broadcast_as((ids.len(), self.cfg.nhid_ctx))?.contiguous()?)?;
        let (out, h) = self.reader.forward(&x.unsqueeze(1)?, lang_h, false)?;
        Ok((out.squeeze(1)?, h))
    }

    /// Samples an utterance word by word. When `resume` is false, the writer
    /// first reads `YOU:` and special tokens are masked out.
    #[allow(clippy::too_many_arguments)]
    pub fn write<R: Rng>(
        &self,
        lang_h: &Tensor,
        ctx_h: &Tensor,
        max_words: usize,
        temperature: f64,
        stop_tokens: &[u32],
        resume: bool,
        rng: &mut R,
    ) -> Result<Utterance> {
        let mut lang_h = lang_h.clone();
        let mut lang_hs = Vec::new();
        let mut words = Vec::new();
        let mut logprobs = Vec::new();
        let mut inpt = (!resume).then_some(self.you);

        for _ in 0..max_words {
            if let Some(w) = inpt {
                lang_h = self.reader.step(&self.writer_input(&self.ids(&[w])?, ctx_h)?, &lang_h)?;
                lang_hs.push(lang_h.clone());
            }
            let mut scores = (self.word_scores(&lang_h)?.squeeze(0)? / temperature)?;
            if !resume {
                scores = (scores + &self.special_mask)?;
            }
            let logprob = candle_nn::ops::log_softmax(&scores, D::Minus1)?;
            let w = sample(&logprob, rng)?;
            logprobs.push(logprob.get(w as usize)?);
            words.push(w);
            inpt = Some(w);
            if stop_tokens.contains(&w) {
                break;
            }
        }
        if let Some(w) = inpt {
            lang_h = self.reader.step(&self.writer_input(&self.ids(&[w])?, ctx_h)?, &lang_h)?;
            lang_hs.push(lang_h.clone());
        }
        Ok(Utterance { words, logprobs, lang_h, lang_hs: Tensor::cat(&lang_hs, 0)? })
    }

    /// Samples `bsz` continuations in parallel until each emits `eod` (or
    /// `max_words`). Returns per-sample tokens and states `(steps + 2, bsz, H)`,
    /// where index 0 is the starting state and index `k + 1` follows input `k`
    /// of `[YOU:, w0, w1, ...]`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_batch<R: Rng>(
        &self,
        bsz: usize,
        lang_h: &Tensor,
        ctx_h: &Tensor,
        temperature: f64,
        max_words: usize,
        eod: u32,
        rng: &mut R,
    ) -> Result<(Vec<Vec<u32>>, Tensor)> {
        let ctx_h = ctx_h.broadcast_as((bsz, self.cfg.nhid_ctx))?.contiguous()?;
        let mut lang_h = lang_h.detach().broadcast_as((bsz, self.cfg.nhid_lang))?.contiguous()?;
        let mut lang_hs = vec![lang_h.clone()];
        let mut outs = vec![Vec::new(); bsz];
        let mut inpt = vec![self.you; bsz];
        let mut done = vec![false; bsz];

        for _ in 0..max_words {
            lang_h = self.reader.step(&self.writer_input(&self.ids(&inpt)?, &ctx_h)?, &lang_h)?.detach();
            lang_hs.push(lang_h.clone());
            let scores = (self.word_scores(&lang_h)? / temperature)?;
            let logprobs = candle_nn::ops::log_softmax(&scores, D::Minus1)?.to_vec2::<f32>()?;
            for (i, lp) in logprobs.iter().enumerate() {
                let w = sample_host(lp, rng)?;
                outs[i].push(w);
                inpt[i] = w;
                done[i] |= w == eod;
            }
            if done.iter().all(|&d| d) {
                break;
            }
        }
        lang_h = self.reader.step(&self.writer_input(&self.ids(&inpt)?, &ctx_h)?, &lang_h)?.detach();
        lang_hs.push(lang_h);
        Ok((outs, Tensor::stack(&lang_hs, 0)?))
    }
}

/// Draws one index from a log-probability vector on the device.
fn sample<R: Rng>(logprob: &Tensor, rng: &mut R) -> Result<u32> {
    sample_host(&logprob.to_vec1::<f32>()?, rng)
}

fn sample_host<R: Rng>(logprob: &[f32], rng: &mut R) -> Result<u32> {
    let probs = logprob.iter().map(|lp| lp.exp().max(0.0));
    let dist = WeightedIndex::new(probs).context("degenerate distribution")?;
    Ok(dist.sample(rng) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn tiny_model() -> Result<DialogModel> {
        let mut word = Dictionary::new(true);
        for w in ["YOU:", "THEM:", "hi", "deal", "item0=1"] {
            word.add_word(w);
        }
        let mut item = Dictionary::new(false);
        for w in ["item0=0", "item0=1", "<no_agreement>"] {
            item.add_word(w);
        }
        let mut context = Dictionary::new(false);
        for w in ["0", "1", "2"] {
            context.add_word(w);
        }
        let cfg = ModelConfig {
            nembed_word: 8,
            nembed_ctx: 4,
            nhid_lang: 6,
            nhid_ctx: 5,
            nhid_attn: 3,
            nhid_sel: 7,
            dropout: 0.5,
            init_range: 0.1,
            output_length: 6,
            sel_weight: 1.0,
        };
        DialogModel::new(cfg, Vocab { word, item, context }, &Device::Cpu)
    }

    #[test]
    fn shapes_line_up_through_training_forward() -> Result<()> {
        let m = tiny_model()?;
        let ctx = Tensor::new(&[[1u32, 0], [2, 1], [0, 0], [1, 2], [2, 2], [0, 1]], &Device::Cpu)?;
        let ctx_h = m.forward_context(&ctx)?;
        assert_eq!(ctx_h.dims(), &[2, 5]);
        let inpt = Tensor::new(&[[4u32, 5], [6, 7], [0, 0]], &Device::Cpu)?;
        let (logits, hs) = m.forward_lm(&inpt, &m.zero_hid(2, 6)?, &ctx_h, true)?;
        assert_eq!(logits.dims(), &[6, m.vocab.word.len()]);
        let sel = m.forward_selection(&inpt, &hs, &ctx_h, true)?;
        assert_eq!(sel.dims(), &[12, 3]);
        Ok(())
    }

    #[test]
    fn write_never_emits_masked_tokens_and_tracks_states() -> Result<()> {
        let m = tiny_model()?;
        let ctx = Tensor::new(&[[1u32], [2], [0], [1], [2], [0]], &Device::Cpu)?;
        let ctx_h = m.forward_context(&ctx)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let stop = [m.word_idx("<eos>")?, m.word_idx("<selection>")?];
        let masked = ["YOU:", "THEM:", "item0=1", "<pad>", "<unk>"].map(|w| m.word_idx(w).unwrap());
        for _ in 0..20 {
            let u = m.write(&m.zero_hid(1, 6)?, &ctx_h, 10, 1.0, &stop, false, &mut rng)?;
            assert!(u.words.iter().all(|w| !masked.contains(w)));
            assert_eq!(u.lang_hs.dim(0)?, u.words.len() + 1);
            assert_eq!(u.logprobs.len(), u.words.len());
        }
        let (outs, hs) = m.write_batch(4, &m.zero_hid(1, 6)?, &ctx_h, 1.0, 5, stop[1], &mut rng)?;
        assert_eq!(hs.dims(), &[outs[0].len() + 2, 4, 6]);
        Ok(())
    }

    #[test]
    fn save_load_round_trip() -> Result<()> {
        let m = tiny_model()?;
        let dir = std::env::temp_dir().join(format!("negotiator-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("m");
        m.save(&path)?;
        let m2 = DialogModel::load(&path, &Device::Cpu)?;
        let a = m.params.snapshot()?;
        let b = m2.params.snapshot()?;
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.flatten_all()?.to_vec1::<f32>()?, y.flatten_all()?.to_vec1::<f32>()?);
        }
        assert_eq!(m2.vocab.word.idx("deal")?, m.vocab.word.idx("deal")?);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
