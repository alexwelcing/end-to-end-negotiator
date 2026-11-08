//! Train the JEPA energy for a finite spec (port of `nerve/train/loop.py`).
//!
//! The training set is a pure function of `(spec, config)`: every valid
//! `(state, action)` pair the oracle can answer, `negatives` spec-wrong
//! successors per pair, and the `learn` holes' examples oversampled. Pairs the
//! hash mask holds out are never trained on and are scored separately.

use crate::encode::Encoder;
use crate::jepa::{shift_indices, FactInputs, Jepa, JepaConfig};
use crate::oracle::{effective_holdout, holdout_mask, Oracle};
use crate::serve::LearnedEnergy;
use crate::spec::{ActionInst, Spec, State};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as J};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    pub d: usize,
    pub hidden: usize,
    pub ema: f64,
    pub tau: f64,
    pub holdout: f64,
    pub seed: u64,
    /// Oversampling of hole examples (there are very few).
    pub hole_weight: usize,
    /// Spec-wrong successors sampled per positive.
    pub negatives: usize,
    pub weight_decay: f64,
    pub ord_weight: f64,
    pub fact_weight: f64,
    pub holdout_salt: String,
    pub log_every: usize,
    /// Continuous specs: sampled (state, action) pairs to train on.
    pub cont_pairs: usize,
    /// Continuous specs: near-miss perturbations contrasted with each true successor.
    pub cont_negatives: usize,
    /// Continuous specs: sampled pairs in the final evaluation.
    pub eval_samples: usize,
    /// Finite specs: which states the model may answer with.
    pub candidates: Candidates,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 60,
            batch: 128,
            lr: 3e-3,
            d: 64,
            hidden: 128,
            ema: 0.99,
            tau: 0.1,
            holdout: 0.15,
            seed: 0,
            hole_weight: 20,
            negatives: 3,
            weight_decay: 1e-4,
            ord_weight: 1.0,
            fact_weight: 10.0,
            holdout_salt: "nerve".into(),
            log_every: 10,
            cont_pairs: 4000,
            cont_negatives: 16,
            eval_samples: 1000,
            candidates: Candidates::Valid,
        }
    }
}

/// Which states a finite model may answer with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Candidates {
    /// Every enumerable state, including ones that break an invariant (the Python reference).
    All,
    /// Only the states the spec's invariants allow, so no answer can break one.
    Valid,
}

/// A trained network bound to its spec: the argmin over its candidate states is its step.
pub struct Model {
    pub spec: Spec,
    pub enc: Encoder,
    pub jepa: Jepa,
    pub candidates: Candidates,
    /// Encoder state ids of the candidates, in row order of `cand`.
    rows: Vec<usize>,
    /// Encoder state id -> candidate row, `None` for a state that is not a candidate.
    row_of: Vec<Option<usize>>,
    cand: Tensor,
    cand_idx: Tensor,
    /// `(state, action) -> energies of every candidate`, so drag frames do not re-run the network.
    cache: Mutex<HashMap<String, Vec<f32>>>,
    /// `model.json` of a loaded checkpoint (how it was trained and scored).
    meta: Option<J>,
}

impl Model {
    pub fn new(spec: Spec, cfg: JepaConfig, candidates: Candidates, dev: &Device) -> Result<Self> {
        ensure!(spec.finite(), "{}: only finite specs train in Rust so far", spec.name);
        let enc = Encoder::new(&spec)?;
        let jepa = Jepa::new(cfg, &enc.field_masks(), dev)?;
        let oracle = Oracle::new(&spec);
        let rows: Vec<usize> = (0..enc.states.len())
            .filter(|&i| candidates == Candidates::All || !oracle.breaks_invariant(&enc.states[i]))
            .collect();
        ensure!(!rows.is_empty(), "{}: no state satisfies the invariants", spec.name);
        let mut row_of = vec![None; enc.states.len()];
        let (mut vecs, mut idxs) = (Vec::new(), Vec::new());
        for (r, &i) in rows.iter().enumerate() {
            row_of[i] = Some(r);
            vecs.extend(enc.state(&spec, &enc.states[i])?);
            idxs.extend(enc.field_index(&spec, &enc.states[i])?);
        }
        let cand = Tensor::from_vec(vecs, (rows.len(), enc.state_dim), dev)?;
        let cand_idx = Tensor::from_vec(idxs, (rows.len(), enc.slots.len()), dev)?;
        Ok(Self { spec, enc, jepa, candidates, rows, row_of, cand, cand_idx, cache: Mutex::new(HashMap::new()), meta: None })
    }

    pub fn candidate_count(&self) -> usize {
        self.rows.len()
    }

    /// The candidate row of `s`, or `None` if it is not a state this model may answer with.
    fn row(&self, s: &State) -> Result<Option<usize>> {
        Ok(self.row_of[self.enc.state_id(&self.spec, s)?])
    }

    /// `model.json` of a loaded checkpoint.
    pub fn meta(&self) -> Option<&J> {
        self.meta.as_ref()
    }

    pub fn config_for(spec: &Spec, t: &TrainConfig) -> Result<JepaConfig> {
        let enc = Encoder::new(spec)?;
        let mut cfg = JepaConfig::new(enc.state_dim, enc.action_dim());
        cfg.d = t.d;
        cfg.hidden = t.hidden;
        cfg.ema = t.ema;
        cfg.tau = t.tau;
        cfg.action_cont_dim = enc.cont_params.len();
        cfg.ord_dim = enc.ord_slots.len();
        cfg.ord_weight = t.ord_weight;
        cfg.fields = enc.slots.iter().map(|s| (s.offset, s.n, s.ordinal)).collect();
        cfg.fact_weight = t.fact_weight;
        Ok(cfg)
    }

    fn device(&self) -> &Device {
        &self.jepa.device
    }

    /// Field indices `[B, F]` and the factored head's shift indices for a batch of states.
    fn fact_inputs(&self, states: &[&State]) -> Result<(Tensor, Vec<Option<Tensor>>)> {
        let mut cur = Vec::with_capacity(states.len() * self.enc.slots.len());
        for s in states {
            cur.extend(self.enc.field_index(&self.spec, s)?);
        }
        let shift = shift_indices(&self.jepa.cfg.fields, &cur, states.len(), self.device())?;
        Ok((Tensor::from_vec(cur, (states.len(), self.enc.slots.len()), self.device())?, shift))
    }

    /// Energies of every candidate state as the successor of `s -a->`, in candidate row order.
    pub fn energies(&self, s: &State, a: &ActionInst) -> Result<Vec<f32>> {
        let key = format!("{}|{}", self.spec.state_key(s), self.spec.action_key(a));
        if let Some(e) = self.cache.lock().expect("energy cache poisoned").get(&key) {
            return Ok(e.clone());
        }
        let dev = self.device();
        let x = Tensor::from_vec(self.enc.state(&self.spec, s)?, (1, self.enc.state_dim), dev)?;
        let ai = Tensor::new(&[self.enc.action_id(&self.spec, a)? as u32], dev)?;
        let ac = if self.enc.cont_params.is_empty() {
            None
        } else {
            Some(Tensor::from_vec(self.enc.action_cont(&self.spec, a), (1, self.enc.cont_params.len()), dev)?)
        };
        let (cur, shift) = self.fact_inputs(&[s])?;
        let fi = FactInputs { cur_idx: &cur, shift_idx: &shift };
        let e: Vec<f32> = self.jepa.energy_all(&x, &ai, &self.cand, ac.as_ref(), Some((&fi, &self.cand_idx)))?.squeeze(0)?.to_vec1()?;
        self.cache.lock().expect("energy cache poisoned").insert(key, e.clone());
        Ok(e)
    }

    /// The network's successor: argmin energy over its candidate states.
    pub fn step(&self, s: &State, a: &ActionInst) -> Result<State> {
        let e = self.energies(s, a)?;
        let best = (0..e.len()).min_by(|&i, &j| e[i].total_cmp(&e[j])).context("empty state space")?;
        Ok(self.enc.states[self.rows[best]].clone())
    }

    pub fn save(&self, dir: &Path, t: &TrainConfig, report: &J) -> Result<()> {
        let meta = json!({
            "spec": self.spec.name,
            "kind": "finite",
            "candidates": self.candidates,
            "jepa": self.jepa.cfg,
            "train": t,
            "eval": report,
        });
        save_checkpoint(dir, &self.jepa, meta)
    }

    /// Loads `dir/model.{json,safetensors}` for `spec` onto `dev`, with the candidates it was trained with.
    pub fn load(spec: Spec, dir: &Path, dev: &Device) -> Result<Self> {
        Self::load_as(spec, dir, dev, None)
    }

    /// Like `load`, optionally answering from a different candidate set (to rescore a checkpoint).
    /// A checkpoint from before the choice existed scored every state.
    pub fn load_as(spec: Spec, dir: &Path, dev: &Device, candidates: Option<Candidates>) -> Result<Self> {
        let meta = read_meta(dir, &spec.name)?;
        let cfg: JepaConfig = serde_json::from_value(meta["jepa"].clone())?;
        let saved = match meta.get("candidates") {
            Some(c) => serde_json::from_value(c.clone())?,
            None => Candidates::All,
        };
        let mut m = Self::new(spec, cfg, candidates.unwrap_or(saved), dev)?;
        load_weights(&m.jepa, dir, dev)?;
        m.meta = Some(meta);
        Ok(m)
    }
}

/// What a checkpoint says about its own training, for people looking at the served model:
/// how it was trained and how it scored on transitions it never saw.
pub(crate) fn trained_facts(meta: &J) -> J {
    let (e, t) = (&meta["eval"], &meta["train"]);
    json!({
        "continuous": e["continuous"].as_bool().unwrap_or(false),
        "epochs": t["epochs"],
        "holdout": t["holdout"],
        "pairs": e["train_pairs"],
        "trained_on": e["trained_on"],
        "seconds": e["seconds"],
        "heldout": e["heldout"],
        "all": e["all"],
        "hole_examples": e["hole_examples"],
    })
}

/// Writes `dir/model.safetensors` (every named parameter) and `dir/model.json` (`meta`).
pub(crate) fn save_checkpoint(dir: &Path, jepa: &Jepa, meta: J) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let tensors: HashMap<String, Tensor> = jepa.store.vars.iter().map(|(n, v, _)| (n.clone(), v.as_tensor().clone())).collect();
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors"))?;
    std::fs::write(dir.join("model.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

/// Reads `dir/model.json`, refusing a checkpoint trained for another spec.
pub(crate) fn read_meta(dir: &Path, spec: &str) -> Result<J> {
    let path = dir.join("model.json");
    let meta: J = serde_json::from_str(&std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?)?;
    ensure!(meta["spec"] == spec, "{} holds a model for {}, not {spec}", dir.display(), meta["spec"]);
    Ok(meta)
}

pub(crate) fn load_weights(jepa: &Jepa, dir: &Path, dev: &Device) -> Result<()> {
    let weights = candle_core::safetensors::load(dir.join("model.safetensors"), dev)?;
    for (name, var, _) in &jepa.store.vars {
        let w = weights.get(name).with_context(|| format!("checkpoint is missing {name}"))?;
        var.set(w)?;
    }
    Ok(())
}

/// Loads whichever kind of model `dir` holds for `spec`, ready to serve.
pub fn load_learned(spec: Spec, dir: &Path, dev: &Device) -> Result<std::sync::Arc<dyn LearnedEnergy>> {
    let meta = read_meta(dir, &spec.name)?;
    Ok(if meta["kind"] == "continuous" {
        std::sync::Arc::new(crate::continuous::ContinuousModel::load(spec, dir, dev)?)
    } else {
        std::sync::Arc::new(Model::load(spec, dir, dev)?)
    })
}

impl LearnedEnergy for Model {
    /// How far a candidate is from the network's own best guess for this action; staying put costs nothing.
    fn energy(&self, start: &State, action: Option<&ActionInst>, cand: &State) -> Result<f64> {
        let Some(a) = action else { return Ok(0.0) };
        // A state this model may not answer with costs infinite energy.
        let Some(r) = self.row(cand)? else { return Ok(f64::INFINITY) };
        let e = self.energies(start, a)?;
        let min = e.iter().copied().fold(f32::INFINITY, f32::min);
        Ok(f64::from(e[r] - min))
    }

    fn step_energies(&self, start: &State, action: &ActionInst) -> Result<Option<Vec<(State, f64)>>> {
        let e = self.energies(start, action)?;
        Ok(Some(self.rows.iter().map(|&i| self.enc.states[i].clone()).zip(e.into_iter().map(f64::from)).collect()))
    }

    fn describe(&self) -> J {
        let params = self.jepa.store.count_trainable();
        let device = if self.device().is_cuda() { "cuda" } else { "cpu" };
        json!({
            "kind": "jepa",
            "params": params,
            "device": device,
            "summary": format!("{} JEPA, {params} params on {device}", self.spec.name),
            "candidates": self.candidates,
            "candidate_count": self.rows.len(),
            "trained": self.meta.as_ref().map(trained_facts),
        })
    }
}

/// One training pair (positive) with its sampled spec-wrong successors.
struct Pair {
    s: State,
    a: ActionInst,
    next: State,
}

struct Dataset {
    pos: Vec<Pair>,
    neg: Vec<Pair>,
}

fn hole_examples(spec: &Spec) -> Vec<(State, ActionInst, State)> {
    spec.holes
        .iter()
        .flat_map(|h| h.examples.iter().map(|e| (e.state.clone(), ActionInst { name: h.action.clone(), params: e.params.clone() }, e.next.clone())))
        .collect()
}

/// `(state, action)` pairs the oracle is the truth for: valid states, non-hole actions.
fn oracle_pairs(spec: &Spec) -> Result<Vec<(State, ActionInst)>> {
    let oracle = Oracle::new(spec);
    let holes = spec.hole_actions();
    let actions: Vec<ActionInst> = spec.action_instances()?.into_iter().filter(|a| !holes.contains(&a.name.as_str())).collect();
    let mut out = Vec::new();
    for s in spec.state_space()? {
        if oracle.breaks_invariant(&s) {
            continue;
        }
        out.extend(actions.iter().map(|a| (s.clone(), a.clone())));
    }
    Ok(out)
}

fn dataset<R: Rng>(spec: &Spec, cfg: &TrainConfig, rng: &mut R) -> Result<Dataset> {
    let oracle = Oracle::new(spec);
    let frac = effective_holdout(spec, cfg.holdout)?;
    let space = spec.state_space()?;
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    for (s, a) in oracle_pairs(spec)? {
        if holdout_mask(spec, &s, &a, frac, &cfg.holdout_salt) {
            continue;
        }
        let next = oracle.step(&s, &a)?;
        for _ in 0..cfg.negatives {
            if let Some(c) = space.choose(rng).filter(|c| **c != next) {
                neg.push(Pair { s: s.clone(), a: a.clone(), next: c.clone() });
            }
        }
        pos.push(Pair { s, a, next });
    }
    for (s, a, next) in hole_examples(spec) {
        if holdout_mask(spec, &s, &a, frac, &cfg.holdout_salt) {
            continue;
        }
        for _ in 0..cfg.hole_weight {
            pos.push(Pair { s: s.clone(), a: a.clone(), next: next.clone() });
        }
    }
    Ok(Dataset { pos, neg })
}

/// Exact-match and violation rates over a set of pairs, against the oracle.
fn conformance(model: &Model, pairs: &[(State, ActionInst)]) -> Result<J> {
    let oracle = Oracle::new(&model.spec);
    let (mut exact, mut viol) = (0usize, 0usize);
    for (s, a) in pairs {
        let pred = model.step(s, a)?;
        exact += usize::from(pred == oracle.step(s, a)?);
        viol += usize::from(!oracle.violations(s, Some(a), &pred, 0.01).is_empty());
    }
    let n = pairs.len().max(1) as f64;
    Ok(json!({"n": pairs.len(), "exact": exact as f64 / n, "violation_rate": viol as f64 / n}))
}

/// Same report and score as `nerve/train/eval.py` (the Rust model has no compiled circuit,
/// so `learned_only` is the network itself and equals `all`).
pub fn evaluate(model: &Model, holdout: f64, salt: &str) -> Result<J> {
    let spec = &model.spec;
    let frac = effective_holdout(spec, holdout)?;
    let all = oracle_pairs(spec)?;
    let (held, seen): (Vec<_>, Vec<_>) = all.iter().cloned().partition(|(s, a)| holdout_mask(spec, s, a, frac, salt));
    let all_r = conformance(model, &all)?;
    let held_r = if held.is_empty() { all_r.clone() } else { conformance(model, &held)? };
    let mut out = json!({"all": all_r, "seen": conformance(model, &seen)?, "heldout": held_r, "learned_only": all_r});
    let f = |k: &str, m: &str| out[k][m].as_f64().unwrap_or(0.0);
    let mut score = 0.4 * f("heldout", "exact") + 0.2 * f("all", "exact") + 0.2 * (1.0 - f("all", "violation_rate")) + 0.2 * f("learned_only", "exact");
    let ex = hole_examples(spec);
    if !ex.is_empty() {
        let mut hit = 0;
        for (s, a, n) in &ex {
            hit += usize::from(model.step(s, a)? == *n);
        }
        let rate = hit as f64 / ex.len() as f64;
        out["hole_examples"] = json!({"n": ex.len(), "exact": rate});
        score = 0.8 * score + 0.2 * rate;
    }
    out["score"] = json!((score * 1e4).round() / 1e4);
    Ok(out)
}

/// Builds the batch tensors for `idx` into `pairs`.
fn batch_tensors(m: &Model, pairs: &[Pair], idx: &[usize]) -> Result<(Tensor, Tensor, Tensor)> {
    let dev = m.device();
    let mut s = Vec::with_capacity(idx.len() * m.enc.state_dim);
    let mut nx = Vec::with_capacity(idx.len() * m.enc.state_dim);
    let mut a = Vec::with_capacity(idx.len());
    for &i in idx {
        s.extend(m.enc.state(&m.spec, &pairs[i].s)?);
        nx.extend(m.enc.state(&m.spec, &pairs[i].next)?);
        a.push(m.enc.action_id(&m.spec, &pairs[i].a)? as u32);
    }
    let d = m.enc.state_dim;
    Ok((Tensor::from_vec(s, (idx.len(), d), dev)?, Tensor::from_vec(a, idx.len(), dev)?, Tensor::from_vec(nx, (idx.len(), d), dev)?))
}

fn train_batch(m: &Model, data: &Dataset, idx: &[usize], rng: &mut impl Rng) -> Result<Tensor> {
    let dev = m.device().clone();
    let (s, a, _) = batch_tensors(m, &data.pos, idx)?;
    let pos: Vec<u32> = idx
        .iter()
        .map(|&i| {
            let next = &data.pos[i].next;
            let r = m.row(next)?.with_context(|| format!("successor {} breaks an invariant, so it is not a candidate", m.spec.state_key(next)))?;
            Ok(r as u32)
        })
        .collect::<Result<_>>()?;
    let pos = Tensor::from_vec(pos, idx.len(), &dev)?;
    let neg = if data.neg.is_empty() {
        None
    } else {
        let j: Vec<usize> = (0..idx.len()).map(|_| rng.gen_range(0..data.neg.len())).collect();
        Some(batch_tensors(m, &data.neg, &j)?)
    };
    let ord = if m.jepa.cfg.ord_dim > 0 {
        let (mut os, mut on) = (Vec::new(), Vec::new());
        for &i in idx {
            os.extend(m.enc.ordinal_values(&m.spec, &data.pos[i].s)?);
            on.extend(m.enc.ordinal_values(&m.spec, &data.pos[i].next)?);
        }
        let k = m.jepa.cfg.ord_dim;
        Some((Tensor::from_vec(os, (idx.len(), k), &dev)?, Tensor::from_vec(on, (idx.len(), k), &dev)?))
    } else {
        None
    };
    let states: Vec<&State> = idx.iter().map(|&i| &data.pos[i].s).collect();
    let (cur, shift) = m.fact_inputs(&states)?;
    let mut next_idx = Vec::new();
    for &i in idx {
        next_idx.extend(m.enc.field_index(&m.spec, &data.pos[i].next)?);
    }
    let next_idx = Tensor::from_vec(next_idx, (idx.len(), m.enc.slots.len()), &dev)?;
    let fi = FactInputs { cur_idx: &cur, shift_idx: &shift };
    let (total, _) = m.jepa.loss(
        &s,
        &a,
        &pos,
        &m.cand,
        neg.as_ref().map(|(ns, na, nn)| (nn, ns, na)),
        ord.as_ref().map(|(a, b)| (a, b)),
        Some((&fi, &next_idx)),
    )?;
    Ok(total)
}

/// Trains a model for `spec` on `dev`, printing progress; returns it with its final report.
pub fn train(spec: &Spec, cfg: &TrainConfig, dev: &Device) -> Result<(Model, J)> {
    ensure!(spec.finite(), "{}: has continuous fields; train it with `continuous::train`", spec.name);
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    if dev.is_cuda() {
        dev.set_seed(cfg.seed)?;
    }
    let model = Model::new(spec.clone(), Model::config_for(spec, cfg)?, cfg.candidates, dev)?;
    let data = dataset(spec, cfg, &mut rng)?;
    ensure!(data.pos.len() >= 2, "{}: fewer than two training pairs", spec.name);
    println!(
        "{}: {} pairs, {} negatives, {} candidates ({:?}), {} params on {:?}",
        spec.name,
        data.pos.len(),
        data.neg.len(),
        model.candidate_count(),
        model.candidates,
        model.jepa.store.count_trainable(),
        dev
    );
    let params = ParamsAdamW { lr: cfg.lr, weight_decay: cfg.weight_decay, ..Default::default() };
    let mut opt = AdamW::new(model.jepa.store.trainable(), params)?;
    let t0 = std::time::Instant::now();
    let mut order: Vec<usize> = (0..data.pos.len()).collect();
    for ep in 0..cfg.epochs {
        // Cosine annealing to 5% of the base rate, stepped per epoch like the reference.
        let t = ep as f64 / cfg.epochs.max(1) as f64;
        opt.set_learning_rate(cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())));
        order.shuffle(&mut rng);
        let (mut tot, mut skipped) = (0f64, 0usize);
        for idx in order.chunks(cfg.batch).filter(|c| c.len() >= 2) {
            let loss = train_batch(&model, &data, idx, &mut rng)?;
            let v = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            if !v.is_finite() {
                skipped += 1; // never let one bad batch poison the run
                continue;
            }
            opt.backward_step(&loss)?;
            model.jepa.ema_update()?;
            tot += f64::from(v) * idx.len() as f64;
        }
        model.cache.lock().expect("energy cache poisoned").clear();
        if ep % cfg.log_every.max(1) == 0 || ep + 1 == cfg.epochs {
            let ev = evaluate(&model, cfg.holdout, &cfg.holdout_salt)?;
            println!(
                "  ep {ep:3} loss {:.4} | exact all {:.3} heldout {:.3} score {:.3}{}",
                tot / data.pos.len() as f64,
                ev["all"]["exact"].as_f64().unwrap_or(0.0),
                ev["heldout"]["exact"].as_f64().unwrap_or(0.0),
                ev["score"].as_f64().unwrap_or(0.0),
                if skipped > 0 { format!(" ({skipped} batches skipped)") } else { String::new() }
            );
        }
    }
    let mut report = evaluate(&model, cfg.holdout, &cfg.holdout_salt)?;
    report["train_pairs"] = json!(data.pos.len());
    report["negatives"] = json!(data.neg.len());
    report["seconds"] = json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
    report["params"] = json!(model.jepa.store.count_trainable());
    report["trained_on"] = json!(if dev.is_cuda() { "cuda" } else { "cpu" });
    Ok((model, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> Spec {
        Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs").join(format!("{name}.nerve"))).unwrap()
    }

    #[test]
    fn toggle_learns_and_round_trips_through_a_checkpoint() -> Result<()> {
        let sp = spec("toggle");
        let cfg = TrainConfig { epochs: 40, d: 16, hidden: 32, log_every: 1000, ..Default::default() };
        let (m, report) = train(&sp, &cfg, &Device::Cpu)?;
        assert_eq!(report["all"]["exact"], 1.0, "{report}");
        let dir = std::env::temp_dir().join(format!("nerve-train-test-{}", std::process::id()));
        m.save(&dir, &cfg, &report)?;
        let back = Model::load(sp.clone(), &dir, &Device::Cpu)?;
        let a = ActionInst::new("flip");
        assert_eq!(back.step(&sp.initial, &a)?, m.step(&sp.initial, &a)?);
        assert_eq!(back.energies(&sp.initial, &a)?, m.energies(&sp.initial, &a)?);
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn valid_candidates_never_answer_with_a_broken_invariant() -> Result<()> {
        let sp = spec("reorder");
        let cfg = Model::config_for(&sp, &TrainConfig { d: 8, hidden: 16, ..Default::default() })?;
        let valid = Model::new(sp.clone(), cfg.clone(), Candidates::Valid, &Device::Cpu)?;
        let all = Model::new(sp.clone(), cfg, Candidates::All, &Device::Cpu)?;
        assert_eq!((valid.candidate_count(), all.candidate_count()), (120, 3125));
        // Untrained, so its answers are arbitrary, but every one of them is an order.
        let oracle = Oracle::new(&sp);
        let actions = sp.action_instances()?;
        for a in &actions {
            assert!(!oracle.breaks_invariant(&valid.step(&sp.initial, a)?), "{}", sp.action_key(a));
        }
        // In a drag, a state the invariants rule out costs infinite energy.
        let broken = sp.state_from_json(&json!({"pa": 0, "pb": 0, "pc": 2, "pd": 3, "pe": 4}))?;
        assert!(valid.energy(&sp.initial, Some(&actions[0]), &broken)?.is_infinite());
        assert!(all.energy(&sp.initial, Some(&actions[0]), &broken)?.is_finite());
        Ok(())
    }

    #[test]
    fn checkpoints_remember_their_candidates_and_old_ones_mean_all() -> Result<()> {
        let sp = spec("counter");
        let cfg = TrainConfig { epochs: 1, d: 8, hidden: 16, log_every: 1000, ..Default::default() };
        let (m, report) = train(&sp, &cfg, &Device::Cpu)?;
        assert_eq!(m.candidates, Candidates::Valid, "valid is the default");
        let dir = std::env::temp_dir().join(format!("nerve-cands-test-{}", std::process::id()));
        m.save(&dir, &cfg, &report)?;
        assert_eq!(Model::load(sp.clone(), &dir, &Device::Cpu)?.candidates, Candidates::Valid);
        assert_eq!(Model::load_as(sp.clone(), &dir, &Device::Cpu, Some(Candidates::All))?.candidates, Candidates::All);
        // A checkpoint from before the choice existed scored every state.
        let path = dir.join("model.json");
        let mut meta: J = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        meta.as_object_mut().unwrap().remove("candidates");
        std::fs::write(&path, meta.to_string())?;
        assert_eq!(Model::load(sp, &dir, &Device::Cpu)?.candidates, Candidates::All);
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn dataset_respects_the_holdout_and_oversamples_holes() -> Result<()> {
        let sp = spec("counter");
        let cfg = TrainConfig::default();
        let d = dataset(&sp, &cfg, &mut rand::rngs::StdRng::seed_from_u64(0))?;
        let frac = effective_holdout(&sp, cfg.holdout)?;
        assert!(d.pos.iter().all(|p| !holdout_mask(&sp, &p.s, &p.a, frac, &cfg.holdout_salt)));
        let holes = sp.hole_actions();
        let n_hole = d.pos.iter().filter(|p| holes.contains(&p.a.name.as_str())).count();
        assert!(n_hole > 0 && n_hole % cfg.hole_weight == 0);
        assert!(d.neg.iter().all(|n| n.next != Oracle::new(&sp).step(&n.s, &n.a).unwrap()));
        Ok(())
    }
}
