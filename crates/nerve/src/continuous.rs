//! The JEPA energy for specs with continuous fields (port of `_train_continuous`,
//! `NerveEnergy._continuous` and `_evaluate_continuous`).
//!
//! There is no finite candidate set to contrast against, so training contrasts each
//! true successor with the batch's other successors and `K` near-miss perturbations,
//! and a step is an argmin over (assignments of the discrete fields) x (continuous
//! values found by gradient descent on the energy from several seeds).

use crate::encode::{Encoder, CONT_WIDTH, N_FOURIER};
use crate::jepa::{Jepa, JepaConfig};
use crate::oracle::{sample_action, sample_state, Oracle};
use crate::serve::LearnedEnergy;
use crate::spec::{ActionInst, Spec, State};
use crate::train::{load_weights, read_meta, save_checkpoint, TrainConfig};
use crate::value::Value;
use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value as J};
use std::collections::BTreeMap;
use std::path::Path;

/// Gradient steps and Adam rate for the descent over continuous fields (as the reference).
const DESCENT_STEPS: usize = 60;
const DESCENT_LR: f64 = 0.05;
/// Random descent seeds on top of current / 0 / 1 / 0.5 / the action's own params.
const RANDOM_SEEDS: usize = 6;
/// Continuous fields within this fraction of their range of the truth count as exact.
const TOLERANCE: f64 = 0.01;
/// Pairs scored at the periodic evaluations during training.
const PROGRESS_SAMPLES: usize = 200;
/// Fixed evaluation seed, so every model is scored on the same pairs.
const EVAL_SEED: u64 = 123;

pub struct ContinuousModel {
    pub spec: Spec,
    pub enc: Encoder,
    pub jepa: Jepa,
    /// One full state per assignment of the discrete fields (continuous fields at `initial`).
    combos: Vec<State>,
    /// `[D, disc_dim]` one-hots and `[D, ord_dim * CONT_WIDTH]` ordinal features of each combo.
    disc: Tensor,
    ord: Option<Tensor>,
    /// `model.json` of a loaded checkpoint (how it was trained and scored).
    meta: Option<J>,
}

/// `u`, then `sin` and `cos` of `2 pi k u` for each Fourier frequency, per column, as the encoder lays them out.
fn cont_features(u: &Tensor) -> Result<Tensor> {
    let mut parts = Vec::with_capacity(u.dim(1)? * CONT_WIDTH);
    for j in 0..u.dim(1)? {
        let c = u.narrow(1, j, 1)?;
        parts.push(c.clone());
        for k in 1..=N_FOURIER {
            let a = (&c * (2.0 * std::f64::consts::PI * k as f64))?;
            parts.push(a.sin()?);
            parts.push(a.cos()?);
        }
    }
    Ok(Tensor::cat(&parts, 1)?)
}

impl ContinuousModel {
    pub fn new(spec: Spec, cfg: JepaConfig, dev: &Device) -> Result<Self> {
        ensure!(!spec.finite(), "{} is finite; train it with `train::train`", spec.name);
        let enc = Encoder::new(&spec)?;
        let jepa = Jepa::new(cfg, &[], dev)?;
        let disc_fields = spec.discrete_fields();
        let axes = disc_fields.iter().map(|&i| spec.fields[i].domain()).collect::<Result<Vec<_>>>()?;
        let combos: Vec<State> = crate::spec::product(&axes)?
            .into_iter()
            .map(|vals| {
                let mut s = spec.initial.clone();
                for (&i, v) in disc_fields.iter().zip(vals) {
                    s[i] = v;
                }
                s
            })
            .collect();
        let ord_off = enc.disc_dim + enc.cont_fields.len() * CONT_WIDTH;
        let ord_w = enc.state_dim - ord_off;
        let (mut disc, mut ord) = (Vec::new(), Vec::new());
        for c in &combos {
            let x = enc.state(&spec, c)?;
            disc.extend_from_slice(&x[..enc.disc_dim]);
            ord.extend_from_slice(&x[ord_off..]);
        }
        let d = combos.len();
        let disc = Tensor::from_vec(disc, (d, enc.disc_dim), dev)?;
        let ord = if ord_w > 0 { Some(Tensor::from_vec(ord, (d, ord_w), dev)?) } else { None };
        Ok(Self { spec, enc, jepa, combos, disc, ord, meta: None })
    }

    pub fn config_for(spec: &Spec, t: &TrainConfig) -> Result<JepaConfig> {
        let enc = Encoder::new(spec)?;
        let mut cfg = JepaConfig::new(enc.state_dim, enc.action_dim());
        (cfg.d, cfg.hidden, cfg.ema, cfg.tau) = (t.d, t.hidden, t.ema, t.tau);
        cfg.action_cont_dim = enc.cont_params.len();
        // The continuous loss uses neither the ordinal readout nor the factored head.
        (cfg.ord_dim, cfg.fact_weight) = (0, 0.0);
        Ok(cfg)
    }

    fn device(&self) -> &Device {
        &self.jepa.device
    }

    fn action_inputs(&self, a: &ActionInst) -> Result<(Tensor, Option<Tensor>)> {
        let dev = self.device();
        let ai = Tensor::new(&[self.enc.action_id(&self.spec, a)? as u32], dev)?;
        let n = self.enc.cont_params.len();
        let ac = if n == 0 { None } else { Some(Tensor::from_vec(self.enc.action_cont(&self.spec, a), (1, n), dev)?) };
        Ok((ai, ac))
    }

    /// The network's prediction `[1, d]` for `s -a->`, with no gradient.
    fn predict(&self, s: &State, a: &ActionInst) -> Result<Tensor> {
        let x = Tensor::from_vec(self.enc.state(&self.spec, s)?, (1, self.enc.state_dim), self.device())?;
        let (ai, ac) = self.action_inputs(a)?;
        Ok(self.jepa.predict(&x, &ai, ac.as_ref())?.detach())
    }

    fn cont_norm(&self, s: &State) -> Vec<f32> {
        self.enc.cont_fields.iter().map(|&k| self.spec.fields[k].norm(&s[k]) as f32).collect()
    }

    /// Descent seeds in `[0, 1]` per continuous field.
    fn seeds(&self, s: &State, a: &ActionInst) -> Vec<Vec<f32>> {
        let n = self.enc.cont_fields.len();
        let mut seeds = vec![self.cont_norm(s), vec![0.0; n], vec![1.0; n], vec![0.5; n]];
        for (an, pk) in &self.enc.cont_params {
            if *an == a.name {
                if let (Some(v), Some(p)) = (a.param(pk), self.spec.action(an).and_then(|x| x.param(pk))) {
                    seeds.push(vec![p.norm(v) as f32; n]);
                }
            }
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        seeds.extend((0..RANDOM_SEEDS).map(|_| (0..n).map(|_| rng.gen::<f32>()).collect()));
        seeds
    }

    /// Argmin over discrete assignments x descended continuous values; also the best
    /// state and energy per discrete assignment.
    pub fn step_all(&self, s: &State, a: &ActionInst) -> Result<(State, Vec<(State, f64)>)> {
        let dev = self.device().clone();
        let zh = self.predict(s, a)?;
        let seeds = self.seeds(s, a);
        let (d, k, n) = (self.combos.len(), seeds.len(), self.enc.cont_fields.len());
        let rows: Vec<u32> = (0..d as u32).flat_map(|i| std::iter::repeat_n(i, k)).collect();
        let rows = Tensor::from_vec(rows, d * k, &dev)?;
        let disc = self.disc.index_select(&rows, 0)?;
        let ord = self.ord.as_ref().map(|o| o.index_select(&rows, 0)).transpose()?;
        let zh = zh.broadcast_as((d * k, self.jepa.cfg.d))?.contiguous()?;
        let u0: Vec<f32> = (0..d).flat_map(|_| seeds.iter().flatten().copied()).collect();
        let u = Var::from_tensor(&Tensor::from_vec(u0, (d * k, n), &dev)?)?;
        let features = |u: &Tensor| -> Result<Tensor> {
            let c = cont_features(u)?;
            let parts: Vec<&Tensor> = [Some(&disc), Some(&c), ord.as_ref()].into_iter().flatten().collect();
            Ok(Tensor::cat(&parts, 1)?)
        };
        let mut opt = AdamW::new(vec![u.clone()], ParamsAdamW { lr: DESCENT_LR, weight_decay: 0.0, ..Default::default() })?;
        for _ in 0..DESCENT_STEPS {
            let e = self.jepa.energy_to(&zh, &features(u.as_tensor())?)?.sum_all()?;
            opt.backward_step(&e)?;
            u.set(&u.as_tensor().clamp(0f32, 1f32)?)?;
        }
        let e: Vec<f32> = self.jepa.energy_to(&zh, &features(u.as_tensor())?)?.to_vec1()?;
        let uv: Vec<Vec<f32>> = u.as_tensor().to_vec2()?;
        let state_at = |row: usize| -> State {
            let mut st = self.combos[row / k].clone();
            for (j, &fi) in self.enc.cont_fields.iter().enumerate() {
                let f = &self.spec.fields[fi];
                let (lo, hi) = f.range();
                st[fi] = f.canon(&Value::Float(lo + f64::from(uv[row][j]) * (hi - lo)));
            }
            st
        };
        let per: Vec<(State, f64)> = (0..d)
            .map(|c| {
                let best = (c * k..(c + 1) * k).min_by(|&i, &j| e[i].total_cmp(&e[j])).unwrap_or(c * k);
                (state_at(best), f64::from(e[best]))
            })
            .collect();
        let best = (0..per.len()).min_by(|&i, &j| per[i].1.total_cmp(&per[j].1)).context("no discrete assignment")?;
        Ok((per[best].0.clone(), per))
    }

    /// The same network on another device. The descent in `step_all` is a few hundred tiny
    /// kernels, which the CPU runs about 3x faster than the GPU for these sizes.
    pub fn on_device(&self, dev: &Device) -> Result<Self> {
        let m = Self::new(self.spec.clone(), self.jepa.cfg.clone(), dev)?;
        for ((name, src, _), (dst_name, dst, _)) in self.jepa.store.vars.iter().zip(&m.jepa.store.vars) {
            ensure!(name == dst_name, "parameter order changed: {name} vs {dst_name}");
            dst.set(&src.as_tensor().to_device(dev)?)?;
        }
        Ok(m)
    }

    pub fn step(&self, s: &State, a: &ActionInst) -> Result<State> {
        Ok(self.step_all(s, a)?.0)
    }

    pub fn save(&self, dir: &Path, t: &TrainConfig, report: &J) -> Result<()> {
        save_checkpoint(dir, &self.jepa, json!({"spec": self.spec.name, "kind": "continuous", "jepa": self.jepa.cfg, "train": t, "eval": report}))
    }

    pub fn load(spec: Spec, dir: &Path, dev: &Device) -> Result<Self> {
        let meta = read_meta(dir, &spec.name)?;
        ensure!(meta["kind"] == "continuous", "{} does not hold a continuous model", dir.display());
        let cfg: JepaConfig = serde_json::from_value(meta["jepa"].clone())?;
        let mut m = Self::new(spec, cfg, dev)?;
        load_weights(&m.jepa, dir, dev)?;
        m.meta = Some(meta);
        Ok(m)
    }
}

impl LearnedEnergy for ContinuousModel {
    /// How surprised the network is by `cand` as the successor of `start -action->`: one forward
    /// pass, no descent, so a drag can afford it for every probe. Staying put costs nothing.
    fn energy(&self, start: &State, action: Option<&ActionInst>, cand: &State) -> Result<f64> {
        let Some(a) = action else { return Ok(0.0) };
        let x = Tensor::from_vec(self.enc.state(&self.spec, cand)?, (1, self.enc.state_dim), self.device())?;
        Ok(f64::from(self.jepa.energy_to(&self.predict(start, a)?, &x)?.squeeze(0)?.to_scalar::<f32>()?))
    }

    /// The best state the network finds for each discrete assignment, with its energy.
    fn step_energies(&self, start: &State, action: &ActionInst) -> Result<Option<Vec<(State, f64)>>> {
        Ok(Some(self.step_all(start, action)?.1))
    }

    /// A forward pass per probe: about 35 s for a full tracks field against 1 s without.
    fn cheap(&self) -> bool {
        false
    }

    fn describe(&self) -> J {
        let params = self.jepa.store.count_trainable();
        let device = if self.device().is_cuda() { "cuda" } else { "cpu" };
        json!({
            "kind": "jepa-continuous",
            "params": params,
            "device": device,
            "summary": format!("{} continuous JEPA, {params} params on {device}", self.spec.name),
            "trained": self.meta.as_ref().map(crate::train::trained_facts),
        })
    }
}

/// A state that satisfies the spec's invariants (rejection sampling).
fn valid_state<R: Rng>(spec: &Spec, oracle: &Oracle, rng: &mut R) -> Result<State> {
    for _ in 0..10_000 {
        let s = sample_state(spec, rng)?;
        if !oracle.breaks_invariant(&s) {
            return Ok(s);
        }
    }
    bail!("{}: could not sample a state that satisfies the invariants", spec.name)
}

fn gauss<R: Rng>(rng: &mut R) -> f64 {
    let (u1, u2): (f64, f64) = (rng.gen_range(f64::EPSILON..1.0), rng.gen());
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// A near-miss successor: jitter continuous fields (sd 15% of range), sometimes re-draw a
/// discrete one (`rollout._perturb`). Unlike the reference, never the truth itself, which
/// would make a negative identical to its positive.
fn near_miss<R: Rng>(spec: &Spec, truth: &State, rng: &mut R) -> Result<State> {
    let disc = spec.discrete_fields();
    let mut cand = truth.clone();
    for _ in 0..20 {
        cand = truth.clone();
        for k in spec.continuous_fields() {
            if rng.gen::<f64>() < 0.7 {
                let (lo, hi) = spec.fields[k].range();
                let v = truth[k].as_f64().unwrap_or(lo) + gauss(rng) * 0.15 * (hi - lo);
                cand[k] = Value::Float(v.clamp(lo, hi));
            }
        }
        if rng.gen::<f64>() < 0.3 && !disc.is_empty() {
            let k = disc[rng.gen_range(0..disc.len())];
            let dom = spec.fields[k].domain()?;
            cand[k] = dom[rng.gen_range(0..dom.len())].clone();
        }
        if cand != *truth {
            break;
        }
    }
    Ok(cand)
}

/// Discrete fields exact, continuous fields within `tol` of their range.
pub fn close(spec: &Spec, pred: &State, truth: &State, tol: f64) -> bool {
    spec.fields.iter().enumerate().all(|(i, f)| {
        if f.continuous() {
            let (lo, hi) = f.range();
            (pred[i].as_f64().unwrap_or(f64::NAN) - truth[i].as_f64().unwrap_or(f64::NAN)).abs() <= tol * (hi - lo)
        } else {
            pred[i] == truth[i]
        }
    })
}

/// Same report and score as `_evaluate_continuous`, on `n` valid states and random actions.
pub fn evaluate(model: &ContinuousModel, n: usize) -> Result<J> {
    let spec = &model.spec;
    let oracle = Oracle::new(spec);
    let mut rng = rand::rngs::StdRng::seed_from_u64(EVAL_SEED);
    let (mut ok, mut bad, mut err, mut n_err) = (0usize, 0usize, 0f64, 0usize);
    let mut per: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for _ in 0..n {
        let s = valid_state(spec, &oracle, &mut rng)?;
        let a = sample_action(spec, &mut rng)?;
        let truth = oracle.step(&s, &a)?;
        let pred = model.step(&s, &a)?;
        let hit = close(spec, &pred, &truth, TOLERANCE);
        ok += usize::from(hit);
        bad += usize::from(!oracle.violations(&s, Some(&a), &pred, TOLERANCE).is_empty());
        let e = per.entry(a.name.clone()).or_default();
        *e = (e.0 + usize::from(hit), e.1 + 1);
        for k in spec.continuous_fields() {
            let (lo, hi) = spec.fields[k].range();
            err += (pred[k].as_f64().unwrap_or(lo) - truth[k].as_f64().unwrap_or(lo)).abs() / (hi - lo);
            n_err += 1;
        }
    }
    let nf = n.max(1) as f64;
    let c = json!({
        "n": n,
        "exact": ok as f64 / nf,
        "violation_rate": bad as f64 / nf,
        "per_action": per.iter().map(|(k, (h, t))| (k.clone(), json!(*h as f64 / *t as f64))).collect::<serde_json::Map<_, _>>(),
        "mean_abs_err_frac": err / n_err.max(1) as f64,
        "tol": TOLERANCE,
    });
    let score = 0.8 * (ok as f64 / nf) + 0.2 * (1.0 - bad as f64 / nf);
    Ok(json!({"all": c, "seen": c, "heldout": c, "learned_only": c, "continuous": true, "score": (score * 1e4).round() / 1e4}))
}

/// Training tensors, all on the device: states, actions, continuous action params,
/// successors and `[n, K, state_dim]` near misses.
struct Data {
    s: Tensor,
    a: Tensor,
    ac: Option<Tensor>,
    next: Tensor,
    negs: Tensor,
    n: usize,
}

fn dataset<R: Rng>(model: &ContinuousModel, cfg: &TrainConfig, rng: &mut R) -> Result<Data> {
    let (spec, enc, dev) = (&model.spec, &model.enc, model.device());
    let oracle = Oracle::new(spec);
    let (n, k, sd, nc) = (cfg.cont_pairs, cfg.cont_negatives.max(1), enc.state_dim, enc.cont_params.len());
    let (mut s, mut a, mut ac, mut next, mut negs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let st = valid_state(spec, &oracle, rng)?;
        let act = sample_action(spec, rng)?;
        let nx = oracle.step(&st, &act)?;
        s.extend(enc.state(spec, &st)?);
        a.push(enc.action_id(spec, &act)? as u32);
        ac.extend(enc.action_cont(spec, &act));
        next.extend(enc.state(spec, &nx)?);
        for _ in 0..k {
            negs.extend(enc.state(spec, &near_miss(spec, &nx, rng)?)?);
        }
    }
    Ok(Data {
        s: Tensor::from_vec(s, (n, sd), dev)?,
        a: Tensor::from_vec(a, n, dev)?,
        ac: if nc == 0 { None } else { Some(Tensor::from_vec(ac, (n, nc), dev)?) },
        next: Tensor::from_vec(next, (n, sd), dev)?,
        negs: Tensor::from_vec(negs, (n, k, sd), dev)?,
        n,
    })
}

/// Trains a model for a spec with continuous fields on `dev`; returns it with its report.
pub fn train(spec: &Spec, cfg: &TrainConfig, dev: &Device) -> Result<(ContinuousModel, J)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    if dev.is_cuda() {
        dev.set_seed(cfg.seed)?;
    }
    let model = ContinuousModel::new(spec.clone(), ContinuousModel::config_for(spec, cfg)?, dev)?;
    let data = dataset(&model, cfg, &mut rng)?;
    ensure!(data.n >= 2, "{}: fewer than two training pairs", spec.name);
    println!(
        "{}: {} pairs x {} near misses, {} discrete assignments, {} params on {:?}",
        spec.name,
        data.n,
        cfg.cont_negatives,
        model.combos.len(),
        model.jepa.store.count_trainable(),
        dev
    );
    let params = ParamsAdamW { lr: cfg.lr, weight_decay: cfg.weight_decay, ..Default::default() };
    let mut opt = AdamW::new(model.jepa.store.trainable(), params)?;
    let t0 = std::time::Instant::now();
    let mut order: Vec<u32> = (0..data.n as u32).collect();
    for ep in 0..cfg.epochs {
        let t = ep as f64 / cfg.epochs.max(1) as f64;
        opt.set_learning_rate(cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())));
        order.shuffle(&mut rng);
        let (mut tot, mut skipped) = (0f64, 0usize);
        for chunk in order.chunks(cfg.batch).filter(|c| c.len() >= 2) {
            let idx = Tensor::from_slice(chunk, chunk.len(), dev)?;
            let ac = data.ac.as_ref().map(|t| t.index_select(&idx, 0)).transpose()?;
            let (loss, _) = model.jepa.loss_continuous(
                &data.s.index_select(&idx, 0)?,
                &data.a.index_select(&idx, 0)?,
                ac.as_ref(),
                &data.next.index_select(&idx, 0)?,
                &data.negs.index_select(&idx, 0)?,
            )?;
            let v = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            if !v.is_finite() {
                skipped += 1; // never let one bad batch poison the run
                continue;
            }
            opt.backward_step(&loss)?;
            model.jepa.ema_update()?;
            tot += f64::from(v) * chunk.len() as f64;
        }
        if ep % cfg.log_every.max(1) == 0 || ep + 1 == cfg.epochs {
            let ev = evaluate(&model.on_device(&Device::Cpu)?, PROGRESS_SAMPLES.min(cfg.eval_samples))?;
            println!(
                "  ep {ep:3} loss {:.4} | within-tol {:.3} violations {:.3} score {:.3}{}",
                tot / data.n as f64,
                ev["all"]["exact"].as_f64().unwrap_or(0.0),
                ev["all"]["violation_rate"].as_f64().unwrap_or(0.0),
                ev["score"].as_f64().unwrap_or(0.0),
                if skipped > 0 { format!(" ({skipped} batches skipped)") } else { String::new() }
            );
        }
    }
    let mut report = evaluate(&model.on_device(&Device::Cpu)?, cfg.eval_samples)?;
    report["train_pairs"] = json!(data.n);
    report["negatives"] = json!(data.n * cfg.cont_negatives);
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

    fn small() -> TrainConfig {
        TrainConfig { epochs: 3, d: 16, hidden: 32, cont_pairs: 64, cont_negatives: 4, eval_samples: 4, log_every: 1000, ..Default::default() }
    }

    #[test]
    fn features_match_the_encoder() -> Result<()> {
        let sp = spec("tracks");
        let m = ContinuousModel::new(sp.clone(), ContinuousModel::config_for(&sp, &small())?, &Device::Cpu)?;
        let s = sp.state_from_json(&json!({"track": 2, "pos": 150.0}))?;
        let want = m.enc.state(&sp, &s)?;
        let u = Tensor::new(&[[150f32 / 240.0]], &Device::Cpu)?;
        let combo = m.combos.iter().position(|c| c[0] == Value::Int(2)).unwrap() as u32;
        let row = Tensor::new(&[combo], &Device::Cpu)?;
        let mut parts = vec![m.disc.index_select(&row, 0)?, cont_features(&u)?];
        parts.extend(m.ord.as_ref().map(|o| o.index_select(&row, 0)).transpose()?);
        let got: Vec<f32> = Tensor::cat(&parts, 1)?.squeeze(0)?.to_vec1()?;
        assert_eq!(got.len(), want.len());
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-5), "{got:?} vs {want:?}");
        assert_eq!(m.combos.len(), 3);
        Ok(())
    }

    #[test]
    fn near_misses_are_never_the_truth_and_stay_in_range() -> Result<()> {
        let sp = spec("tracks");
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let truth = sp.state_from_json(&json!({"track": 1, "pos": 100.0}))?;
        for _ in 0..500 {
            let c = near_miss(&sp, &truth, &mut rng)?;
            assert_ne!(c, truth);
            assert!((0.0..=240.0).contains(&c[1].as_f64().unwrap()));
        }
        Ok(())
    }

    #[test]
    fn trains_steps_and_round_trips_through_a_checkpoint() -> Result<()> {
        let sp = spec("tracks");
        let cfg = small();
        let (m, report) = train(&sp, &cfg, &Device::Cpu)?;
        assert!(report["all"]["exact"].as_f64().is_some() && report["continuous"] == true);
        let s = sp.state_from_json(&json!({"track": 1, "pos": 100.0}))?;
        let a = ActionInst { name: "move".into(), params: vec![("t".into(), Value::Int(2)), ("to".into(), Value::Float(30.0))] };
        let (pred, per) = m.step_all(&s, &a)?;
        assert_eq!(per.len(), 3, "one best state per track");
        assert!(sp.fields[1].contains(&pred[1]) && sp.fields[0].contains(&pred[0]));
        let dir = std::env::temp_dir().join(format!("nerve-cont-test-{}", std::process::id()));
        m.save(&dir, &cfg, &report)?;
        let back = ContinuousModel::load(sp.clone(), &dir, &Device::Cpu)?;
        assert_eq!(back.step(&s, &a)?, pred);
        assert_eq!(back.energy(&s, Some(&a), &pred)?, m.energy(&s, Some(&a), &pred)?);
        assert!(crate::train::load_learned(sp, &dir, &Device::Cpu)?.describe()["kind"] == "jepa-continuous");
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn close_compares_continuous_fields_within_tolerance() {
        let sp = spec("tracks");
        let s = |t: i64, p: f64| sp.state_from_json(&json!({"track": t, "pos": p})).unwrap();
        assert!(close(&sp, &s(1, 100.0), &s(1, 102.0), 0.01));
        assert!(!close(&sp, &s(1, 100.0), &s(1, 103.0), 0.01));
        assert!(!close(&sp, &s(0, 100.0), &s(1, 100.0), 0.01));
    }
}
