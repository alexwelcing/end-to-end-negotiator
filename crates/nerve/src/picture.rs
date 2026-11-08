//! The render network: state -> pixels, one per interface, trained from the render oracle
//! (`render.rs`) the way the energy network is trained from the behaviour oracle. The spec says what
//! the interface looks like; this network is what draws it in the lab, and the oracle checks every
//! picture it draws (mean absolute pixel error). A small decoder: the state encoding through two
//! linear layers to a coarse grid, then five nearest-upsample-and-convolve stages to the raster.

use crate::encode::Encoder;
use crate::jepa::{Lin, Store};
use crate::oracle::Oracle;
use crate::render;
use crate::spec::{Spec, State};
use crate::value::Value;
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as J};
use std::collections::HashMap;
use std::path::Path;

pub const META_FILE: &str = "picture.json";
pub const WEIGHTS_FILE: &str = "picture.safetensors";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PictureConfig {
    pub w: usize,
    pub h: usize,
    pub state_dim: usize,
    pub hidden: usize,
    /// Channels of the coarse grid the decoder starts from (`w/32` x `h/32`).
    pub base: usize,
    /// Channels after each of the five upsampling stages.
    pub channels: Vec<usize>,
}

impl PictureConfig {
    pub fn for_spec(spec: &Spec) -> Result<Self> {
        let scene = spec.scene.as_ref().with_context(|| format!("{} has no scene: nothing to draw", spec.name))?;
        let (w, h) = scene.raster_size();
        let enc = Encoder::new(spec)?;
        Ok(Self { w, h, state_dim: enc.state_dim, hidden: 256, base: 64, channels: vec![48, 32, 24, 16, 16] })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PictureTrainConfig {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub seed: u64,
    /// States drawn for a spec whose state space cannot be enumerated.
    pub samples: usize,
    pub log_every: usize,
}

impl Default for PictureTrainConfig {
    fn default() -> Self {
        Self { epochs: 400, batch: 32, lr: 2e-3, weight_decay: 0.01, seed: 0, samples: 512, log_every: 50 }
    }
}

/// A 3x3 convolution with padding 1.
struct Conv {
    w: Var,
    b: Var,
    cout: usize,
}

impl Conv {
    fn new(st: &mut Store, name: &str, cin: usize, cout: usize, dev: &Device) -> Result<Self> {
        let r = (1.0 / (cin * 9) as f64).sqrt() as f32;
        let w = st.add(format!("{name}.weight"), Tensor::rand(-r, r, (cout, cin, 3, 3), dev)?, true)?;
        let b = st.add(format!("{name}.bias"), Tensor::zeros(cout, DType::F32, dev)?, true)?;
        Ok(Self { w, b, cout })
    }

    /// The convolution as one matrix product over the nine shifted copies of the padded input
    /// (im2col). candle's CUDA `conv2d` returns a wrong input gradient, padded by itself or by hand
    /// (bias gradients match the CPU, kernel gradients do not); pads, narrows, a cat and a matmul
    /// agree on every backend, and one product per layer keeps it quick.
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        let (b, cin, h, w) = x.dims4()?;
        let xp = x.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?;
        let mut taps = Vec::with_capacity(9);
        for dy in 0..3 {
            for dx in 0..3 {
                taps.push(xp.narrow(2, dy, h)?.narrow(3, dx, w)?.contiguous()?);
            }
        }
        // Rows of `cols` are pixels, columns are (tap, input channel); the kernel is laid out the same way.
        let cols = Tensor::cat(&taps, 1)?.reshape((b, 9 * cin, h * w))?.transpose(1, 2)?.contiguous()?;
        let wk = self.w.as_tensor().permute((2, 3, 1, 0))?.contiguous()?.reshape((9 * cin, self.cout))?;
        let y = cols.broadcast_matmul(&wk)?.transpose(1, 2)?.contiguous()?.reshape((b, self.cout, h, w))?;
        Ok(y.broadcast_add(&self.b.as_tensor().reshape((1, self.cout, 1, 1))?)?)
    }
}

pub struct Picture {
    pub spec: Spec,
    pub cfg: PictureConfig,
    pub store: Store,
    fc1: Lin,
    fc2: Lin,
    convs: Vec<Conv>,
    out: Conv,
    enc: Encoder,
    dev: Device,
    pub meta: Option<J>,
}

impl Picture {
    pub fn new(spec: Spec, cfg: PictureConfig, dev: &Device) -> Result<Self> {
        ensure!(cfg.channels.len() == 5, "the decoder has five upsampling stages");
        ensure!(cfg.w.is_multiple_of(32) && cfg.h.is_multiple_of(32), "raster sides must be multiples of 32");
        let enc = Encoder::new(&spec)?;
        ensure!(enc.state_dim == cfg.state_dim, "checkpoint encodes {} state features, the spec {}", cfg.state_dim, enc.state_dim);
        let mut st = Store::default();
        let (w0, h0) = (cfg.w / 32, cfg.h / 32);
        let fc1 = Lin::new(&mut st, "fc1", cfg.state_dim, cfg.hidden, true, dev, true)?;
        let fc2 = Lin::new(&mut st, "fc2", cfg.hidden, cfg.base * w0 * h0, true, dev, true)?;
        let mut convs = Vec::new();
        let mut cin = cfg.base;
        for (i, &c) in cfg.channels.iter().enumerate() {
            convs.push(Conv::new(&mut st, &format!("up{i}"), cin, c, dev)?);
            cin = c;
        }
        let out = Conv::new(&mut st, "out", cin, 3, dev)?;
        Ok(Self { spec, cfg, store: st, fc1, fc2, convs, out, enc, dev: dev.clone(), meta: None })
    }

    /// `[B, state_dim] -> [B, 3, h, w]` in 0..1.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (w0, h0) = (self.cfg.w / 32, self.cfg.h / 32);
        let b = x.dim(0)?;
        let z = self.fc1.fwd(x, false)?.gelu_erf()?;
        let mut z = self.fc2.fwd(&z, false)?.gelu_erf()?.reshape((b, self.cfg.base, h0, w0))?;
        let (mut h, mut w) = (h0, w0);
        for conv in &self.convs {
            // Nearest-neighbour upsampling by two as a broadcast: every backend's backward of a
            // broadcast is a sum, where the dedicated upsample op's gradient is not trustworthy on CUDA.
            let c = z.dim(1)?;
            z = z.reshape((b, c, h, 1, w, 1))?.broadcast_as((b, c, h, 2, w, 2))?.contiguous()?.reshape((b, c, 2 * h, 2 * w))?;
            h *= 2;
            w *= 2;
            z = conv.fwd(&z)?.gelu_erf()?;
        }
        // Linear output: a sigmoid here saturates on the (mostly background) targets and its gradient
        // dies before the shapes are learned. Pictures are clamped to 0..1 when drawn.
        self.out.fwd(&z)
    }

    fn encode(&self, states: &[State]) -> Result<Tensor> {
        let mut xs = Vec::with_capacity(states.len() * self.cfg.state_dim);
        for s in states {
            xs.extend(self.enc.state(&self.spec, s)?);
        }
        Ok(Tensor::from_vec(xs, (states.len(), self.cfg.state_dim), &self.dev)?)
    }

    /// The network's picture of one state, RGB floats in CHW order.
    pub fn draw(&self, s: &State) -> Result<Vec<f32>> {
        let x = self.encode(std::slice::from_ref(s))?;
        Ok(self.forward(&x)?.clamp(0f32, 1f32)?.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Mean absolute pixel error against the render oracle, over `states`, with the worst state.
    pub fn evaluate(&self, states: &[State]) -> Result<J> {
        let (w, h) = (self.cfg.w, self.cfg.h);
        let mut total = 0.0f64;
        let mut worst = (0.0f32, String::new());
        for chunk in states.chunks(64) {
            let pred = self.forward(&self.encode(chunk)?)?.clamp(0f32, 1f32)?.to_vec3_flat(chunk.len())?;
            for (s, p) in chunk.iter().zip(pred.chunks(3 * w * h)) {
                let truth = render::rasterize(&self.spec, s, w, h)?;
                let e = render::mae(p, &truth);
                total += f64::from(e);
                if e > worst.0 {
                    worst = (e, self.spec.state_key(s));
                }
            }
        }
        Ok(json!({"n": states.len(), "mae": total / states.len().max(1) as f64, "worst_mae": worst.0, "worst_state": worst.1, "w": w, "h": h}))
    }

    pub fn describe(&self) -> J {
        json!({
            "kind": "picture",
            "w": self.cfg.w,
            "h": self.cfg.h,
            "params": self.store.count_trainable(),
            "device": if self.dev.is_cuda() { "cuda" } else { "cpu" },
            "trained": self.meta.as_ref().map(|m| json!({"eval": m["eval"], "train": m["train"]})),
        })
    }

    pub fn save(&self, dir: &Path, t: &PictureTrainConfig, report: &J) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let tensors: HashMap<String, Tensor> = self.store.vars.iter().map(|(n, v, _)| (n.clone(), v.as_tensor().clone())).collect();
        candle_core::safetensors::save(&tensors, dir.join(WEIGHTS_FILE))?;
        let meta = json!({"spec": self.spec.name, "kind": "picture", "picture": self.cfg, "train": t, "eval": report});
        std::fs::write(dir.join(META_FILE), serde_json::to_string_pretty(&meta)?)?;
        Ok(())
    }

    pub fn load(spec: Spec, dir: &Path, dev: &Device) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join(META_FILE)).with_context(|| format!("reading {}", dir.join(META_FILE).display()))?;
        let weights = candle_core::safetensors::load(dir.join(WEIGHTS_FILE), dev)?;
        Self::from_parts(spec, &text, weights, dev)
    }

    /// From a checkpoint in memory: the `picture.json` text and the `picture.safetensors` bytes.
    pub fn from_bytes(spec: Spec, meta_json: &str, weights: &[u8], dev: &Device) -> Result<Self> {
        Self::from_parts(spec, meta_json, candle_core::safetensors::load_buffer(weights, dev)?, dev)
    }

    fn from_parts(spec: Spec, meta_json: &str, weights: HashMap<String, Tensor>, dev: &Device) -> Result<Self> {
        let meta: J = serde_json::from_str(meta_json)?;
        ensure!(meta["spec"] == spec.name && meta["kind"] == "picture", "the checkpoint is not a picture of {}", spec.name);
        let cfg: PictureConfig = serde_json::from_value(meta["picture"].clone())?;
        let mut p = Self::new(spec, cfg, dev)?;
        for (name, var, _) in &p.store.vars {
            let w = weights.get(name).with_context(|| format!("checkpoint is missing {name}"))?;
            var.set(w)?;
        }
        p.meta = Some(meta);
        Ok(p)
    }
}

trait FlatBatch {
    fn to_vec3_flat(&self, b: usize) -> Result<Vec<f32>>;
}

impl FlatBatch for Tensor {
    fn to_vec3_flat(&self, b: usize) -> Result<Vec<f32>> {
        Ok(self.reshape((b, ()))?.flatten_all()?.to_vec1::<f32>()?)
    }
}

/// The states a picture is trained and scored on: every valid state of a finite spec, or `samples`
/// random valid states (plus the initial one) of a continuous spec.
pub fn training_states(spec: &Spec, samples: usize, seed: u64) -> Result<Vec<State>> {
    let oracle = Oracle::new(spec);
    if spec.finite() {
        return Ok(spec.state_space()?.into_iter().filter(|s| !oracle.breaks_invariant(s)).collect());
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut out = vec![spec.initial.clone()];
    let mut tries = 0;
    while out.len() < samples && tries < samples * 50 {
        tries += 1;
        let s: State = spec
            .fields
            .iter()
            .map(|f| {
                if f.continuous() {
                    let (lo, hi) = f.range();
                    f.canon(&Value::Float(rng.gen_range(lo..=hi)))
                } else {
                    f.domain().map(|d| d[rng.gen_range(0..d.len())].clone()).unwrap_or_else(|_| f.canon(&Value::Int(0)))
                }
            })
            .collect();
        if !oracle.breaks_invariant(&s) {
            out.push(s);
        }
    }
    Ok(out)
}

/// Trains the picture of `spec` on the render oracle: every valid state (or a sample of them), the
/// raster as the target, mean squared plus mean absolute error.
pub fn train(spec: &Spec, cfg: &PictureTrainConfig, dev: &Device) -> Result<(Picture, J)> {
    train_with(spec, cfg, PictureConfig::for_spec(spec)?, dev)
}

/// `train` with an explicit decoder shape (tests use a small raster).
pub fn train_with(spec: &Spec, cfg: &PictureTrainConfig, pc: PictureConfig, dev: &Device) -> Result<(Picture, J)> {
    let t0 = std::time::Instant::now();
    let (w, h) = (pc.w, pc.h);
    let states = training_states(spec, cfg.samples, cfg.seed)?;
    ensure!(!states.is_empty(), "no states to draw");
    let mut targets = Vec::with_capacity(states.len() * 3 * w * h);
    for s in &states {
        targets.extend(render::rasterize(spec, s, w, h)?);
    }
    let targets = Tensor::from_vec(targets, (states.len(), 3, h, w), &Device::Cpu)?;
    let model = Picture::new(spec.clone(), pc, dev)?;
    let inputs = model.encode(&states)?.to_device(&Device::Cpu)?;
    let mut opt = AdamW::new(model.store.trainable(), ParamsAdamW { lr: cfg.lr, weight_decay: cfg.weight_decay, ..Default::default() })?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let n = states.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut last = f32::NAN;
    for epoch in 0..cfg.epochs {
        order.shuffle(&mut rng);
        let mut sum = 0.0f32;
        let mut batches = 0;
        for chunk in order.chunks(cfg.batch) {
            let idx = Tensor::new(chunk, &Device::Cpu)?;
            let x = inputs.index_select(&idx, 0)?.to_device(dev)?;
            let y = targets.index_select(&idx, 0)?.to_device(dev)?;
            let pred = model.forward(&x)?;
            let diff = (pred - &y)?;
            let loss = (diff.sqr()?.mean_all()? + (diff.abs()?.mean_all()? * 0.3)?)?;
            opt.backward_step(&loss)?;
            sum += loss.to_scalar::<f32>()?;
            batches += 1;
        }
        last = sum / batches.max(1) as f32;
        if cfg.log_every > 0 && (epoch + 1) % cfg.log_every == 0 {
            println!("epoch {:>4}  loss {:.5}", epoch + 1, last);
        }
    }
    let mut report = model.evaluate(&states)?;
    report["loss"] = json!(last);
    report["seconds"] = json!(t0.elapsed().as_secs_f64());
    report["trained_on"] = json!(if dev.is_cuda() { "cuda" } else { "cpu" });
    report["params"] = json!(model.store.count_trainable());
    Ok((model, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOGGLE_WITH_SCENE: &str = r#"
nerve: 1
name: toggle_pic
state:
  status: {type: enum, values: [off, on]}
actions:
  flip: {}
transitions:
  flip: { status: "'on' if status == 'off' else 'off'" }
scene:
  box: [0, 0, 120, 60]
  shapes:
    - {rect: [60, 30, 92, 34], r: 17, color: track}
    - {rect: [60, 30, 92, 34], r: 17, color: acc, when: "status == 'on'"}
render:
  knob: { x: "90 if status == 'on' else 30", y: 30, shape: {circle: [0, 0, 14], color: panel} }
initial: {status: off}
"#;

    #[test]
    fn a_two_state_picture_trains_to_the_oracle_and_round_trips() -> Result<()> {
        let spec = Spec::parse(TOGGLE_WITH_SCENE)?;
        let cfg = PictureTrainConfig { epochs: 200, batch: 2, lr: 3e-3, log_every: 0, ..Default::default() };
        let small = PictureConfig { w: 64, h: 32, state_dim: Encoder::new(&spec)?.state_dim, hidden: 64, base: 32, channels: vec![32, 24, 16, 16, 16] };
        let (pic, report) = train_with(&spec, &cfg, small, &Device::Cpu)?;
        let mae = report["mae"].as_f64().unwrap();
        assert!(mae < 0.06, "the picture should get close in 200 steps: {report}");
        // The two states are told apart: the network's pictures differ where the oracle's do.
        let states = training_states(&spec, 0, 0)?;
        let (a, b) = (pic.draw(&states[0])?, pic.draw(&states[1])?);
        assert!(render::mae(&a, &b) > 0.01);
        let dir = std::env::temp_dir().join(format!("nerve-picture-test-{}", std::process::id()));
        pic.save(&dir, &cfg, &report)?;
        let back = Picture::load(spec.clone(), &dir, &Device::Cpu)?;
        assert_eq!(back.draw(&states[0])?, a);
        let meta = std::fs::read_to_string(dir.join(META_FILE)).with_context(|| format!("reading {}", dir.join(META_FILE).display()))?;
        let bytes = std::fs::read(dir.join(WEIGHTS_FILE)).with_context(|| format!("reading {}", dir.join(WEIGHTS_FILE).display()))?;
        assert_eq!(Picture::from_bytes(spec, &meta, &bytes, &Device::Cpu)?.draw(&states[1])?, b);
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }
}


#[cfg(all(test, feature = "cuda"))]
mod cuda_check {
    use super::*;

    /// The same weights, the same batch: the GPU's loss and gradients must match the CPU's.
    #[test]
    fn cuda_gradients_match_the_cpu() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else { return Ok(()) };
        let spec = Spec::parse(include_str!("../../../specs/toggle.nerve"))?;
        let states = training_states(&spec, 0, 0)?;
        let cfg = PictureConfig::for_spec(&spec)?;
        let cpu = Picture::new(spec.clone(), cfg.clone(), &Device::Cpu)?;
        let dir = std::env::temp_dir().join(format!("nerve-cuda-check-{}", std::process::id()));
        cpu.save(&dir, &PictureTrainConfig::default(), &json!({}))?;
        let gpu = Picture::load(spec.clone(), &dir, &cuda)?;
        std::fs::remove_dir_all(&dir).ok();
        let (w, h) = (cfg.w, cfg.h);
        let mut t = Vec::new();
        for s in &states {
            t.extend(render::rasterize(&spec, s, w, h)?);
        }
        let report = |m: &Picture, dev: &Device| -> Result<(f32, Vec<(String, f32)>)> {
            let x = m.encode(&states)?;
            let y = Tensor::from_vec(t.clone(), (states.len(), 3, h, w), dev)?;
            let diff = (m.forward(&x)? - &y)?;
            let loss = (diff.sqr()?.mean_all()? + (diff.abs()?.mean_all()? * 0.3)?)?;
            let grads = loss.backward()?;
            let norms = m.store.vars.iter().map(|(n, v, _)| Ok((n.clone(), grads.get(v).map(|g| g.sqr()?.sum_all()?.to_scalar::<f32>()).transpose()?.unwrap_or(0.0).sqrt()))).collect::<Result<Vec<_>>>()?;
            Ok((loss.to_scalar::<f32>()?, norms))
        };
        let (lc, gc) = report(&cpu, &Device::Cpu)?;
        let (lg, gg) = report(&gpu, &cuda)?;
        eprintln!("loss cpu {lc:.6} cuda {lg:.6}");
        let mut bad = Vec::new();
        for ((n, a), (_, b)) in gc.iter().zip(&gg) {
            let rel = (a - b).abs() / a.abs().max(1e-6);
            eprintln!("{n:<14} cpu {a:.6e} cuda {b:.6e} rel {rel:.3}");
            if rel > 0.05 {
                bad.push(n.clone());
            }
        }
        assert!((lc - lg).abs() < 1e-3 && bad.is_empty(), "gradients differ on: {bad:?}");
        Ok(())
    }
}
