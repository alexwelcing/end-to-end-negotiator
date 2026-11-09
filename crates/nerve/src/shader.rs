//! The shader: the picture as a function of the pixel, learned from the spec. Where `picture.rs`
//! decodes a state into a fixed raster, the shader answers one question, "what colour is the point
//! (x, y) of the interface in this state?", so it draws at any resolution, and its edges are as sharp
//! as its highest Fourier octave. It is trained on the render oracle at several times the raster's
//! resolution, on random pixels, and scored like the picture: mean absolute pixel error against the
//! oracle. Because each pixel is independent, the browser runs the same weights as a fragment shader.
//!
//! ```text
//! c      = cond(state features)                       what to draw
//! f(x,y) = (x, y, sin(2^k π x), cos(2^k π x), ... )    where, k < octaves
//! h_0    = gelu(W_in f)
//! h_i    = gelu(W_i h_{i-1}) * (1 + g_i(c)) + b_i(c)   FiLM: the state modulates every layer
//! rgb    = W_out h_L                                   clamped to 0..1 when drawn
//! ```

use crate::encode::Encoder;
use crate::jepa::{Lin, Store};
use crate::picture::training_states;
use crate::render;
use crate::spec::{Spec, State};
use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as J};
use std::collections::HashMap;
use std::path::Path;

pub const META_FILE: &str = "shader.json";
pub const WEIGHTS_FILE: &str = "shader.safetensors";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderConfig {
    /// The training raster (`scale` times the scene's raster): what `evaluate` scores against, and the
    /// aspect the coordinates are normalised to.
    pub w: usize,
    pub h: usize,
    pub state_dim: usize,
    pub cond: usize,
    pub hidden: usize,
    pub layers: usize,
    pub octaves: usize,
    /// `"mlp"`: every pixel runs the whole network (the first shaders). `"grid"`: most of the capacity is
    /// a learned multi-resolution feature grid, sampled bilinearly (a hardware texture fetch on the GPU),
    /// and the per-pixel network is small: about ten times fewer operations per pixel.
    #[serde(default = "default_kind")]
    pub kind: String,
    /// Grid levels as `[width, height]` in cells, coarse to fine (grid kind only).
    #[serde(default)]
    pub grid_levels: Vec<[usize; 2]>,
    /// Channels per grid cell (grid kind only).
    #[serde(default)]
    pub grid_channels: usize,
}

fn default_kind() -> String {
    "mlp".to_string()
}

impl ShaderConfig {
    pub fn for_spec(spec: &Spec, scale: usize) -> Result<Self> {
        let scene = spec.scene.as_ref().context("the spec has no scene to draw")?;
        let (w, h) = scene.raster_size();
        let enc = Encoder::new(spec)?;
        Ok(Self { w: w * scale.max(1), h: h * scale.max(1), state_dim: enc.state_dim, cond: 128, hidden: 128, layers: 4, octaves: 8, kind: "mlp".into(), grid_levels: vec![], grid_channels: 0 })
    }

    /// The grid kind: feature grids of `bases` rows (columns follow the aspect), `channels` each, and a
    /// small per-pixel network.
    pub fn grid_for_spec(spec: &Spec, scale: usize, bases: &[usize], channels: usize) -> Result<Self> {
        let mut c = Self::for_spec(spec, scale)?;
        let aspect = c.aspect();
        c.kind = "grid".into();
        c.grid_levels = bases.iter().map(|&b| [((b as f64 * aspect).round() as usize).max(2), b.max(2)]).collect();
        c.grid_channels = channels;
        c.hidden = 64;
        c.layers = 2;
        c.octaves = 4;
        Ok(c)
    }

    pub fn is_grid(&self) -> bool {
        self.kind == "grid"
    }

    pub fn grid_dim(&self) -> usize {
        if self.is_grid() { self.grid_levels.len() * self.grid_channels } else { 0 }
    }

    pub fn aspect(&self) -> f64 {
        self.w as f64 / self.h as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderTrainConfig {
    pub steps: usize,
    pub batch_states: usize,
    pub pixels: usize,
    pub lr: f64,
    pub seed: u64,
    pub samples: usize,
    pub scale: usize,
    pub log_every: usize,
}

impl Default for ShaderTrainConfig {
    fn default() -> Self {
        Self { steps: 4000, batch_states: 8, pixels: 2048, lr: 1e-3, seed: 0, samples: 256, scale: 4, log_every: 200 }
    }
}

pub struct Shader {
    pub spec: Spec,
    pub cfg: ShaderConfig,
    pub store: Store,
    cond: [Lin; 2],
    inp: Lin,
    layers: Vec<Lin>,
    film: Vec<Lin>,
    out: Lin,
    /// Feature grids, one per level, `[cells, channels]` (grid kind only).
    grids: Vec<candle_core::Var>,
    enc: Encoder,
    dev: Device,
    pub meta: Option<J>,
}

/// Bilinear sampling of a `[gw, gh]` grid at normalised coordinates: the four corner cell indices and
/// their weights for every pixel. Pixel coordinates are -1..1 (x right, y down); cell `i` sits at
/// `i / (g - 1)` of the way across, so the grid's edges meet the raster's.
fn bilinear(xy: &[f32], gw: usize, gh: usize) -> (Vec<u32>, Vec<f32>) {
    let n = xy.len() / 2;
    let (mut idx, mut wts) = (Vec::with_capacity(4 * n), Vec::with_capacity(4 * n));
    for p in 0..n {
        let u = ((xy[2 * p] + 1.0) * 0.5).clamp(0.0, 1.0) * (gw - 1) as f32;
        let v = ((xy[2 * p + 1] + 1.0) * 0.5).clamp(0.0, 1.0) * (gh - 1) as f32;
        let (x0, y0) = (u.floor().min((gw - 2) as f32), v.floor().min((gh - 2) as f32));
        let (fx, fy) = (u - x0, v - y0);
        let (x0, y0) = (x0 as usize, y0 as usize);
        for (dx, dy, w) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
            idx.push(((y0 + dy) * gw + x0 + dx) as u32);
            wts.push(w);
        }
    }
    (idx, wts)
}

impl Shader {
    pub fn new(spec: Spec, cfg: ShaderConfig, dev: &Device) -> Result<Self> {
        let enc = Encoder::new(&spec)?;
        ensure!(enc.state_dim == cfg.state_dim, "checkpoint encodes {} state features, the spec {}", cfg.state_dim, enc.state_dim);
        ensure!(cfg.layers >= 1 && cfg.octaves >= 1, "a shader needs at least one layer and one octave");
        let mut st = Store::default();
        let cond = [Lin::new(&mut st, "cond.0", cfg.state_dim, 256, true, dev, true)?, Lin::new(&mut st, "cond.2", 256, cfg.cond, true, dev, true)?];
        let mut grids = Vec::new();
        if cfg.is_grid() {
            ensure!(cfg.grid_channels > 0 && !cfg.grid_levels.is_empty(), "the grid kind needs levels and channels");
            for (l, [gw, gh]) in cfg.grid_levels.iter().enumerate() {
                grids.push(st.add(format!("grid.{l}"), Tensor::rand(-0.1f32, 0.1f32, (gw * gh, cfg.grid_channels), dev)?, true)?);
            }
        }
        let inp = Lin::new(&mut st, "inp", 2 + 4 * cfg.octaves + cfg.grid_dim(), cfg.hidden, true, dev, true)?;
        let mut layers = Vec::new();
        let mut film = Vec::new();
        for i in 0..cfg.layers {
            layers.push(Lin::new(&mut st, &format!("layers.{i}"), cfg.hidden, cfg.hidden, true, dev, true)?);
            film.push(Lin::new(&mut st, &format!("film.{i}"), cfg.cond, 2 * cfg.hidden, true, dev, true)?);
        }
        let out = Lin::new(&mut st, "out", cfg.hidden, 3, true, dev, true)?;
        Ok(Self { spec, cfg, store: st, cond, inp, layers, film, out, grids, enc, dev: dev.clone(), meta: None })
    }

    /// Per-pixel inputs from raw coordinates: the `[B, P, 2]` tensor and, for the grid kind, each level's
    /// bilinear corner indices `[B*P*4]` and weights `[B*P, 4]`.
    fn inputs(&self, xy: &[f32], b: usize) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let p = xy.len() / 2 / b;
        let t = Tensor::from_slice(xy, (b, p, 2), &self.dev)?;
        let mut taps = Vec::new();
        for [gw, gh] in &self.cfg.grid_levels {
            let (idx, wts) = bilinear(xy, *gw, *gh);
            taps.push((Tensor::from_vec(idx, b * p * 4, &self.dev)?, Tensor::from_vec(wts, (b * p, 4), &self.dev)?));
        }
        Ok((t, taps))
    }

    /// Grid features: `[B, P, levels * channels]`, bilinear over each level.
    fn grid_features(&self, taps: &[(Tensor, Tensor)], b: usize, p: usize) -> Result<Tensor> {
        let c = self.cfg.grid_channels;
        let mut feats = Vec::new();
        for (g, (idx, wts)) in self.grids.iter().zip(taps) {
            let corners = g.as_tensor().index_select(idx, 0)?.reshape((b * p, 4, c))?;           // [B*P, 4, C]
            let f = corners.broadcast_mul(&wts.unsqueeze(2)?)?.sum(1)?;                          // [B*P, C]
            feats.push(f.reshape((b, p, c))?);
        }
        Ok(Tensor::cat(&feats.iter().collect::<Vec<_>>(), 2)?)
    }

    /// Fourier features of coordinates `xy` `[B, P, 2]` in -1..1 (x across the raster's width, y down
    /// its height): `[B, P, 2 + 4 * octaves]`.
    fn features(&self, xy: &Tensor) -> Result<Tensor> {
        let o = self.cfg.octaves;
        let aspect = Tensor::new(&[self.cfg.aspect() as f32, 1f32], &self.dev)?;
        let xy = xy.broadcast_mul(&aspect)?;
        let freqs: Vec<f32> = (0..o).map(|k| (2f32).powi(k as i32) * std::f32::consts::PI).collect();
        let freqs = Tensor::from_vec(freqs, (1, 1, o, 1), &self.dev)?;
        let ang = xy.unsqueeze(2)?.broadcast_mul(&freqs)?; // [B, P, O, 2]
        let (b, p) = (xy.dim(0)?, xy.dim(1)?);
        let sin = ang.sin()?.reshape((b, p, 2 * o))?;
        let cos = ang.cos()?.reshape((b, p, 2 * o))?;
        Ok(Tensor::cat(&[&xy, &sin, &cos], 2)?)
    }

    /// `x` `[B, state_dim]`, `xy` `[B, P, 2]` (+ grid taps) -> `[B, P, 3]`, unclamped.
    fn forward(&self, x: &Tensor, xy: &Tensor, taps: &[(Tensor, Tensor)]) -> Result<Tensor> {
        let c = self.cond[1].fwd(&self.cond[0].fwd(x, false)?.gelu_erf()?, false)?.gelu_erf()?; // [B, cond]
        let mut f = self.features(xy)?;
        if self.cfg.is_grid() {
            let (b, p) = (xy.dim(0)?, xy.dim(1)?);
            f = Tensor::cat(&[&f, &self.grid_features(taps, b, p)?], 2)?;
        }
        let mut h = self.inp.fwd_nd(&f)?.gelu_erf()?; // [B, P, hidden]
        let nh = self.cfg.hidden;
        for (lin, film) in self.layers.iter().zip(&self.film) {
            let gb = film.fwd(&c, false)?; // [B, 2 * hidden]
            let g = gb.narrow(1, 0, nh)?.unsqueeze(1)?; // [B, 1, hidden]
            let bb = gb.narrow(1, nh, nh)?.unsqueeze(1)?;
            h = lin.fwd_nd(&h)?.gelu_erf()?.broadcast_mul(&(g + 1.0)?)?.broadcast_add(&bb)?;
        }
        self.out.fwd_nd(&h)
    }

    /// The state's share of the work, done once per state on the CPU: the FiLM scale and shift of
    /// every layer, `[layers][2 * hidden]` flattened (`g_0, b_0, g_1, b_1, ...`). The page's fragment
    /// shader does the rest per pixel, so the state encoding never has to exist in JavaScript.
    pub fn film(&self, s: &State) -> Result<Vec<f32>> {
        let x = self.encode(std::slice::from_ref(s))?;
        let c = self.cond[1].fwd(&self.cond[0].fwd(&x, false)?.gelu_erf()?, false)?.gelu_erf()?;
        let mut out = Vec::with_capacity(self.cfg.layers * 2 * self.cfg.hidden);
        for film in &self.film {
            out.extend(film.fwd(&c, false)?.flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(out)
    }

    fn encode(&self, states: &[State]) -> Result<Tensor> {
        let mut xs = Vec::with_capacity(states.len() * self.cfg.state_dim);
        for s in states {
            xs.extend(self.enc.state(&self.spec, s)?);
        }
        Ok(Tensor::from_vec(xs, (states.len(), self.cfg.state_dim), &self.dev)?)
    }

    /// Pixel-centre coordinates of a `w` x `h` raster, row-major: `[w * h, 2]`.
    pub fn grid(w: usize, h: usize) -> Vec<f32> {
        let mut g = Vec::with_capacity(2 * w * h);
        for j in 0..h {
            let y = 2.0 * (j as f32 + 0.5) / h as f32 - 1.0;
            for i in 0..w {
                g.push(2.0 * (i as f32 + 0.5) / w as f32 - 1.0);
                g.push(y);
            }
        }
        g
    }

    /// The shader's picture of one state at any resolution: RGB floats in CHW order, 0..1.
    pub fn draw(&self, s: &State, w: usize, h: usize) -> Result<Vec<f32>> {
        let x = self.encode(std::slice::from_ref(s))?;
        let n = w * h;
        let mut hwc = Vec::with_capacity(3 * n);
        let grid = Self::grid(w, h);
        let chunk = 16384;
        for start in (0..n).step_by(chunk) {
            let len = chunk.min(n - start);
            let (xy, taps) = self.inputs(&grid[2 * start..2 * (start + len)], 1)?;
            hwc.extend(self.forward(&x, &xy, &taps)?.clamp(0f32, 1f32)?.flatten_all()?.to_vec1::<f32>()?);
        }
        let mut chw = vec![0f32; 3 * n];
        for i in 0..n {
            for ch in 0..3 {
                chw[ch * n + i] = hwc[3 * i + ch];
            }
        }
        Ok(chw)
    }

    /// Box-filters a CHW raster of `w` x `h` down by `k`: what a display does when it shows the
    /// shader's picture smaller than it was drawn (supersampling).
    pub fn downsample(px: &[f32], w: usize, h: usize, k: usize) -> Vec<f32> {
        let (ow, oh) = (w / k, h / k);
        let mut out = vec![0f32; 3 * ow * oh];
        let inv = 1.0 / (k * k) as f32;
        for ch in 0..3 {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0f32;
                    for dy in 0..k {
                        for dx in 0..k {
                            acc += px[ch * w * h + (oy * k + dy) * w + ox * k + dx];
                        }
                    }
                    out[ch * ow * oh + oy * ow + ox] = acc * inv;
                }
            }
        }
        out
    }

    /// Mean absolute pixel error against the oracle, over `states`, with the worst state: at the
    /// training resolution (`mae`); at the scene's own raster, point-sampled (`mae_native_point`,
    /// which aliases thin lines, as any point sampler does); and at the scene's raster from the
    /// training-resolution picture box-filtered down (`mae_native`), which is how a display shows it.
    pub fn evaluate(&self, states: &[State]) -> Result<J> {
        let (w, h) = (self.cfg.w, self.cfg.h);
        let (nw, nh) = self.spec.scene.as_ref().context("no scene")?.raster_size();
        let k = (w / nw).max(1);
        let (mut total, mut total_point, mut total_ss) = (0.0f64, 0.0f64, 0.0f64);
        let mut worst = (0.0f32, String::new());
        for s in states {
            let big = self.draw(s, w, h)?;
            let e = render::mae(&big, &render::rasterize(&self.spec, s, w, h)?);
            total += f64::from(e);
            let truth = render::rasterize(&self.spec, s, nw, nh)?;
            total_point += f64::from(render::mae(&self.draw(s, nw, nh)?, &truth));
            if w == nw * k && h == nh * k {
                total_ss += f64::from(render::mae(&Self::downsample(&big, w, h, k), &truth));
            }
            if e > worst.0 {
                worst = (e, self.spec.state_key(s));
            }
        }
        let n = states.len().max(1) as f64;
        Ok(json!({"n": states.len(), "mae": total / n, "mae_native": total_ss / n, "mae_native_point": total_point / n,
                  "worst_mae": worst.0, "worst_state": worst.1, "w": w, "h": h, "native_w": nw, "native_h": nh, "supersample": k}))
    }

    pub fn describe(&self) -> J {
        json!({
            "kind": "shader", "variant": self.cfg.kind, "grid_levels": self.cfg.grid_levels, "grid_channels": self.cfg.grid_channels,
            "w": self.cfg.w, "h": self.cfg.h, "octaves": self.cfg.octaves, "hidden": self.cfg.hidden, "layers": self.cfg.layers, "cond": self.cfg.cond,
            "params": self.store.count_trainable(),
            "device": if self.dev.is_cuda() { "cuda" } else { "cpu" },
            "trained": self.meta.as_ref().map(|m| json!({"eval": m["eval"], "train": m["train"]})),
        })
    }

    pub fn save(&self, dir: &Path, t: &ShaderTrainConfig, report: &J) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let tensors: HashMap<String, Tensor> = self.store.vars.iter().map(|(n, v, _)| (n.clone(), v.as_tensor().clone())).collect();
        candle_core::safetensors::save(&tensors, dir.join(WEIGHTS_FILE))?;
        let meta = json!({"spec": self.spec.name, "kind": "shader", "shader": self.cfg, "train": t, "eval": report});
        std::fs::write(dir.join(META_FILE), serde_json::to_string_pretty(&meta)?)?;
        Ok(())
    }

    pub fn load(spec: Spec, dir: &Path, dev: &Device) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join(META_FILE)).with_context(|| format!("reading {}", dir.join(META_FILE).display()))?;
        let weights = candle_core::safetensors::load(dir.join(WEIGHTS_FILE), dev)?;
        Self::from_parts(spec, &text, weights, dev)
    }

    pub fn from_bytes(spec: Spec, meta_json: &str, weights: &[u8], dev: &Device) -> Result<Self> {
        Self::from_parts(spec, meta_json, candle_core::safetensors::load_buffer(weights, dev)?, dev)
    }

    fn from_parts(spec: Spec, meta_json: &str, weights: HashMap<String, Tensor>, dev: &Device) -> Result<Self> {
        let meta: J = serde_json::from_str(meta_json)?;
        ensure!(meta["spec"] == spec.name && meta["kind"] == "shader", "the checkpoint is not a shader of {}", spec.name);
        let cfg: ShaderConfig = serde_json::from_value(meta["shader"].clone())?;
        let mut p = Self::new(spec, cfg, dev)?;
        for (name, var, _) in &p.store.vars {
            let w = weights.get(name).with_context(|| format!("checkpoint is missing {name}"))?;
            var.set(w)?;
        }
        p.meta = Some(meta);
        Ok(p)
    }

    /// The weights as the page's fragment shader wants them: every tensor flattened row-major, with
    /// its shape, so a texture can hold it.
    pub fn export_json(&self) -> Result<J> {
        let mut t = serde_json::Map::new();
        for (n, v, _) in &self.store.vars {
            if n.starts_with("cond.") || n.starts_with("film.") {
                continue; // the state's share stays in the engine (`film`); the page gets the per-pixel layers
            }
            let x = v.as_tensor();
            t.insert(n.clone(), json!({"shape": x.dims(), "data": x.flatten_all()?.to_vec1::<f32>()?}));
        }
        let scene = self.spec.scene.as_ref().context("no scene")?;
        Ok(json!({"shader": self.cfg, "tensors": t, "view": render::raster_view(scene, self.cfg.w, self.cfg.h), "aspect": self.cfg.aspect()}))
    }
}

/// Trains the shader of `spec` on the render oracle: every valid state (or a sample), rasterised at
/// `scale` times the scene's raster, random pixels each step, mean squared plus mean absolute error.
pub fn train(spec: &Spec, cfg: &ShaderTrainConfig, dev: &Device) -> Result<(Shader, J)> {
    train_with(spec, cfg, ShaderConfig::for_spec(spec, cfg.scale)?, dev)
}

pub fn train_with(spec: &Spec, cfg: &ShaderTrainConfig, sc: ShaderConfig, dev: &Device) -> Result<(Shader, J)> {
    let t0 = std::time::Instant::now();
    let (w, h) = (sc.w, sc.h);
    let states = training_states(spec, cfg.samples, cfg.seed)?;
    ensure!(!states.is_empty(), "no states to draw");
    let mut targets = Vec::with_capacity(states.len()); // CHW rasters on the CPU
    for s in &states {
        targets.push(render::rasterize(spec, s, w, h)?);
    }
    let model = Shader::new(spec.clone(), sc, dev)?;
    let inputs = model.encode(&states)?;
    let mut opt = AdamW::new(model.store.trainable(), ParamsAdamW { lr: cfg.lr, weight_decay: 0.0, ..Default::default() })?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let (n, p, npx) = (states.len(), cfg.pixels, w * h);
    let b = cfg.batch_states.min(n).max(1);
    let mut last = f32::NAN;
    for step in 0..cfg.steps {
        // cosine schedule, as the Python reference trains it
        let lr = cfg.lr * 0.02 + (cfg.lr - cfg.lr * 0.02) * 0.5 * (1.0 + (std::f64::consts::PI * step as f64 / cfg.steps.max(1) as f64).cos());
        opt.set_learning_rate(lr);
        let mut idx = Vec::with_capacity(b);
        let mut xy = Vec::with_capacity(2 * b * p);
        let mut y = Vec::with_capacity(3 * b * p);
        for _ in 0..b {
            let si = rng.gen_range(0..n);
            idx.push(si as u32);
            let t = &targets[si];
            for _ in 0..p {
                let k = rng.gen_range(0..npx);
                let (i, j) = (k % w, k / w);
                xy.push(2.0 * (i as f32 + 0.5) / w as f32 - 1.0);
                xy.push(2.0 * (j as f32 + 0.5) / h as f32 - 1.0);
                y.extend([t[k], t[npx + k], t[2 * npx + k]]);
            }
        }
        let x = inputs.index_select(&Tensor::new(idx.as_slice(), dev)?, 0)?;
        let (xy, taps) = model.inputs(&xy, b)?;
        let y = Tensor::from_vec(y, (b, p, 3), dev)?;
        let diff = (model.forward(&x, &xy, &taps)? - &y)?;
        let loss = (diff.sqr()?.mean_all()? + (diff.abs()?.mean_all()? * 0.3)?)?;
        opt.backward_step(&loss)?;
        last = loss.to_scalar::<f32>()?;
        if cfg.log_every > 0 && (step + 1) % cfg.log_every == 0 {
            println!("step {:>6}  loss {:.5}  lr {:.2e}", step + 1, last, lr);
        }
    }
    let mut report = model.evaluate(&states)?;
    report["loss"] = json!(last);
    report["seconds"] = json!(t0.elapsed().as_secs_f64());
    report["trained_on"] = json!(if dev.is_cuda() { "cuda" } else { "cpu" });
    report["params"] = json!(model.store.count_trainable());
    report["steps"] = json!(cfg.steps);
    Ok((model, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_shader_trains_and_round_trips() -> Result<()> {
        let spec = Spec::parse(TOGGLE_WITH_SCENE)?;
        let cfg = ShaderTrainConfig { steps: 250, batch_states: 2, pixels: 1024, lr: 3e-3, log_every: 0, scale: 1, ..Default::default() };
        let mut sc = ShaderConfig::grid_for_spec(&spec, 1, &[8, 32], 8)?;
        sc.w = 64;
        sc.h = 32;
        sc.cond = 32;
        let (sh, report) = train_with(&spec, &cfg, sc, &Device::Cpu)?;
        assert!(report["mae"].as_f64().unwrap() < 0.12, "the grid shader should get close in 250 steps: {report}");
        let states = training_states(&spec, 0, 0)?;
        let a = sh.draw(&states[0], 64, 32)?;
        let dir = std::env::temp_dir().join(format!("nerve-grid-shader-test-{}", std::process::id()));
        sh.save(&dir, &cfg, &report)?;
        let back = Shader::load(spec, &dir, &Device::Cpu)?;
        assert!(back.cfg.is_grid() && back.cfg.grid_levels.len() == 2);
        assert_eq!(back.draw(&states[0], 64, 32)?, a);
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    const TOGGLE_WITH_SCENE: &str = r#"
nerve: 1
name: toggle_sh
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
    fn a_two_state_shader_trains_draws_at_two_sizes_and_round_trips() -> Result<()> {
        let spec = Spec::parse(TOGGLE_WITH_SCENE)?;
        let cfg = ShaderTrainConfig { steps: 250, batch_states: 2, pixels: 1024, lr: 2e-3, log_every: 0, scale: 1, ..Default::default() };
        let small = ShaderConfig { w: 64, h: 32, state_dim: Encoder::new(&spec)?.state_dim, cond: 32, hidden: 64, layers: 3, octaves: 6 };
        let (sh, report) = train_with(&spec, &cfg, small, &Device::Cpu)?;
        let mae = report["mae"].as_f64().unwrap();
        assert!(mae < 0.12, "the shader should get close in 250 steps: {report}");
        let states = training_states(&spec, 0, 0)?;
        let (a, b) = (sh.draw(&states[0], 64, 32)?, sh.draw(&states[1], 64, 32)?);
        assert!(render::mae(&a, &b) > 0.01, "the two states should draw differently");
        // any resolution: the 4x picture is the same picture, not a stretched one
        let big = sh.draw(&states[1], 256, 128)?;
        assert_eq!(big.len(), 3 * 256 * 128);
        let dir = std::env::temp_dir().join(format!("nerve-shader-test-{}", std::process::id()));
        sh.save(&dir, &cfg, &report)?;
        let back = Shader::load(spec.clone(), &dir, &Device::Cpu)?;
        assert_eq!(back.draw(&states[0], 64, 32)?, a);
        let meta = std::fs::read_to_string(dir.join(META_FILE))?;
        let bytes = std::fs::read(dir.join(WEIGHTS_FILE))?;
        assert_eq!(Shader::from_bytes(spec, &meta, &bytes, &Device::Cpu)?.draw(&states[1], 64, 32)?, b);
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }
}
