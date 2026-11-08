//! Minimal layers built from raw candle ops so weights can be shared freely
//! (the reader GRU and writer GRU cell are the same parameters) and every
//! parameter is a named [`Var`] for the optimizer and checkpointing.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};

/// Named trainable parameters in registration order.
#[derive(Default, Clone)]
pub struct Params {
    vars: Vec<(String, Var)>,
}

impl Params {
    pub fn uniform(&mut self, name: &str, shape: &[usize], range: f64, device: &Device) -> Result<Var> {
        if range <= 0.0 {
            return self.zeros(name, shape, device);
        }
        let t = Tensor::rand(-range as f32, range as f32, shape, device)?;
        self.push(name, Var::from_tensor(&t)?)
    }

    pub fn zeros(&mut self, name: &str, shape: &[usize], device: &Device) -> Result<Var> {
        self.push(name, Var::zeros(shape, DType::F32, device)?)
    }

    fn push(&mut self, name: &str, var: Var) -> Result<Var> {
        anyhow::ensure!(self.vars.iter().all(|(n, _)| n != name), "duplicate parameter {name}");
        self.vars.push((name.to_string(), var.clone()));
        Ok(var)
    }

    pub fn iter(&self) -> impl Iterator<Item = &(String, Var)> {
        self.vars.iter()
    }

    pub fn vars(&self) -> Vec<Var> {
        self.vars.iter().map(|(_, v)| v.clone()).collect()
    }

    pub fn count(&self) -> usize {
        self.vars.iter().map(|(_, v)| v.elem_count()).sum()
    }

    /// Copies current values so they can be restored after further training.
    pub fn snapshot(&self) -> Result<Vec<Tensor>> {
        self.vars.iter().map(|(_, v)| Ok(v.as_tensor().copy()?)).collect()
    }

    pub fn restore(&self, values: &[Tensor]) -> Result<()> {
        anyhow::ensure!(values.len() == self.vars.len(), "snapshot does not match parameters");
        self.vars.iter().zip(values).try_for_each(|((_, v), t)| Ok(v.set(t)?))
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let map: std::collections::HashMap<String, Tensor> =
            self.vars.iter().map(|(n, v)| (n.clone(), v.as_tensor().clone())).collect();
        candle_core::safetensors::save(&map, path)?;
        Ok(())
    }

    pub fn load(&self, path: &std::path::Path, device: &Device) -> Result<()> {
        let mut map = candle_core::safetensors::load(path, device)?;
        for (name, var) in &self.vars {
            let t = map.remove(name).ok_or_else(|| anyhow::anyhow!("checkpoint lacks {name}"))?;
            anyhow::ensure!(t.dims() == var.dims(), "{name}: shape {:?} != {:?}", t.dims(), var.dims());
            var.set(&t)?;
        }
        anyhow::ensure!(map.is_empty(), "checkpoint has unknown tensors: {:?}", map.keys());
        Ok(())
    }
}

/// Sigmoid whose gradient is computed from its output, so it stays finite
/// for large pre-activations (unlike differentiating `1 / (1 + exp(-x))`).
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::sigmoid(x)?)
}

/// Looks rows up in an embedding matrix; output shape is `ids.shape + [dim]`.
pub fn embed(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let mut dims = ids.dims().to_vec();
    dims.push(weight.dim(1)?);
    Ok(weight.index_select(&ids.flatten_all()?, 0)?.reshape(dims)?)
}

pub fn dropout(x: &Tensor, p: f32, train: bool) -> Result<Tensor> {
    if !train || p <= 0.0 {
        return Ok(x.clone());
    }
    Ok(candle_nn::ops::dropout(x, p)?)
}

#[derive(Clone)]
pub struct Linear {
    w: Var,
    b: Var,
}

impl Linear {
    /// Uniform weights in `[-range, range]` and zero bias.
    pub fn new(p: &mut Params, name: &str, nin: usize, nout: usize, range: f64, dev: &Device) -> Result<Self> {
        Ok(Self {
            w: p.uniform(&format!("{name}.weight"), &[nout, nin], range, dev)?,
            b: p.zeros(&format!("{name}.bias"), &[nout], dev)?,
        })
    }

    /// `x: (N, nin) -> (N, nout)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.matmul(&self.w.t()?)?.broadcast_add(&self.b)?)
    }
}

/// A single-layer GRU with PyTorch's gate layout (`r`, `z`, `n`).
#[derive(Clone)]
pub struct Gru {
    w_ih: Var,
    w_hh: Var,
    b_ih: Var,
    b_hh: Var,
    hidden: usize,
}

impl Gru {
    /// Uniform weights in `[-range, range]`; biases zero when `zero_bias`,
    /// otherwise uniform too (PyTorch's default GRU init).
    pub fn new(
        p: &mut Params,
        name: &str,
        nin: usize,
        hidden: usize,
        range: f64,
        zero_bias: bool,
        dev: &Device,
    ) -> Result<Self> {
        let bias = |p: &mut Params, n: &str| {
            if zero_bias {
                p.zeros(n, &[3 * hidden], dev)
            } else {
                p.uniform(n, &[3 * hidden], range, dev)
            }
        };
        Ok(Self {
            w_ih: p.uniform(&format!("{name}.weight_ih"), &[3 * hidden, nin], range, dev)?,
            w_hh: p.uniform(&format!("{name}.weight_hh"), &[3 * hidden, hidden], range, dev)?,
            b_ih: bias(p, &format!("{name}.bias_ih"))?,
            b_hh: bias(p, &format!("{name}.bias_hh"))?,
            hidden,
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    fn input_proj(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.matmul(&self.w_ih.t()?)?.broadcast_add(&self.b_ih)?)
    }

    fn cell(&self, gi: &Tensor, h: &Tensor) -> Result<Tensor> {
        let n = self.hidden;
        let gh = h.matmul(&self.w_hh.t()?)?.broadcast_add(&self.b_hh)?;
        let r = sigmoid(&(gi.narrow(1, 0, n)? + gh.narrow(1, 0, n)?)?)?;
        let z = sigmoid(&(gi.narrow(1, n, n)? + gh.narrow(1, n, n)?)?)?;
        let cand = (gi.narrow(1, 2 * n, n)? + (r * gh.narrow(1, 2 * n, n)?)?)?.tanh()?;
        // (1 - z) * cand + z * h
        Ok((&cand + (z * (h - &cand)?)?)?)
    }

    /// One step: `x: (B, nin)`, `h: (B, H) -> (B, H)`.
    pub fn step(&self, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        self.cell(&self.input_proj(x)?, h)
    }

    /// Runs over `xs: (T, B, nin)` from `h0: (B, H)`, optionally right to left.
    /// Returns outputs `(T, B, H)` in input order and the final hidden state.
    pub fn forward(&self, xs: &Tensor, h0: &Tensor, reverse: bool) -> Result<(Tensor, Tensor)> {
        let (t, b, nin) = xs.dims3()?;
        let gi = self.input_proj(&xs.reshape((t * b, nin))?)?.reshape((t, b, 3 * self.hidden))?;
        let mut h = h0.clone();
        let mut outs = Vec::with_capacity(t);
        let steps: Box<dyn Iterator<Item = usize>> = if reverse { Box::new((0..t).rev()) } else { Box::new(0..t) };
        for i in steps {
            h = self.cell(&gi.get(i)?, &h)?;
            outs.push(h.clone());
        }
        if reverse {
            outs.reverse();
        }
        Ok((Tensor::stack(&outs, 0)?, h))
    }
}

/// Mean cross-entropy with per-class weights, normalised by the summed
/// weights of the targets (PyTorch's weighted `CrossEntropyLoss`).
pub fn weighted_cross_entropy(logits: &Tensor, targets: &Tensor, class_weights: &Tensor) -> Result<Tensor> {
    let logp = candle_nn::ops::log_softmax(logits, 1)?;
    let picked = logp.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?;
    let w = class_weights.index_select(targets, 0)?;
    // A batch whose targets are all zero-weight contributes no loss (not 0/0).
    let norm = w.sum_all()?.to_scalar::<f32>()?.max(f32::MIN_POSITIVE);
    Ok(((picked * &w)?.sum_all()?.neg()? / norm as f64)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gru_matches_hand_computed_step() -> Result<()> {
        let dev = Device::Cpu;
        let mut p = Params::default();
        let gru = Gru::new(&mut p, "g", 1, 1, 0.0, true, &dev)?;
        // With all-zero parameters: r = z = 0.5, cand = 0, h' = 0.5 * h.
        let h = gru.step(&Tensor::new(&[[3f32]], &dev)?, &Tensor::new(&[[2f32]], &dev)?)?;
        assert!((h.to_vec2::<f32>()?[0][0] - 1.0).abs() < 1e-6);
        let xs = Tensor::zeros((4, 2, 1), DType::F32, &dev)?;
        let (outs, last) = gru.forward(&xs, &Tensor::ones((2, 1), DType::F32, &dev)?, true)?;
        assert_eq!(outs.dims(), &[4, 2, 1]);
        // Reverse direction: index 0 is the last step processed.
        assert!((outs.get(0)?.to_vec2::<f32>()?[0][0] - 0.0625).abs() < 1e-6);
        assert!((last.to_vec2::<f32>()?[0][0] - 0.0625).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn weighted_cross_entropy_ignores_zero_weight_classes() -> Result<()> {
        let dev = Device::Cpu;
        let logits = Tensor::new(&[[0f32, 0.0], [10.0, 0.0]], &dev)?;
        let tgt = Tensor::new(&[0u32, 1], &dev)?;
        let w = Tensor::new(&[1f32, 0.0], &dev)?;
        let loss = weighted_cross_entropy(&logits, &tgt, &w)?.to_scalar::<f32>()?;
        assert!((loss - std::f32::consts::LN_2).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn snapshot_restore_round_trips() -> Result<()> {
        let dev = Device::Cpu;
        let mut p = Params::default();
        let v = p.uniform("w", &[3], 1.0, &dev)?;
        let snap = p.snapshot()?;
        let before = v.as_tensor().to_vec1::<f32>()?;
        v.set(&Tensor::zeros(3, DType::F32, &dev)?)?;
        p.restore(&snap)?;
        assert_eq!(v.as_tensor().to_vec1::<f32>()?, before);
        Ok(())
    }
}
