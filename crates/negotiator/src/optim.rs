//! SGD with PyTorch semantics (momentum, Nesterov) plus global-norm clipping.

use anyhow::Result;
use candle_core::backprop::GradStore;
use candle_core::{Tensor, Var};

pub struct Sgd {
    vars: Vec<Var>,
    bufs: Vec<Option<Tensor>>,
    pub lr: f64,
    momentum: f64,
    nesterov: bool,
}

impl Sgd {
    pub fn new(vars: Vec<Var>, lr: f64, momentum: f64, nesterov: bool) -> Self {
        let bufs = vec![None; vars.len()];
        Self { vars, bufs, lr, momentum, nesterov: nesterov && momentum > 0.0 }
    }

    /// Global L2 norm of all gradients present in `grads`.
    pub fn grad_norm(&self, grads: &GradStore) -> Result<f64> {
        let mut total = 0f64;
        for v in &self.vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                total += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            }
        }
        Ok(total.sqrt())
    }

    /// Applies one update, first rescaling gradients so their global norm is
    /// at most `clip` (when `clip > 0`). Returns the pre-clip norm.
    pub fn step(&mut self, grads: &GradStore, clip: f64) -> Result<f64> {
        let norm = self.grad_norm(grads)?;
        if !norm.is_finite() {
            // One bad batch must not poison every parameter.
            eprintln!("warning: skipping update with non-finite gradient norm {norm}");
            return Ok(norm);
        }
        let scale = if clip > 0.0 { (clip / (norm + 1e-6)).min(1.0) } else { 1.0 };
        for (v, buf) in self.vars.iter().zip(self.bufs.iter_mut()) {
            let Some(g) = grads.get(v.as_tensor()) else { continue };
            let g = if scale < 1.0 { (g * scale)? } else { g.clone() };
            let update = if self.momentum > 0.0 {
                let b = match buf.take() {
                    Some(b) => ((b * self.momentum)? + &g)?,
                    None => g.clone(),
                };
                let u = if self.nesterov { (&g + (&b * self.momentum)?)? } else { b.clone() };
                *buf = Some(b.detach());
                u
            } else {
                g
            };
            v.set(&(v.as_tensor().detach() - (update * self.lr)?)?)?;
        }
        Ok(norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn run(momentum: f64, nesterov: bool) -> Result<Vec<f32>> {
        let x = Var::new(&[1f32], &Device::Cpu)?;
        let mut opt = Sgd::new(vec![x.clone()], 0.1, momentum, nesterov);
        let mut seen = Vec::new();
        for _ in 0..2 {
            // d/dx x^2 = 2x
            let grads = x.as_tensor().sqr()?.sum_all()?.backward()?;
            opt.step(&grads, 0.0)?;
            seen.push(x.as_tensor().to_vec1::<f32>()?[0]);
        }
        Ok(seen)
    }

    #[test]
    fn matches_torch_sgd() -> Result<()> {
        // Values from torch.optim.SGD on f(x) = x^2, x0 = 1, lr = 0.1.
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        assert!(close(&run(0.0, false)?, &[0.8, 0.64]));
        assert!(close(&run(0.5, false)?, &[0.8, 0.54]));
        assert!(close(&run(0.5, true)?, &[0.7, 0.44]));
        Ok(())
    }

    #[test]
    fn clips_to_global_norm() -> Result<()> {
        let x = Var::new(&[3f32, 4.0], &Device::Cpu)?;
        let mut opt = Sgd::new(vec![x.clone()], 1.0, 0.0, false);
        // grad = x, norm 5; clipped to norm 1 the step is x / 5.
        let grads = (x.as_tensor().sqr()?.sum_all()? * 0.5)?.backward()?;
        assert!((opt.step(&grads, 1.0)? - 5.0).abs() < 1e-6);
        let v = x.as_tensor().to_vec1::<f32>()?;
        assert!((v[0] - 2.4).abs() < 1e-5 && (v[1] - 3.2).abs() < 1e-5);
        Ok(())
    }
}
