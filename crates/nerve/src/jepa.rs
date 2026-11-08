//! Action-conditioned JEPA whose prediction error is the energy, plus the
//! equivariant factored head (port of `nerve/model/jepa.py`).
//!
//! ```text
//! z = f(s)   z' = f_ema(s')   ẑ' = g(z, a)   E(s, a, s') = ||ẑ' - z'||² + w_fact · E_fact
//! ```
//!
//! Parameter names follow the PyTorch `state_dict` so checkpoints export to
//! the same JSON the browser and Worker runtimes read.

use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor, Var, D};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JepaConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub d: usize,
    pub hidden: usize,
    pub ema: f64,
    pub tau: f64,
    pub margin: f64,
    pub var_weight: f64,
    pub action_cont_dim: usize,
    pub ord_dim: usize,
    pub ord_weight: f64,
    /// `(offset, n, ordinal)` per discrete field.
    pub fields: Vec<(usize, usize, bool)>,
    pub fact_weight: f64,
}

impl JepaConfig {
    pub fn new(state_dim: usize, action_dim: usize) -> Self {
        Self {
            state_dim,
            action_dim,
            d: 64,
            hidden: 128,
            ema: 0.99,
            tau: 0.1,
            margin: 1.0,
            var_weight: 0.05,
            action_cont_dim: 0,
            ord_dim: 0,
            ord_weight: 1.0,
            fields: Vec::new(),
            fact_weight: 10.0,
        }
    }
}

/// Named parameters, trainable or not.
#[derive(Default)]
pub struct Store {
    pub vars: Vec<(String, Var, bool)>,
}

impl Store {
    pub(crate) fn add(&mut self, name: String, t: Tensor, trainable: bool) -> Result<Var> {
        let v = Var::from_tensor(&t)?;
        self.vars.push((name, v.clone(), trainable));
        Ok(v)
    }

    pub fn trainable(&self) -> Vec<Var> {
        self.vars.iter().filter(|(_, _, t)| *t).map(|(_, v, _)| v.clone()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&Var> {
        self.vars.iter().find(|(n, _, _)| n == name).map(|(_, v, _)| v)
    }

    pub fn count_trainable(&self) -> usize {
        self.vars.iter().filter(|(_, _, t)| *t).map(|(_, v, _)| v.elem_count()).sum()
    }
}

#[derive(Clone)]
pub(crate) struct Lin {
    w: Var,
    b: Option<Var>,
}

impl Lin {
    /// PyTorch's default `nn.Linear` init: U(±1/sqrt(fan_in)) for weight and bias.
    pub(crate) fn new(st: &mut Store, name: &str, nin: usize, nout: usize, bias: bool, dev: &Device, trainable: bool) -> Result<Self> {
        let r = 1.0 / (nin.max(1) as f64).sqrt();
        let w = st.add(format!("{name}.weight"), Tensor::rand(-r as f32, r as f32, (nout, nin), dev)?, trainable)?;
        let b = if bias {
            Some(st.add(format!("{name}.bias"), Tensor::rand(-r as f32, r as f32, nout, dev)?, trainable)?)
        } else {
            None
        };
        Ok(Self { w, b })
    }

    pub(crate) fn fwd(&self, x: &Tensor, detach: bool) -> Result<Tensor> {
        let w = if detach { self.w.as_tensor().detach() } else { self.w.as_tensor().clone() };
        let y = x.matmul(&w.t()?)?;
        Ok(match &self.b {
            Some(b) => y.broadcast_add(&if detach { b.as_tensor().detach() } else { b.as_tensor().clone() })?,
            None => y,
        })
    }
}

/// `Linear -> GELU -> Linear -> GELU -> Linear`, named like `nn.Sequential` (`.0`, `.2`, `.4`).
#[derive(Clone)]
struct Mlp {
    l: [Lin; 3],
}

impl Mlp {
    fn new(st: &mut Store, name: &str, i: usize, h: usize, o: usize, dev: &Device, trainable: bool) -> Result<Self> {
        Ok(Self {
            l: [
                Lin::new(st, &format!("{name}.0"), i, h, true, dev, trainable)?,
                Lin::new(st, &format!("{name}.2"), h, h, true, dev, trainable)?,
                Lin::new(st, &format!("{name}.4"), h, o, true, dev, trainable)?,
            ],
        })
    }

    fn fwd(&self, x: &Tensor, detach: bool) -> Result<Tensor> {
        let h = self.l[0].fwd(x, detach)?.gelu_erf()?;
        let h = self.l[1].fwd(&h, detach)?.gelu_erf()?;
        self.l[2].fwd(&h, detach)
    }
}

struct FieldHead {
    n: usize,
    ordinal: bool,
    net0: Lin,
    net2: Lin,
    own: Lin,
    mask: Tensor,
}

/// Per-field next-value log-probabilities, equivariant by construction: the
/// shift and absolute terms cannot see the field's own value, so "+1"
/// learned at 3 is the same function at 8.
struct FactoredHead {
    fields: Vec<FieldHead>,
    own_scale: f64,
}

/// Per-batch inputs the factored head needs.
pub struct FactInputs<'a> {
    /// `[B, F]` current domain index of each field.
    pub cur_idx: &'a Tensor,
    /// Per ordinal field `f`, `[B, n_f]` gather indices `j - cur + n - 1` into the shift logits.
    pub shift_idx: &'a [Option<Tensor>],
}

pub struct Jepa {
    pub cfg: JepaConfig,
    pub store: Store,
    enc: Mlp,
    tgt: Mlp,
    act: Var,
    act_cont: Option<Lin>,
    pred: Mlp,
    ord_head: Option<Lin>,
    fact: Option<FactoredHead>,
    pub device: Device,
}

/// Squared Euclidean distances `[N, M]` without a square root.
pub fn sqdist(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let aa = a.sqr()?.sum_keepdim(1)?;
    let bb = b.sqr()?.sum_keepdim(1)?.t()?;
    let ab = a.matmul(&b.t()?)?;
    Ok(aa.broadcast_add(&bb)?.broadcast_sub(&(ab * 2.0)?)?.relu()?)
}

impl Jepa {
    /// `masks` is `[F, state_dim]` from the encoder.
    pub fn new(cfg: JepaConfig, masks: &[f32], dev: &Device) -> Result<Self> {
        let mut st = Store::default();
        let (s, d, h) = (cfg.state_dim, cfg.d, cfg.hidden);
        let enc = Mlp::new(&mut st, "enc", s, h, d, dev, true)?;
        let act = st.add("act.weight".into(), Tensor::randn(0f32, 1.0, (cfg.action_dim, d), dev)?, true)?;
        let act_cont = if cfg.action_cont_dim > 0 {
            Some(Lin::new(&mut st, "act_cont", cfg.action_cont_dim, d, false, dev, true)?)
        } else {
            None
        };
        let pred = Mlp::new(&mut st, "pred", 2 * d, h, d, dev, true)?;
        let ord_head = if cfg.ord_dim > 0 { Some(Lin::new(&mut st, "ord_head", d, cfg.ord_dim, true, dev, true)?) } else { None };
        let fact = if !cfg.fields.is_empty() && cfg.fact_weight > 0.0 {
            ensure!(masks.len() == cfg.fields.len() * s, "field masks do not match the config");
            let mut fields = Vec::new();
            for (f, &(_, n, ordinal)) in cfg.fields.iter().enumerate() {
                let out = n + if ordinal { 2 * n - 1 } else { 0 };
                let net0 = Lin::new(&mut st, &format!("fact.net.{f}.0"), d + s, 64, true, dev, true)?;
                let net2 = Lin::new(&mut st, &format!("fact.net.{f}.2"), 64, out, true, dev, true)?;
                let own = Lin::new(&mut st, &format!("fact.own.{f}"), d, n, true, dev, true)?;
                let mask = Tensor::from_slice(&masks[f * s..(f + 1) * s], (1, s), dev)?;
                fields.push(FieldHead { n, ordinal, net0, net2, own, mask });
            }
            st.add("fact.masks".into(), Tensor::from_slice(masks, (cfg.fields.len(), s), dev)?, false)?;
            Some(FactoredHead { fields, own_scale: 0.1 })
        } else {
            None
        };
        // The target encoder starts as a copy of the online one and only moves by EMA.
        let tgt = Mlp::new(&mut st, "tgt", s, h, d, dev, false)?;
        let jepa = Self { cfg, store: st, enc, tgt, act, act_cont, pred, ord_head, fact, device: dev.clone() };
        jepa.copy_target()?;
        Ok(jepa)
    }

    fn copy_target(&self) -> Result<()> {
        for (t, o) in self.tgt.l.iter().zip(&self.enc.l) {
            t.w.set(o.w.as_tensor())?;
            if let (Some(tb), Some(ob)) = (&t.b, &o.b) {
                tb.set(ob.as_tensor())?;
            }
        }
        Ok(())
    }

    pub fn ema_update(&self) -> Result<()> {
        let m = self.cfg.ema;
        let mix = |t: &Var, o: &Var| -> Result<()> {
            let next = ((t.as_tensor() * m)? + (o.as_tensor().detach() * (1.0 - m))?)?;
            Ok(t.set(&next)?)
        };
        for (t, o) in self.tgt.l.iter().zip(&self.enc.l) {
            mix(&t.w, &o.w)?;
            if let (Some(tb), Some(ob)) = (&t.b, &o.b) {
                mix(tb, ob)?;
            }
        }
        Ok(())
    }

    pub fn has_fact(&self) -> bool {
        self.fact.is_some()
    }

    pub fn action_embed(&self, a_idx: &Tensor, a_cont: Option<&Tensor>) -> Result<Tensor> {
        let za = self.act.as_tensor().index_select(a_idx, 0)?;
        Ok(match (&self.act_cont, a_cont) {
            (Some(l), Some(c)) => (za + l.fwd(c, false)?)?,
            _ => za,
        })
    }

    pub fn encode(&self, s: &Tensor) -> Result<Tensor> {
        self.enc.fwd(s, false)
    }

    pub fn predict(&self, s: &Tensor, a_idx: &Tensor, a_cont: Option<&Tensor>) -> Result<Tensor> {
        let z = self.enc.fwd(s, false)?;
        self.pred.fwd(&Tensor::cat(&[&z, &self.action_embed(a_idx, a_cont)?], 1)?, false)
    }

    /// Target-encoder latent; no gradient flows into the target.
    pub fn target(&self, x: &Tensor) -> Result<Tensor> {
        self.tgt.fwd(x, true)
    }

    fn fact_logprobs(&self, za: &Tensor, x: &Tensor, zh: &Tensor, fi: &FactInputs) -> Result<Vec<Tensor>> {
        let Some(fact) = &self.fact else { return Ok(Vec::new()) };
        let mut out = Vec::with_capacity(fact.fields.len());
        for (f, fh) in fact.fields.iter().enumerate() {
            let masked = x.broadcast_mul(&fh.mask)?;
            let h = fh.net2.fwd(&fh.net0.fwd(&Tensor::cat(&[za, &masked], 1)?, false)?.gelu_erf()?, false)?;
            let mut lg = (h.narrow(1, 0, fh.n)? + (fh.own.fwd(zh, false)? * fact.own_scale)?)?;
            if fh.ordinal {
                let idx = fi.shift_idx[f].as_ref().ok_or_else(|| anyhow::anyhow!("missing shift index for field {f}"))?;
                lg = (lg + h.narrow(1, fh.n, 2 * fh.n - 1)?.contiguous()?.gather(idx, 1)?)?;
            }
            out.push(candle_nn::ops::log_softmax(&lg, D::Minus1)?);
        }
        Ok(out)
    }

    /// `[B, C]` energies of every candidate: JEPA prediction error plus the factored term.
    pub fn energy_all(
        &self,
        s: &Tensor,
        a_idx: &Tensor,
        cand: &Tensor,
        a_cont: Option<&Tensor>,
        fact: Option<(&FactInputs, &Tensor)>,
    ) -> Result<Tensor> {
        self.energy_all_z(s, a_idx, &self.target(cand)?, a_cont, fact)
    }

    /// `energy_all` with the candidates' target latents `zc` already computed: they do not change
    /// between calls once the target encoder stops moving, so a served model encodes them once.
    pub fn energy_all_z(
        &self,
        s: &Tensor,
        a_idx: &Tensor,
        zc: &Tensor,
        a_cont: Option<&Tensor>,
        fact: Option<(&FactInputs, &Tensor)>,
    ) -> Result<Tensor> {
        let zh = self.predict(s, a_idx, a_cont)?;
        let mut e = sqdist(&zh, zc)?;
        if let (Some((fi, cand_idx)), true) = (fact, self.fact.is_some()) {
            let za = self.action_embed(a_idx, a_cont)?;
            for (f, lp) in self.fact_logprobs(&za, s, &zh, fi)?.iter().enumerate() {
                let col = cand_idx.narrow(1, f, 1)?.squeeze(1)?.contiguous()?;
                e = (e - (lp.index_select(&col, 1)? * self.cfg.fact_weight)?)?;
            }
        }
        Ok(e)
    }

    /// Training loss for finite specs. `next_idx` is `[B, F]` (the true successor's field indices).
    #[allow(clippy::too_many_arguments)]
    pub fn loss(
        &self,
        s: &Tensor,
        a_idx: &Tensor,
        pos_idx: &Tensor,
        cand: &Tensor,
        neg: Option<(&Tensor, &Tensor, &Tensor)>,
        ord: Option<(&Tensor, &Tensor)>,
        fact: Option<(&FactInputs, &Tensor)>,
    ) -> Result<(Tensor, Vec<(&'static str, Tensor)>)> {
        let mut parts = Vec::new();
        let zh = self.predict(s, a_idx, None)?;
        let e = sqdist(&zh, &self.target(cand)?)?; // JEPA term only, as in the reference
        let nce = candle_nn::loss::cross_entropy(&(e.neg()? / self.cfg.tau)?, pos_idx)?;
        let e_pos = e.gather(&pos_idx.unsqueeze(1)?, 1)?.squeeze(1)?;
        let jepa = e_pos.mean_all()?;
        let mut total = (&nce + (&jepa * 0.1)?)?;
        parts.push(("nce", nce));
        parts.push(("jepa", jepa));
        if let Some((neg_next, neg_s, neg_a)) = neg {
            let zn = self.predict(neg_s, neg_a, None)?;
            let e_neg = (zn - self.target(neg_next)?)?.sqr()?.sum(1)?;
            // Negatives are paired with positives by (s, a): relu(margin + mean(e_pos) - e_neg).
            let hinge = ((e_pos.mean_all()?.broadcast_sub(&e_neg)? + self.cfg.margin)?.relu()?).mean_all()?;
            total = (total + &hinge)?;
            parts.push(("hinge", hinge));
        }
        let z = self.encode(s)?;
        let b = z.dim(0)?;
        if b > 1 {
            // Unbiased variance over the batch; one sample would give NaN and poison the run.
            let mean = z.mean_keepdim(0)?;
            let var = (z.broadcast_sub(&mean)?.sqr()?.sum(0)? / (b - 1) as f64)?;
            let reg = ((var + 1e-4)?.sqrt()?.neg()? + 1.0)?.relu()?.mean_all()?;
            total = (total + (&reg * self.cfg.var_weight)?)?;
            parts.push(("var", reg));
        }
        if let (Some(head), Some((ord_s, ord_next))) = (&self.ord_head, ord) {
            let aux = ((head.fwd(&zh, false)? - ord_next)?.sqr()?.mean_all()? + (head.fwd(&z, false)? - ord_s)?.sqr()?.mean_all()?)?;
            total = (total + (&aux * self.cfg.ord_weight)?)?;
            parts.push(("ord", aux));
        }
        if let (Some(_), Some((fi, next_idx))) = (&self.fact, fact) {
            let za = self.action_embed(a_idx, None)?;
            let mut fl = Tensor::zeros((), DType::F32, &self.device)?;
            for (f, lp) in self.fact_logprobs(&za, s, &zh, fi)?.iter().enumerate() {
                let tgt = next_idx.narrow(1, f, 1)?.contiguous()?;
                fl = (fl - lp.gather(&tgt, 1)?.mean_all()?)?;
            }
            total = (total + (&fl * self.cfg.fact_weight)?)?;
            parts.push(("fact", fl));
        }
        Ok((total, parts))
    }
}

impl Jepa {
    /// Energy of candidate states `x` `[N, state_dim]` against fixed predictions `zh` `[N, d]`,
    /// with the gradient flowing into `x` (descent over continuous fields) but not the target.
    pub fn energy_to(&self, zh: &Tensor, x: &Tensor) -> Result<Tensor> {
        Ok(zh.broadcast_sub(&self.target(x)?)?.sqr()?.sum(1)?)
    }

    /// Variance floor on the online latents, so the encoder cannot collapse to a point.
    fn var_reg(&self, s: &Tensor) -> Result<Option<Tensor>> {
        let z = self.encode(s)?;
        let b = z.dim(0)?;
        if b < 2 {
            return Ok(None); // one sample would give NaN and poison the run
        }
        let mean = z.mean_keepdim(0)?;
        let var = (z.broadcast_sub(&mean)?.sqr()?.sum(0)? / (b - 1) as f64)?;
        Ok(Some(((var + 1e-4)?.sqrt()?.neg()? + 1.0)?.relu()?.mean_all()?))
    }

    /// Continuous specs have no finite candidate set: the true successor is contrasted with
    /// the batch's other successors and `K` near-miss perturbations, `negs` `[B, K, state_dim]`
    /// (port of `JEPA.loss_continuous`).
    pub fn loss_continuous(
        &self,
        s: &Tensor,
        a_idx: &Tensor,
        a_cont: Option<&Tensor>,
        s_next: &Tensor,
        negs: &Tensor,
    ) -> Result<(Tensor, Vec<(&'static str, Tensor)>)> {
        let zh = self.predict(s, a_idx, a_cont)?;
        let zp = self.target(s_next)?;
        let (b, k, d) = negs.dims3()?;
        let zn = self.target(&negs.reshape((b * k, d))?)?.reshape((b, k, self.cfg.d))?;
        let e_pos = (&zh - &zp)?.sqr()?.sum(1)?;
        let e_batch = sqdist(&zh, &zp)?; // [B, B], the diagonal is e_pos
        let e_neg = zh.unsqueeze(1)?.broadcast_sub(&zn)?.sqr()?.sum(2)?; // [B, K]
        let logits = (Tensor::cat(&[&e_batch, &e_neg], 1)?.neg()? / self.cfg.tau)?;
        let target = Tensor::arange(0u32, b as u32, &self.device)?;
        let nce = candle_nn::loss::cross_entropy(&logits, &target)?;
        let jepa = e_pos.mean_all()?;
        let mut total = (&nce + (&jepa * 0.1)?)?;
        let mut parts = vec![("nce", nce), ("jepa", jepa)];
        if let Some(reg) = self.var_reg(s)? {
            total = (total + (&reg * self.cfg.var_weight)?)?;
            parts.push(("var", reg));
        }
        Ok((total, parts))
    }
}

/// Builds the factored head's per-field shift gather indices for a batch of current indices `[B, F]`.
pub fn shift_indices(fields: &[(usize, usize, bool)], cur: &[u32], b: usize, dev: &Device) -> Result<Vec<Option<Tensor>>> {
    let nf = fields.len();
    fields
        .iter()
        .enumerate()
        .map(|(f, &(_, n, ordinal))| {
            if !ordinal {
                return Ok(None);
            }
            let mut idx = Vec::with_capacity(b * n);
            for r in 0..b {
                let c = cur[r * nf + f] as usize;
                idx.extend((0..n).map(|j| (j + n - 1 - c) as u32));
            }
            Ok(Some(Tensor::from_vec(idx, (b, n), dev)?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqdist_matches_direct_computation() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::new(&[[0f32, 0.0], [1.0, 2.0]], &dev)?;
        let b = Tensor::new(&[[3f32, 4.0]], &dev)?;
        let d: Vec<Vec<f32>> = sqdist(&a, &b)?.to_vec2()?;
        assert!((d[0][0] - 25.0).abs() < 1e-4 && (d[1][0] - 8.0).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn continuous_loss_trains_the_online_side_only() -> Result<()> {
        let dev = Device::Cpu;
        let mut cfg = JepaConfig::new(5, 2);
        (cfg.d, cfg.hidden, cfg.action_cont_dim) = (8, 16, 1);
        let m = Jepa::new(cfg, &[], &dev)?;
        let s = Tensor::rand(0f32, 1.0, (4, 5), &dev)?;
        let next = Tensor::rand(0f32, 1.0, (4, 5), &dev)?;
        let negs = Tensor::rand(0f32, 1.0, (4, 3, 5), &dev)?;
        let a = Tensor::new(&[0u32, 1, 0, 1], &dev)?;
        let ac = Tensor::new(&[[0.1f32], [0.5], [0.9], [0.3]], &dev)?;
        let (total, parts) = m.loss_continuous(&s, &a, Some(&ac), &next, &negs)?;
        assert!(total.to_scalar::<f32>()?.is_finite());
        assert!(parts.iter().any(|(n, _)| *n == "nce"));
        let grads = total.backward()?;
        for name in ["enc.0.weight", "pred.4.weight", "act.weight", "act_cont.weight"] {
            assert!(grads.get(m.store.get(name).unwrap().as_tensor()).is_some(), "{name} gets no gradient");
        }
        assert!(grads.get(m.store.get("tgt.0.weight").unwrap().as_tensor()).is_none());
        // energy_to: gradient reaches the candidate state, not the prediction.
        let zh = m.predict(&s, &a, Some(&ac))?.detach();
        let x = candle_core::Var::from_tensor(&next)?;
        let g = m.energy_to(&zh, x.as_tensor())?.sum_all()?.backward()?;
        assert!(g.get(x.as_tensor()).is_some());
        Ok(())
    }

    #[test]
    fn loss_is_finite_and_differentiable() -> Result<()> {
        let dev = Device::Cpu;
        let mut cfg = JepaConfig::new(6, 3);
        cfg.d = 8;
        cfg.hidden = 16;
        cfg.fields = vec![(0, 4, true), (4, 2, false)];
        let masks = vec![0f32, 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.];
        let m = Jepa::new(cfg.clone(), &masks, &dev)?;
        let cand = Tensor::eye(6, DType::F32, &dev)?.narrow(0, 0, 6)?;
        let s = cand.narrow(0, 0, 3)?;
        let a = Tensor::new(&[0u32, 1, 2], &dev)?;
        let pos = Tensor::new(&[1u32, 2, 3], &dev)?;
        let cur = [0u32, 0, 1, 1, 2, 0];
        let shift = shift_indices(&cfg.fields, &cur, 3, &dev)?;
        let cur_t = Tensor::from_slice(&cur, (3, 2), &dev)?;
        let fi = FactInputs { cur_idx: &cur_t, shift_idx: &shift };
        let next = Tensor::new(&[[1u32, 0], [2, 1], [3, 0]], &dev)?;
        let (total, parts) = m.loss(&s, &a, &pos, &cand, None, None, Some((&fi, &next)))?;
        assert!(total.to_scalar::<f32>()?.is_finite());
        assert!(parts.iter().any(|(n, _)| *n == "fact"));
        let grads = total.backward()?;
        assert!(grads.get(m.store.get("enc.0.weight").unwrap().as_tensor()).is_some());
        assert!(grads.get(m.store.get("tgt.0.weight").unwrap().as_tensor()).is_none());
        Ok(())
    }
}
