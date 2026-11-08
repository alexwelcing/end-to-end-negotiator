//! Featurise states and actions for the network (same layout as the Python
//! `Encoder`, so exported weights are interchangeable).
//!
//! Discrete fields are one-hots; ordinal fields (ints and gridded floats with
//! more than two values) also get a normalised value plus Fourier features;
//! continuous fields get a normalised scalar plus Fourier features.

use crate::spec::{ActionInst, Kind, Spec, State};
use crate::value::py_round;
use anyhow::{Context, Result};
use std::collections::HashMap;

pub const N_FOURIER: usize = 4;
pub const CONT_WIDTH: usize = 1 + 2 * N_FOURIER;

/// One discrete field's place in the state vector.
#[derive(Debug, Clone)]
pub struct Slot {
    pub field: usize,
    pub offset: usize,
    pub n: usize,
    pub ordinal: bool,
}

pub struct Encoder {
    pub slots: Vec<Slot>,
    domains: Vec<Vec<crate::value::Value>>,
    pub disc_dim: usize,
    pub cont_fields: Vec<usize>,
    /// Indices into `slots` of ordinal fields.
    pub ord_slots: Vec<usize>,
    pub state_dim: usize,
    pub actions: Vec<ActionInst>,
    action_index: HashMap<String, usize>,
    /// `(action name, param name)` of every continuous action parameter.
    pub cont_params: Vec<(String, String)>,
    pub states: Vec<State>,
    state_index: HashMap<String, usize>,
}

pub fn cont_feats(u: f64, out: &mut Vec<f32>) {
    out.push(u as f32);
    for k in 1..=N_FOURIER {
        let a = 2.0 * std::f64::consts::PI * k as f64 * u;
        out.push(a.sin() as f32);
        out.push(a.cos() as f32);
    }
}

impl Encoder {
    pub fn new(spec: &Spec) -> Result<Self> {
        let mut slots = Vec::new();
        let mut domains = Vec::new();
        let mut off = 0;
        for (i, f) in spec.fields.iter().enumerate() {
            if f.continuous() {
                continue;
            }
            let dom = f.domain()?;
            let ordinal = matches!(f.kind, Kind::Int | Kind::Float) && dom.len() > 2;
            slots.push(Slot { field: i, offset: off, n: dom.len(), ordinal });
            off += dom.len();
            domains.push(dom);
        }
        let cont_fields = spec.continuous_fields();
        let ord_slots: Vec<usize> = (0..slots.len()).filter(|&j| slots[j].ordinal).collect();
        let state_dim = off + (cont_fields.len() + ord_slots.len()) * CONT_WIDTH;
        let actions = spec.action_instances()?;
        let action_index = actions.iter().enumerate().map(|(i, a)| (spec.action_key(a), i)).collect();
        let cont_params = spec
            .actions
            .iter()
            .flat_map(|a| a.continuous_params().into_iter().map(|p| (a.name.clone(), p.to_string())))
            .collect();
        let states = if spec.finite() {
            spec.state_space()?
        } else {
            crate::spec::product(&spec.discrete_fields().iter().map(|&i| spec.fields[i].domain()).collect::<Result<Vec<_>>>()?)?
        };
        let state_index = if spec.finite() {
            states.iter().enumerate().map(|(i, s)| (spec.state_key(s), i)).collect()
        } else {
            HashMap::new()
        };
        Ok(Self {
            slots,
            domains,
            disc_dim: off,
            cont_fields,
            ord_slots,
            state_dim,
            actions,
            action_index,
            cont_params,
            states,
            state_index,
        })
    }

    pub fn action_dim(&self) -> usize {
        self.actions.len()
    }

    /// Index of a discrete field's value within its domain.
    fn value_index(&self, spec: &Spec, slot: usize, s: &State) -> Result<usize> {
        let sl = &self.slots[slot];
        let f = &spec.fields[sl.field];
        let v = &s[sl.field];
        if f.kind == Kind::Float {
            let (lo, _) = f.range();
            let step = f.step.unwrap_or(1.0);
            return Ok(py_round((v.as_f64().unwrap_or(lo) - lo) / step) as usize);
        }
        self.domains[slot].iter().position(|d| d == v).with_context(|| format!("{} = {} is not in the domain", f.name, v.repr()))
    }

    /// Per-field domain indices (the factored head's targets and shift origin).
    pub fn field_index(&self, spec: &Spec, s: &State) -> Result<Vec<u32>> {
        (0..self.slots.len()).map(|j| self.value_index(spec, j, s).map(|i| i as u32)).collect()
    }

    /// Normalised value of each ordinal field (the ordinal readout's targets).
    pub fn ordinal_values(&self, spec: &Spec, s: &State) -> Result<Vec<f32>> {
        self.ord_slots
            .iter()
            .map(|&j| {
                let i = self.value_index(spec, j, s)?;
                let n = self.slots[j].n;
                Ok(self.ord_norm(j, i, n))
            })
            .collect()
    }

    fn ord_norm(&self, slot: usize, i: usize, n: usize) -> f32 {
        let dom = &self.domains[slot];
        let (lo, hi) = (dom[0].as_f64().unwrap_or(0.0), dom[n - 1].as_f64().unwrap_or(1.0));
        ((dom[i].as_f64().unwrap_or(lo) - lo) / (hi - lo)) as f32
    }

    pub fn state(&self, spec: &Spec, s: &State) -> Result<Vec<f32>> {
        let mut x = vec![0f32; self.disc_dim];
        let mut idx = Vec::with_capacity(self.slots.len());
        for (j, sl) in self.slots.iter().enumerate() {
            let i = self.value_index(spec, j, s)?;
            x[sl.offset + i] = 1.0;
            idx.push(i);
        }
        for &k in &self.cont_fields {
            cont_feats(spec.fields[k].norm(&s[k]), &mut x);
        }
        for &j in &self.ord_slots {
            cont_feats(f64::from(self.ord_norm(j, idx[j], self.slots[j].n)), &mut x);
        }
        Ok(x)
    }

    pub fn state_id(&self, spec: &Spec, s: &State) -> Result<usize> {
        self.state_index.get(&spec.state_key(s)).copied().with_context(|| format!("state {} is not enumerable", spec.state_key(s)))
    }

    pub fn action_id(&self, spec: &Spec, a: &ActionInst) -> Result<usize> {
        self.action_index.get(&spec.action_key(a)).copied().with_context(|| format!("unknown action instance {}", spec.action_key(a)))
    }

    /// Normalised continuous params of this action (0 for params of other actions).
    pub fn action_cont(&self, spec: &Spec, a: &ActionInst) -> Vec<f32> {
        self.cont_params
            .iter()
            .map(|(an, pk)| {
                if *an != a.name {
                    return 0.0;
                }
                match (a.param(pk), spec.action(an).and_then(|x| x.param(pk))) {
                    (Some(v), Some(p)) => p.norm(v) as f32,
                    _ => 0.0,
                }
            })
            .collect()
    }

    /// `[F, state_dim]` masks hiding each discrete field's own one-hot and ordinal features.
    pub fn field_masks(&self) -> Vec<f32> {
        let mut m = vec![1f32; self.slots.len() * self.state_dim];
        let base = self.disc_dim + self.cont_fields.len() * CONT_WIDTH;
        for (f, sl) in self.slots.iter().enumerate() {
            let row = &mut m[f * self.state_dim..(f + 1) * self.state_dim];
            row[sl.offset..sl.offset + sl.n].iter_mut().for_each(|v| *v = 0.0);
            if let Some(j) = self.ord_slots.iter().position(|&o| o == f) {
                row[base + j * CONT_WIDTH..base + (j + 1) * CONT_WIDTH].iter_mut().for_each(|v| *v = 0.0);
            }
        }
        m
    }

    /// Every enumerable state as feature rows `[C, state_dim]`.
    pub fn all_state_vectors(&self, spec: &Spec) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(self.states.len() * self.state_dim);
        for s in &self.states {
            out.extend(self.state(spec, s)?);
        }
        Ok(out)
    }

    pub fn all_field_indices(&self, spec: &Spec) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(self.states.len() * self.slots.len());
        for s in &self.states {
            out.extend(self.field_index(spec, s)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_python() {
        let sp = Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/counter.nerve")).unwrap();
        let enc = Encoder::new(&sp).unwrap();
        // count: 10 one-hot + armed: 2 one-hot, plus count's ordinal features.
        assert_eq!(enc.disc_dim, 12);
        assert_eq!(enc.state_dim, 12 + CONT_WIDTH);
        assert_eq!(enc.action_dim(), 14);
        let x = enc.state(&sp, &vec![crate::value::Value::Int(9), crate::value::Value::Bool(true)]).unwrap();
        assert_eq!(x[9], 1.0);
        assert_eq!(x[11], 1.0);
        assert_eq!(x[12], 1.0); // normalised ordinal value of count = 9
        let m = enc.field_masks();
        assert!(m[..10].iter().all(|v| *v == 0.0) && m[10] == 1.0 && m[12] == 0.0);
        assert!(m[enc.state_dim..enc.state_dim + 10].iter().all(|v| *v == 1.0));
    }
}
