//! The reference interpreter derived mechanically from a spec. It is what a
//! trained network must match, and the judge of every transition.

use crate::expr::Scope;
use crate::spec::{ActionInst, Kind, Spec, State};
use crate::value::Value;
use anyhow::{Context, Result};
use blake2::digest::{Update, VariableOutput};
use rand::Rng;

/// Expression scope over a state, optional action params, and the previous state.
pub struct StateScope<'a> {
    pub spec: &'a Spec,
    pub cur: &'a [Value],
    pub params: &'a [(String, Value)],
    pub old: Option<&'a [Value]>,
}

impl Scope for StateScope<'_> {
    fn get(&self, name: &str) -> Option<Value> {
        if let Some((_, v)) = self.params.iter().find(|(k, _)| k == name) {
            return Some(v.clone());
        }
        self.spec.field_index(name).map(|i| self.cur[i].clone())
    }

    fn old(&self, name: &str) -> Option<Value> {
        let src = self.old.unwrap_or(self.cur);
        self.spec.field_index(name).map(|i| src[i].clone())
    }
}

pub struct Oracle<'a> {
    pub spec: &'a Spec,
}

impl<'a> Oracle<'a> {
    pub fn new(spec: &'a Spec) -> Self {
        Self { spec }
    }

    pub fn step(&self, state: &State, action: &ActionInst) -> Result<State> {
        let sp = self.spec;
        let t = sp.transition(&action.name).with_context(|| format!("unknown action {:?}", action.name))?;
        let scope = StateScope { spec: sp, cur: state, params: &action.params, old: Some(state) };
        if let Some(when) = &t.when {
            if !when.eval(&scope)?.truthy() {
                return Ok(state.clone());
            }
        }
        let mut next = state.clone();
        for (idx, expr) in &t.updates {
            let f = &sp.fields[*idx];
            let mut v = expr.eval(&scope)?;
            if f.kind == Kind::Enum {
                v = Value::Str(v.py_str());
            }
            next[*idx] = f.canon(&v);
        }
        Ok(next)
    }

    /// Which spec properties does `state -action-> new` break?
    ///
    /// Continuous fields within `tol` of their range of the true value are
    /// taken as the true value before `ensures` are checked.
    pub fn violations(&self, state: &State, action: Option<&ActionInst>, new: &State, tol: f64) -> Vec<String> {
        let sp = self.spec;
        let mut new = new.clone();
        let cont = sp.continuous_fields();
        if let (false, Some(a)) = (cont.is_empty(), action) {
            if let Ok(truth) = self.step(state, a) {
                for k in cont {
                    let (lo, hi) = sp.fields[k].range();
                    if let (Some(p), Some(t)) = (new[k].as_f64(), truth[k].as_f64()) {
                        if (p - t).abs() <= tol * (hi - lo) {
                            new[k] = truth[k].clone();
                        }
                    }
                }
            }
        }
        let mut bad = Vec::new();
        for (f, v) in sp.fields.iter().zip(&new) {
            if !f.contains(v) {
                bad.push(format!("type:{}={}", f.name, v.repr()));
            }
        }
        let no_params: [(String, Value); 0] = [];
        let inv_scope = StateScope { spec: sp, cur: &new, params: &no_params, old: Some(state) };
        for inv in &sp.invariants {
            match inv.eval(&inv_scope) {
                Ok(v) if v.truthy() => {}
                Ok(_) => bad.push(format!("invariant:{}", inv.source())),
                Err(e) => bad.push(format!("invariant-error:{}:{e}", inv.source())),
            }
        }
        for e in &sp.ensures {
            let applies = match (&e.after, action) {
                (None, _) => true,
                (Some(after), Some(a)) => *after == a.name,
                (Some(_), None) => false,
            };
            if !applies {
                continue;
            }
            let params = action.map_or(&no_params[..], |a| &a.params[..]);
            let scope = StateScope { spec: sp, cur: &new, params, old: Some(state) };
            match e.holds.eval(&scope) {
                Ok(v) if v.truthy() => {}
                Ok(_) => bad.push(format!("ensures:{}:{}", e.after.as_deref().unwrap_or("*"), e.holds.source())),
                Err(err) => bad.push(format!("ensures-error:{}:{err}", e.holds.source())),
            }
        }
        bad
    }

    /// True when `s` breaks one of the spec's own invariants (unreachable by construction).
    pub fn breaks_invariant(&self, s: &State) -> bool {
        self.violations(s, None, s, 0.01).first().is_some_and(|v| v.starts_with("invariant"))
    }

    /// Runs the spec against itself over its whole state space (or a sample,
    /// for continuous specs). A spec that breaks its own guarantees is rejected
    /// before any training.
    pub fn self_check<R: Rng>(&self, rng: &mut R) -> Result<Vec<String>> {
        let sp = self.spec;
        let mut problems = Vec::new();
        let mut check = |s: &State, a: &ActionInst| -> Result<()> {
            let n = self.step(s, a)?;
            for v in self.violations(s, Some(a), &n, 0.01) {
                problems.push(format!("{}{} from {}: {v}", a.name, params_repr(a), sp.state_json(s)));
            }
            Ok(())
        };
        if !sp.finite() {
            for _ in 0..2000 {
                let s = sample_state(sp, rng)?;
                let a = sample_action(sp, rng)?;
                if self.violations(&s, None, &s, 0.01).is_empty() {
                    check(&s, &a)?;
                }
            }
            return Ok(problems);
        }
        let actions = sp.action_instances()?;
        for s in sp.state_space()? {
            if self.breaks_invariant(&s) {
                continue;
            }
            for a in &actions {
                check(&s, a)?;
            }
        }
        Ok(problems)
    }
}

fn params_repr(a: &ActionInst) -> String {
    if a.params.is_empty() {
        return String::new();
    }
    let inner: Vec<String> = a.params.iter().map(|(k, v)| format!("'{k}': {}", v.repr())).collect();
    format!("{{{}}}", inner.join(", "))
}

pub fn sample_state<R: Rng>(spec: &Spec, rng: &mut R) -> Result<State> {
    spec.fields
        .iter()
        .map(|f| {
            if f.continuous() {
                let (lo, hi) = f.range();
                Ok(Value::Float(rng.gen_range(lo..=hi)))
            } else {
                let dom = f.domain()?;
                Ok(dom[rng.gen_range(0..dom.len())].clone())
            }
        })
        .collect()
}

pub fn sample_action<R: Rng>(spec: &Spec, rng: &mut R) -> Result<ActionInst> {
    let a = &spec.actions[rng.gen_range(0..spec.actions.len())];
    let params = a
        .params
        .iter()
        .map(|p| {
            let v = if p.continuous() {
                let (lo, hi) = p.range();
                Value::Float(rng.gen_range(lo..=hi))
            } else {
                let dom = p.domain()?;
                dom[rng.gen_range(0..dom.len())].clone()
            };
            Ok((p.name.clone(), v))
        })
        .collect::<Result<_>>()?;
    Ok(ActionInst { name: a.name.clone(), params })
}

/// Tiny specs (fewer than `min_pairs` transitions) train and score on everything.
pub fn effective_holdout(spec: &Spec, frac: f64) -> Result<f64> {
    if !spec.finite() {
        return Ok(frac);
    }
    let holes = spec.hole_actions();
    let n_actions = spec.action_instances()?.iter().filter(|a| !holes.contains(&a.name.as_str())).count();
    Ok(if spec.state_space_size()? * n_actions < 20 { 0.0 } else { frac })
}

/// Deterministic hold-out membership, bit-identical to the Python reference
/// (`blake2b(salt|state_key|action_key, digest_size=4) / 2**32 < frac`).
pub fn holdout_mask(spec: &Spec, s: &State, a: &ActionInst, frac: f64, salt: &str) -> bool {
    let key = format!("{salt}|{}|{}", spec.state_key(s), spec.action_key(a));
    let mut h = blake2::Blake2bVar::new(4).expect("4 is a valid blake2b digest size");
    h.update(key.as_bytes());
    let mut out = [0u8; 4];
    h.finalize_variable(&mut out).expect("output buffer matches digest size");
    (u32::from_be_bytes(out) as f64 / 4_294_967_296.0) < frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn spec(name: &str) -> Spec {
        Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs").join(format!("{name}.nerve"))).unwrap()
    }

    #[test]
    fn steps_match_the_spec() {
        let c = spec("counter");
        let o = Oracle::new(&c);
        let s = vec![Value::Int(9), Value::Bool(true)];
        assert_eq!(o.step(&s, &ActionInst::new("inc")).unwrap(), s);
        let disarmed = vec![Value::Int(3), Value::Bool(false)];
        assert_eq!(o.step(&disarmed, &ActionInst::new("inc")).unwrap(), disarmed);
        let t = spec("toggle");
        let on = Oracle::new(&t).step(&t.initial, &ActionInst::new("flip")).unwrap();
        assert_eq!(on, vec![Value::str("on")]);
    }

    #[test]
    fn shipped_specs_pass_self_check() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for name in ["toggle", "counter", "slider", "negotiate", "slider_continuous"] {
            let s = spec(name);
            let problems = Oracle::new(&s).self_check(&mut rng).unwrap();
            assert!(problems.is_empty(), "{name}: {problems:?}");
        }
    }

    #[test]
    fn violations_flag_broken_successors() {
        let c = spec("counter");
        let o = Oracle::new(&c);
        let s = vec![Value::Int(5), Value::Bool(true)];
        let wrong = vec![Value::Int(4), Value::Bool(true)];
        let v = o.violations(&s, Some(&ActionInst::new("inc")), &wrong, 0.01);
        assert_eq!(v, vec!["ensures:inc:count >= old.count".to_string()]);
        let out_of_type = vec![Value::Int(12), Value::Bool(true)];
        assert!(o.violations(&s, None, &out_of_type, 0.01)[0].starts_with("type:count"));
    }
}
