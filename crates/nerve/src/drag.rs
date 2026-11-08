//! A drag as one minimisation per pointer frame (docs/dragology.md):
//!
//! `s'(p) = argmin over C(s, g) of E_spec(s, g, s') + gap(render(s'), p) + E_mem`
//!
//! Candidates are derived from the spec (every one is a legal successor the
//! oracle reaches from the start state), the gap is the distance from the
//! pointer to where the grabbed element lands, an optional learned energy
//! breaks ties, and operator memory (stickiness, far with hysteresis) lives
//! in the session.

use crate::expr::Expr;
use crate::oracle::{Oracle, StateScope};
use crate::spec::{Action, ActionInst, FieldType, Future, FutureKind, Kind, Spec, State};
use crate::value::Value;
use anyhow::{bail, ensure, Context, Result};

pub type Point = (f64, f64);

/// Optional extra energy per candidate: `(start, action, candidate) -> energy`.
pub type EnergyFn<'a> = dyn Fn(&State, Option<&ActionInst>, &State) -> Result<f64> + 'a;

fn eval_num(e: &Expr, spec: &Spec, s: &State) -> Result<f64> {
    let scope = StateScope { spec, cur: s, params: &[], old: None };
    e.eval(&scope)?.as_f64().with_context(|| format!("render expression {:?} is not a number", e.source()))
}

/// Where `element` is drawn in `state`, per the spec's `render` block.
pub fn element_position(spec: &Spec, element: &str, state: &State) -> Result<Point> {
    let r = spec.render_element(element).with_context(|| format!("no render entry for {element:?}"))?;
    Ok((eval_num(&r.x, spec, state)?, eval_num(&r.y, spec, state)?))
}

pub fn gap(spec: &Spec, element: &str, state: &State, p: Point) -> Result<f64> {
    let (x, y) = element_position(spec, element, state)?;
    Ok((x - p.0).hypot(y - p.1))
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub value: Value,
    pub state: State,
    /// The action instance that realises it; `None` means staying put.
    pub action: Option<ActionInst>,
}

/// One candidate's terms in a frame, for visualising the energy landscape.
#[derive(Debug, Clone)]
pub struct Scored {
    pub value: Value,
    pub gap: f64,
    pub learned: f64,
    pub total: f64,
    pub position: Point,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub preview: State,
    pub drop: State,
    pub gap: f64,
    /// The active candidate value (`None` when fallen back by `far`).
    pub branch: Option<Value>,
    pub far: bool,
    /// `between`: convex interpolation weights over candidate values.
    pub weights: Vec<(Value, f64)>,
    pub scores: Vec<Scored>,
}

/// One `each` branch: a value of the outer field and the action parameters that hold it
/// there while the inner `vary` moves the other field.
#[derive(Debug, Clone)]
pub struct Branch {
    pub value: Value,
    pub fixed: Vec<(String, Value)>,
    /// First and last grid values of the varied parameter that land legally on `value`.
    pub span: (f64, f64),
}

pub struct DragSession<'a> {
    spec: &'a Spec,
    future: Future,
    grab: String,
    start: State,
    energy: Option<&'a EnergyFn<'a>>,
    candidates: Vec<Candidate>,
    branches: Vec<Branch>,
    last_branch: Option<Value>,
    far: bool,
    pub frames: Vec<Frame>,
}

/// The coarse grid `vary` searches, and `each` probes to find its branches: `VARY_GRID + 1`
/// points from `lo` to `hi`, the last exactly `hi`.
fn vary_grid((lo, hi): (f64, f64)) -> impl Iterator<Item = f64> {
    (0..=VARY_GRID).map(move |i| if i == VARY_GRID { hi } else { lo + (hi - lo) * i as f64 / VARY_GRID as f64 })
}

/// Golden-section ratio for the 1-D refinement in `vary`.
const GOLDEN: f64 = 0.618_033_988_749_894_9;
/// Coarse grid cells before the golden-section refinement.
const VARY_GRID: usize = 64;
const VARY_REFINE_STEPS: usize = 40;

impl<'a> DragSession<'a> {
    pub fn new(spec: &'a Spec, future: &Future, start: State, energy: Option<&'a EnergyFn<'a>>) -> Result<Self> {
        let grab = future
            .grab
            .clone()
            .filter(|g| spec.render_element(g).is_some())
            .with_context(|| format!("future {} needs `grab: <element>` with a `render` entry to have a geometry", future.name))?;
        let mut s = Self {
            spec,
            future: future.clone(),
            grab,
            start,
            energy,
            candidates: Vec::new(),
            branches: Vec::new(),
            last_branch: None,
            far: false,
            frames: Vec::new(),
        };
        if let Some(inner) = &future.each {
            s.branches = s.derive_branches(inner)?;
        } else if future.kind != FutureKind::Vary {
            s.candidates = s.derive_candidates()?;
        }
        Ok(s)
    }

    pub fn start(&self) -> &State {
        &self.start
    }

    pub fn grab(&self) -> &str {
        &self.grab
    }

    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    pub fn branches(&self) -> &[Branch] {
        &self.branches
    }

    /// Value of the dragged field -> the first legal action reaching it, plus `stay`.
    fn derive_candidates(&self) -> Result<Vec<Candidate>> {
        let (sp, f, s) = (self.spec, &self.future, &self.start);
        let ft = &sp.fields[f.field];
        let oracle = Oracle::new(sp);
        let values = match &f.values {
            Some(v) => v.clone(),
            None if !ft.continuous() => ft.domain()?,
            None => Vec::new(),
        };
        let mut out = vec![Candidate { value: ft.canon(&s[f.field]), state: s.clone(), action: None }];
        let actions: Vec<ActionInst> =
            sp.action_instances()?.into_iter().filter(|a| f.action.as_ref().is_none_or(|n| *n == a.name)).collect();
        for v in values {
            let v = ft.canon(&v);
            if out.iter().any(|c| c.value == v) {
                continue;
            }
            // Among legal actions reaching `v`, prefer the one that disturbs the fewest other
            // fields (first in spec order on ties): dragging books should not reset hats.
            let mut best: Option<(usize, Candidate)> = None;
            for a in &actions {
                let n = oracle.step(s, a)?;
                if ft.canon(&n[f.field]) != v || !oracle.violations(s, Some(a), &n, 0.01).is_empty() {
                    continue;
                }
                let disturbed = (0..n.len()).filter(|&i| i != f.field && n[i] != s[i]).count();
                if best.as_ref().is_none_or(|(d, _)| disturbed < *d) {
                    best = Some((disturbed, Candidate { value: v.clone(), state: n, action: Some(a.clone()) }));
                }
            }
            out.extend(best.map(|(_, c)| c));
        }
        Ok(out)
    }

    /// `each`: probe every assignment of the vary action's other parameters across the same
    /// grid `vary` searches, and ask the oracle where each probe lands. A probe counts for a
    /// value when it is legal and moves something (a guard that says no is not a future).
    /// Per value, the assignment that reaches it over the most grid points is the branch.
    fn derive_branches(&self, inner: &Future) -> Result<Vec<Branch>> {
        let (sp, s) = (self.spec, &self.start);
        let over = &sp.fields[self.future.field];
        let (act, param) = self.vary_action(inner)?;
        let others: Vec<&FieldType> = act.params.iter().filter(|p| p.name != param).collect();
        let combos = crate::spec::product(&others.iter().map(|p| p.domain()).collect::<Result<Vec<_>>>()?)?;
        let values: Vec<Value> = match &self.future.values {
            Some(v) => v.iter().map(|x| over.canon(x)).collect(),
            None => over.domain()?,
        };
        let oracle = Oracle::new(sp);
        // Per value: (grid points reached, branch); the widest assignment wins, first on ties.
        let mut best: Vec<Option<(usize, Branch)>> = vec![None; values.len()];
        for combo in &combos {
            let fixed: Vec<(String, Value)> = others.iter().map(|p| p.name.clone()).zip(combo.iter().cloned()).collect();
            let mut reach: Vec<(usize, f64, f64)> = vec![(0, f64::INFINITY, f64::NEG_INFINITY); values.len()];
            for u in vary_grid(sp.fields[inner.field].range()) {
                let (n, a) = self.realise(inner, act, &param, &fixed, u)?;
                if n == *s || !oracle.violations(s, Some(&a), &n, 0.01).is_empty() {
                    continue;
                }
                if let Some(k) = values.iter().position(|v| *v == over.canon(&n[self.future.field])) {
                    let r = &mut reach[k];
                    *r = (r.0 + 1, r.1.min(u), r.2.max(u));
                }
            }
            for (k, &(count, u0, u1)) in reach.iter().enumerate() {
                if count > 0 && best[k].as_ref().is_none_or(|(c, _)| count > *c) {
                    best[k] = Some((count, Branch { value: values[k].clone(), fixed: fixed.clone(), span: (u0, u1) }));
                }
            }
        }
        Ok(best.into_iter().flatten().map(|(_, b)| b).collect())
    }

    /// Where the grabbed element lands at each end of every branch's range, through the
    /// oracle, so a track drawn from this is exactly as long as the spec lets it be.
    pub fn branch_extents(&self) -> Result<Vec<(Value, Point, Point)>> {
        let Some(inner) = &self.future.each else { return Ok(Vec::new()) };
        let (act, param) = self.vary_action(inner)?;
        self.branches
            .iter()
            .map(|b| {
                let (n_lo, _) = self.realise(inner, act, &param, &b.fixed, b.span.0)?;
                let (n_hi, _) = self.realise(inner, act, &param, &b.fixed, b.span.1)?;
                let (from, to) = (element_position(self.spec, &self.grab, &n_lo)?, element_position(self.spec, &self.grab, &n_hi)?);
                Ok((b.value.clone(), from, to))
            })
            .collect()
    }

    /// One candidate per `each` branch with the varied field held where it starts (the
    /// spec still clamps it), for drawing every reachable track's ghost before a drag.
    pub fn branch_candidates(&self) -> Result<Vec<Candidate>> {
        let Some(inner) = &self.future.each else { return Ok(Vec::new()) };
        let (act, param) = self.vary_action(inner)?;
        let u = self.start[inner.field].as_f64().unwrap_or_else(|| self.spec.fields[inner.field].range().0);
        self.branches
            .iter()
            .map(|b| {
                let (state, a) = self.realise(inner, act, &param, &b.fixed, u.clamp(b.span.0, b.span.1))?;
                Ok(Candidate { value: b.value.clone(), state, action: Some(a) })
            })
            .collect()
    }

    /// Operator memory to carry a session across requests.
    pub fn memory(&self) -> (Option<Value>, bool) {
        (self.last_branch.clone(), self.far)
    }

    pub fn restore_memory(&mut self, last_branch: Option<Value>, far: bool) {
        self.last_branch = last_branch;
        self.far = far;
    }

    fn learned(&self, action: Option<&ActionInst>, state: &State) -> Result<f64> {
        self.energy.map_or(Ok(0.0), |e| e(&self.start, action, state))
    }

    fn score(&self, c: &Candidate, p: Point) -> Result<Scored> {
        let g = gap(self.spec, &self.grab, &c.state, p)?;
        let learned = self.learned(c.action.as_ref(), &c.state)?;
        let position = element_position(self.spec, &self.grab, &c.state)?;
        Ok(Scored { value: c.value.clone(), gap: g, learned, total: g + learned, position })
    }

    /// Every option this frame chooses between, with its energy terms.
    fn options(&self, p: Point) -> Result<Vec<(Candidate, Scored)>> {
        if let Some(inner) = &self.future.each {
            let mut out = Vec::with_capacity(self.branches.len());
            for b in &self.branches {
                out.push(self.vary_best(p, inner, &b.fixed, Some(&b.value))?);
            }
            if out.is_empty() {
                // Nothing is reachable: the only future is staying put.
                let ft = &self.spec.fields[self.future.field];
                let stay = Candidate { value: ft.canon(&self.start[self.future.field]), state: self.start.clone(), action: None };
                let sc = self.score(&stay, p)?;
                out.push((stay, sc));
            }
            return Ok(out);
        }
        if self.future.kind == FutureKind::Vary {
            return Ok(vec![self.vary_best(p, &self.future, &[], None)?]);
        }
        self.candidates.iter().map(|c| Ok((c.clone(), self.score(c, p)?))).collect()
    }

    pub fn frame(&mut self, p: Point) -> Result<Frame> {
        let options = self.options(p)?;
        // First minimum wins ties, as Python's min over an ordered dict.
        let mut best = 0;
        for (i, (_, s)) in options.iter().enumerate() {
            if s.total < options[best].1.total {
                best = i;
            }
        }
        if self.future.stickiness > 0.0 {
            if let Some(inc) = self.last_branch.as_ref().and_then(|lb| options.iter().position(|(_, s)| s.value == *lb)) {
                // Dragology `stickiness`: the incumbent keeps the branch unless beaten by that margin.
                if options[best].1.total + self.future.stickiness >= options[inc].1.total {
                    best = inc;
                }
            }
        }
        let scores: Vec<Scored> = options.iter().map(|(_, s)| s.clone()).collect();
        let weights = if self.future.kind == FutureKind::Between && scores.len() >= 2 {
            between_weights(&scores, p)
        } else {
            Vec::new()
        };
        let chosen = &options[best].0;
        let mut fr = Frame {
            preview: chosen.state.clone(),
            drop: chosen.state.clone(),
            gap: scores[best].total,
            branch: Some(chosen.value.clone()),
            far: false,
            weights,
            scores,
        };
        // Dragology `whenFar`: beyond `far` fall back to the start; come back inside `far` (out at 1.2x).
        if let Some(far) = self.future.far {
            if !self.far && fr.gap > 1.2 * far {
                self.far = true;
            } else if self.far && fr.gap <= far {
                self.far = false;
            }
            if self.far {
                fr = Frame {
                    preview: self.start.clone(),
                    drop: self.start.clone(),
                    branch: None,
                    far: true,
                    weights: Vec::new(),
                    ..fr
                };
            }
        }
        self.last_branch = fr.branch.clone();
        self.frames.push(fr.clone());
        Ok(fr)
    }

    /// The action a `vary` moves through, and its float parameter.
    fn vary_action(&self, f: &Future) -> Result<(&'a Action, String)> {
        let sp = self.spec;
        let act = match &f.action {
            Some(n) => sp.action(n).with_context(|| format!("unknown action {n:?}"))?,
            None => sp
                .actions
                .iter()
                .find(|a| a.params.iter().any(|p| p.kind == Kind::Float))
                .context("vary needs an action with a float parameter")?,
        };
        let param = match &f.param {
            Some(pn) => pn.clone(),
            None => act.params.iter().find(|p| p.kind == Kind::Float).context("no float param")?.name.clone(),
        };
        Ok((act, param))
    }

    /// Steps the oracle with the vary action at `u` and the other parameters held at `fixed`,
    /// so the spec's own clamps and guards decide where the element lands.
    fn realise(&self, f: &Future, act: &Action, param: &str, fixed: &[(String, Value)], u: f64) -> Result<(State, ActionInst)> {
        let ft = &self.spec.fields[f.field];
        let v = if ft.step.is_some() { ft.canon(&Value::Float(u)) } else { Value::Float(u) };
        let params = act
            .params
            .iter()
            .map(|p| {
                if p.name == param {
                    return Ok((p.name.clone(), v.clone()));
                }
                fixed
                    .iter()
                    .find(|(k, _)| *k == p.name)
                    .cloned()
                    .with_context(|| format!("vary through {} needs a value for {} (nest it under `each`)", act.name, p.name))
            })
            .collect::<Result<Vec<_>>>()?;
        let a = ActionInst { name: act.name.clone(), params };
        Ok((Oracle::new(self.spec).step(&self.start, &a)?, a))
    }

    /// Minimise the energy over one numeric field: coarse grid, then golden-section
    /// refinement, every probe realised through the oracle. `value` labels the option (the
    /// `each` branch); a plain vary is labelled by where it lands.
    fn vary_best(&self, p: Point, f: &Future, fixed: &[(String, Value)], value: Option<&Value>) -> Result<(Candidate, Scored)> {
        let (lo, hi) = self.spec.fields[f.field].range();
        let (act, param) = self.vary_action(f)?;
        let oracle = Oracle::new(self.spec);
        let score = |u: f64| -> Result<f64> {
            let (n, a) = self.realise(f, act, &param, fixed, u)?;
            // An `each` branch owns only the legal states on its own value: anything else
            // costs infinite energy, so a spill onto another track is that track's future.
            if let Some(v) = value {
                let off = self.spec.fields[self.future.field].canon(&n[self.future.field]) != *v;
                if off || (n != self.start && !oracle.violations(&self.start, Some(&a), &n, 0.01).is_empty()) {
                    return Ok(f64::INFINITY);
                }
            }
            Ok(gap(self.spec, &self.grab, &n, p)? + self.learned(Some(&a), &n)?)
        };
        let cell = (hi - lo) / VARY_GRID as f64;
        let (mut best, mut best_score) = (lo, f64::INFINITY);
        for u in vary_grid((lo, hi)) {
            let sc = score(u)?;
            if sc < best_score {
                (best, best_score) = (u, sc);
            }
        }
        // Golden section keeps one interior probe per step, so a step costs one oracle call.
        let (mut a, mut b) = ((best - cell).max(lo), (best + cell).min(hi));
        let (mut c1, mut c2) = (b - GOLDEN * (b - a), a + GOLDEN * (b - a));
        let (mut f1, mut f2) = (score(c1)?, score(c2)?);
        for _ in 0..VARY_REFINE_STEPS {
            if f1 < f2 {
                (b, c2, f2) = (c2, c1, f1);
                c1 = b - GOLDEN * (b - a);
                f1 = score(c1)?;
            } else {
                (a, c1, f1) = (c1, c2, f2);
                c2 = a + GOLDEN * (b - a);
                f2 = score(c2)?;
            }
        }
        // The bracket can straddle a branch edge (an infinite side), so keep the best of the
        // refined point, its bracket ends and the grid winner.
        let mut u = best;
        for cand in [(a + b) / 2.0, a, b] {
            let sc = score(cand)?;
            if sc < best_score {
                (u, best_score) = (cand, sc);
            }
        }
        let (n, act_inst) = self.realise(f, act, &param, fixed, u)?;
        let g = gap(self.spec, &self.grab, &n, p)?;
        let learned = self.learned(Some(&act_inst), &n)?;
        let position = element_position(self.spec, &self.grab, &n)?;
        let value = value.cloned().unwrap_or_else(|| self.spec.fields[f.field].canon(&n[f.field]));
        let cand = Candidate { value: value.clone(), state: n, action: Some(act_inst) };
        Ok((cand, Scored { value, gap: g, learned, total: g + learned, position }))
    }

    /// The drop state is the last frame's, checked like any other transition.
    pub fn release(&self) -> Result<State> {
        let drop = self.frames.last().map_or_else(|| self.start.clone(), |f| f.drop.clone());
        let bad = Oracle::new(self.spec).violations(&self.start, None, &drop, 0.01);
        if !bad.is_empty() {
            bail!("drag {} would drop into a violating state: {bad:?}", self.future.name);
        }
        Ok(drop)
    }
}

/// Two candidates: the pointer's clamped projection onto their segment.
/// More: inverse-gap weights (a stand-in for Dragology's Delaunay scheme).
fn between_weights(scores: &[Scored], p: Point) -> Vec<(Value, f64)> {
    if let [a, b] = scores {
        let (pa, pb) = (a.position, b.position);
        let (dx, dy) = (pb.0 - pa.0, pb.1 - pa.1);
        let l2 = dx * dx + dy * dy;
        let t = if l2 == 0.0 { 0.0 } else { (((p.0 - pa.0) * dx + (p.1 - pa.1) * dy) / l2).clamp(0.0, 1.0) };
        return vec![(a.value.clone(), 1.0 - t), (b.value.clone(), t)];
    }
    let inv: Vec<f64> = scores.iter().map(|s| 1.0 / (1e-6 + (s.position.0 - p.0).hypot(s.position.1 - p.1))).collect();
    let z: f64 = inv.iter().sum();
    scores.iter().zip(inv).map(|(s, w)| (s.value.clone(), w / z)).collect()
}

/// Runs a whole gesture along a pointer path; returns the frames and the committed state.
pub fn drag(spec: &Spec, gesture: &str, start: State, path: &[Point]) -> Result<(Vec<Frame>, State)> {
    let f = spec.future(gesture).with_context(|| format!("unknown gesture {gesture:?}"))?;
    let mut sess = DragSession::new(spec, f, start, None)?;
    for &p in path {
        sess.frame(p)?;
    }
    let drop = sess.release()?;
    ensure!(!sess.frames.is_empty() || path.is_empty(), "no frames");
    Ok((sess.frames, drop))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> Spec {
        Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs").join(format!("{name}.nerve"))).unwrap()
    }

    fn st(sp: &Spec, j: serde_json::Value) -> State {
        sp.state_from_json(&j).unwrap()
    }

    #[test]
    fn toggle_between_commits_past_the_midpoint() {
        let sp = spec("toggle");
        let off = st(&sp, serde_json::json!({"status": "off"}));
        let (frames, fin) = drag(&sp, "drag_knob", off.clone(), &[(30.0, 30.0), (50.0, 30.0), (59.0, 30.0)]).unwrap();
        assert_eq!(fin, off);
        let w = &frames.last().unwrap().weights;
        assert!((w[0].1 - (1.0 - 29.0 / 60.0)).abs() < 1e-9 && (w[1].1 - 29.0 / 60.0).abs() < 1e-9);
        let (frames, fin) = drag(&sp, "drag_knob", off, &[(30.0, 30.0), (61.0, 30.0), (95.0, 30.0)]).unwrap();
        assert_eq!(fin, vec![Value::str("on")]);
        assert_eq!(frames[0].branch, Some(Value::str("off")));
        assert_eq!(frames[1].branch, Some(Value::str("on")));
    }

    #[test]
    fn candidates_are_derived_from_the_spec() {
        let sp = spec("counter");
        let f = sp.future("drag_dial").unwrap();
        let sess = DragSession::new(&sp, f, st(&sp, serde_json::json!({"count": 3, "armed": false})), None).unwrap();
        let c = sess.candidates();
        assert_eq!(c.len(), 10);
        let seven = c.iter().find(|c| c.value == Value::Int(7)).unwrap();
        assert_eq!(seven.action.as_ref().unwrap().param("to"), Some(&Value::Int(7)));
        assert!(c.iter().find(|c| c.value == Value::Int(3)).unwrap().action.is_none());
    }

    #[test]
    fn closest_with_stickiness_holds_the_incumbent() {
        let sp = spec("counter");
        let start = st(&sp, serde_json::json!({"count": 3, "armed": true}));
        let f = sp.future("drag_dial").unwrap();
        let mut sess = DragSession::new(&sp, f, start.clone(), None).unwrap();
        assert_eq!(sess.frame((80.0, 40.0)).unwrap().branch, Some(Value::Int(3)));
        assert_eq!(sess.frame((91.0, 40.0)).unwrap().branch, Some(Value::Int(3)));
        assert_eq!(sess.frame((93.0, 40.0)).unwrap().branch, Some(Value::Int(4)));
        assert_eq!(sess.release().unwrap(), vec![Value::Int(4), Value::Bool(true)]);
        let loose = Future { stickiness: 0.0, ..f.clone() };
        let mut sess2 = DragSession::new(&sp, &loose, start, None).unwrap();
        sess2.frame((80.0, 40.0)).unwrap();
        assert_eq!(sess2.frame((91.0, 40.0)).unwrap().branch, Some(Value::Int(4)));
    }

    #[test]
    fn vary_tracks_the_pointer_through_the_spec_clamp() {
        let sp = spec("slider_continuous");
        let s = st(&sp, serde_json::json!({"value": 10.0, "grabbed": true}));
        let v = |st: &State| st[0].as_f64().unwrap();
        let (_, fin) = drag(&sp, "drag_handle", s.clone(), &[(50.0, 0.0), (137.2, 0.0)]).unwrap();
        assert!((v(&fin) - 137.2).abs() < 0.05, "{}", v(&fin));
        let (_, fin) = drag(&sp, "drag_handle", s, &[(400.0, 0.0)]).unwrap();
        assert!((v(&fin) - 240.0).abs() < 0.05);
        let loose = st(&sp, serde_json::json!({"value": 10.0, "grabbed": false}));
        let (_, fin) = drag(&sp, "drag_handle", loose, &[(137.2, 0.0)]).unwrap();
        assert_eq!(v(&fin), 10.0);
    }

    #[test]
    fn gridded_vary_snaps_to_the_grid() {
        let sp = spec("slider");
        let (_, fin) = drag(&sp, "drag_handle", st(&sp, serde_json::json!({"value": 0, "grabbed": true})), &[(133.0, 0.0)]).unwrap();
        assert_eq!(fin[0], Value::Int(130));
    }

    #[test]
    fn far_falls_back_to_the_start_with_hysteresis() {
        let sp = spec("toggle");
        let f = Future { far: Some(40.0), ..sp.future("drag_knob").unwrap().clone() };
        let off = st(&sp, serde_json::json!({"status": "off"}));
        let mut sess = DragSession::new(&sp, &f, off.clone(), None).unwrap();
        assert_eq!(sess.frame((95.0, 30.0)).unwrap().branch, Some(Value::str("on")));
        let fr = sess.frame((90.0, 120.0)).unwrap();
        assert!(fr.far && fr.preview == off);
        assert!(sess.frame((90.0, 75.0)).unwrap().far);
        assert!(!sess.frame((90.0, 65.0)).unwrap().far);
        assert_eq!(sess.release().unwrap(), vec![Value::str("on")]);
    }

    /// The tracks spec draws the block at `(pos, 80 * track)`; the middle track is clamped to [60, 180].
    fn tracks_at(track: i64, pos: f64) -> (Spec, State) {
        let sp = spec("tracks");
        let s = st(&sp, serde_json::json!({"track": track, "pos": pos}));
        (sp, s)
    }

    fn pos(s: &State) -> f64 {
        s[1].as_f64().unwrap()
    }

    #[test]
    fn each_picks_the_nearest_track_then_slides_along_it() {
        let (sp, start) = tracks_at(0, 0.0);
        let f = sp.future("drag_block").unwrap();
        let mut sess = DragSession::new(&sp, f, start, None).unwrap();
        let fr = sess.frame((100.0, 5.0)).unwrap();
        assert_eq!(fr.branch, Some(Value::Int(0)));
        assert!((pos(&fr.preview) - 100.0).abs() < 0.05, "{}", pos(&fr.preview));
        let fr = sess.frame((120.0, 76.0)).unwrap();
        assert_eq!(fr.branch, Some(Value::Int(1)));
        assert!((fr.gap - 4.0).abs() < 0.05, "{}", fr.gap);
        let drop = sess.release().unwrap();
        assert_eq!(drop[0], Value::Int(1));
        assert!((pos(&drop) - 120.0).abs() < 0.05);
    }

    #[test]
    fn each_branches_are_the_tracks_the_oracle_can_reach() {
        let (sp, start) = tracks_at(0, 100.0);
        let f = Future { far: None, ..sp.future("drag_block").unwrap().clone() };
        let mut sess = DragSession::new(&sp, &f, start, None).unwrap();
        let values: Vec<Value> = sess.branches().iter().map(|b| b.value.clone()).collect();
        assert_eq!(values, vec![Value::Int(0), Value::Int(1)], "track 2 is two moves away");
        // On track 2's line the nearest legal future is track 1, not track 2.
        assert_eq!(sess.frame((100.0, 160.0)).unwrap().branch, Some(Value::Int(1)));
        let (sp, mid) = tracks_at(1, 100.0);
        let sess = DragSession::new(&sp, &f, mid, None).unwrap();
        assert_eq!(sess.branches().len(), 3);
    }

    #[test]
    fn each_track_extents_come_from_the_transition() {
        let (sp, start) = tracks_at(1, 100.0);
        let sess = DragSession::new(&sp, sp.future("drag_block").unwrap(), start, None).unwrap();
        let ext = sess.branch_extents().unwrap();
        let mid = ext.iter().find(|(v, _, _)| *v == Value::Int(1)).unwrap();
        assert_eq!((mid.1, mid.2), ((60.0, 80.0), (180.0, 80.0)), "middle track is clamped by `move`");
        let top = ext.iter().find(|(v, _, _)| *v == Value::Int(0)).unwrap();
        assert_eq!((top.1, top.2), ((0.0, 0.0), (240.0, 0.0)));
    }

    #[test]
    fn each_leaves_a_short_track_when_another_is_closer() {
        let (sp, start) = tracks_at(1, 100.0);
        let f = Future { far: None, ..sp.future("drag_block").unwrap().clone() };
        let mut sess = DragSession::new(&sp, &f, start, None).unwrap();
        // Past the clamp the block stops at 180 on its own track ...
        let fr = sess.frame((230.0, 80.0)).unwrap();
        assert_eq!(fr.branch, Some(Value::Int(1)));
        assert!((pos(&fr.preview) - 180.0).abs() < 0.05);
        // ... until the top track is closer by more than the stickiness.
        let fr = sess.frame((240.0, 50.0)).unwrap();
        assert_eq!(fr.branch, Some(Value::Int(0)));
        assert!((pos(&fr.preview) - 240.0).abs() < 0.05);
    }

    #[test]
    fn each_stickiness_holds_the_track() {
        let (sp, start) = tracks_at(0, 100.0);
        let f = Future { far: None, ..sp.future("drag_block").unwrap().clone() };
        let mut sess = DragSession::new(&sp, &f, start, None).unwrap();
        assert_eq!(sess.frame((100.0, 38.0)).unwrap().branch, Some(Value::Int(0)));
        assert_eq!(sess.frame((100.0, 42.0)).unwrap().branch, Some(Value::Int(0)), "inside 8 px of stickiness");
        assert_eq!(sess.frame((100.0, 47.0)).unwrap().branch, Some(Value::Int(1)));
    }

    #[test]
    fn each_branch_only_covers_states_on_its_own_value() {
        // `move` spills onto track 1 past 80, whatever `t` says: branch 0 must not claim those states.
        let sp = Spec::parse(
            "nerve: 1\nname: spill\nstate:\n  track: {type: int, range: [0, 1]}\n  pos: {type: float, range: [0, 100]}\n\
             actions:\n  move: {params: {t: {type: int, range: [0, 1]}, to: {type: float, range: [0, 100]}}}\n\
             transitions:\n  move: { track: \"1 if to > 80 else t\", pos: \"to\" }\n\
             render:\n  block: { x: \"pos\", y: \"50 * track\" }\n\
             futures:\n  drag: { kind: closest, over: track, grab: block, each: { kind: vary, over: pos, action: move, param: to } }\n\
             initial: {track: 0, pos: 0}\n",
        )
        .unwrap();
        let mut sess = DragSession::new(&sp, sp.future("drag").unwrap(), sp.initial.clone(), None).unwrap();
        let fr = sess.frame((95.0, 45.0)).unwrap();
        assert_eq!(fr.branch, Some(Value::Int(1)));
        assert_eq!(Some(fr.preview[0].clone()), fr.branch, "the label and the state agree");
        let zero = fr.scores.iter().find(|s| s.value == Value::Int(0)).unwrap();
        assert!((zero.position.0 - 80.0).abs() < 0.05, "branch 0 stops where track 0 ends: {:?}", zero.position);
    }

    /// Two rails `y = 50 * track`, `move(t, to)` over `pos in [0, 200]`, with a custom transition.
    fn two_rails(transition: &str, invariant: &str) -> Spec {
        Spec::parse(&format!(
            "nerve: 1\nname: rails\nstate:\n  track: {{type: int, range: [0, 1]}}\n  pos: {{type: float, range: [0, 200]}}\n\
             actions:\n  move: {{params: {{t: {{type: int, range: [0, 1]}}, to: {{type: float, range: [0, 200]}}}}}}\n\
             transitions:\n  move: {transition}\ninvariants:\n  - \"{invariant}\"\n\
             render:\n  block: {{ x: \"pos\", y: \"50 * track\" }}\n\
             futures:\n  drag: {{ kind: closest, over: track, grab: block, each: {{ kind: vary, over: pos, action: move, param: to }} }}\n\
             initial: {{track: 0, pos: 0}}\n"
        ))
        .unwrap()
    }

    #[test]
    fn each_finds_branches_a_guard_opens_mid_range() {
        // `move` only works past 100: both rails are futures, and only from there on.
        let sp = two_rails("{ when: \"to > 100\", track: \"t\", pos: \"to\" }", "True");
        let sess = DragSession::new(&sp, sp.future("drag").unwrap(), sp.initial.clone(), None).unwrap();
        let values: Vec<Value> = sess.branches().iter().map(|b| b.value.clone()).collect();
        assert_eq!(values, vec![Value::Int(0), Value::Int(1)]);
        for (_, from, to) in sess.branch_extents().unwrap() {
            assert!(from.0 > 100.0 && to.0 == 200.0, "rail starts where the guard opens: {from:?} -> {to:?}");
        }
    }

    #[test]
    fn each_never_drops_into_a_hole_in_the_middle_of_a_rail() {
        let sp = two_rails("{ track: \"t\", pos: \"to\" }", "not (track == 0 and 100 < pos < 120)");
        let mut sess = DragSession::new(&sp, sp.future("drag").unwrap(), sp.initial.clone(), None).unwrap();
        let fr = sess.frame((111.0, 0.0)).unwrap();
        let p = pos(&fr.preview);
        assert!(fr.branch == Some(Value::Int(0)) && !(100.0 < p && p < 120.0), "{p}");
        assert!(sess.release().is_ok());
    }

    #[test]
    fn each_falls_back_when_far() {
        let (sp, start) = tracks_at(0, 100.0);
        let mut sess = DragSession::new(&sp, sp.future("drag_block").unwrap(), start.clone(), None).unwrap();
        let fr = sess.frame((100.0, 200.0)).unwrap();
        assert!(fr.far && fr.preview == start);
        assert_eq!(sess.release().unwrap(), start);
    }

    /// Dragology's `perm` demo writes its futures by hand: `items.map(j => moveItem(items, i, j))`.
    fn move_item(order: &[char], i: usize, j: usize) -> Vec<char> {
        let mut out = order.to_vec();
        let x = out.remove(i);
        out.insert(j, x);
        out
    }

    /// Positions `[pa..pe]` of an order like `['b', 'a', 'c', 'd', 'e']`.
    fn positions(order: &[char]) -> State {
        "abcde".chars().map(|x| Value::Int(order.iter().position(|&y| y == x).unwrap() as i64)).collect()
    }

    #[test]
    fn reorder_futures_are_exactly_dragologys_all_orders() {
        let sp = spec("reorder");
        let order = ['c', 'a', 'e', 'b', 'd'];
        let start = positions(&order);
        for (i, item) in order.iter().enumerate() {
            let f = sp.future(&format!("drag_{item}")).unwrap();
            assert!(f.floating);
            let sess = DragSession::new(&sp, f, start.clone(), None).unwrap();
            let mut got: Vec<State> = sess.candidates().iter().map(|c| c.state.clone()).collect();
            let mut want: Vec<State> = (0..order.len()).map(|j| positions(&move_item(&order, i, j))).collect();
            got.sort_by_key(|s| sp.state_key(s));
            want.sort_by_key(|s| sp.state_key(s));
            assert_eq!(got, want, "dragging {item}");
        }
    }

    #[test]
    fn reorder_drag_follows_the_pointer_and_shifts_the_rest() {
        let sp = spec("reorder");
        // Items sit at x = 40 + 64 * position on y = 40. Grab `a` and drop it in slot 3.
        let (_, fin) = drag(&sp, "drag_a", sp.initial.clone(), &[(60.0, 40.0), (150.0, 42.0), (232.0, 38.0)]).unwrap();
        assert_eq!(fin, positions(&move_item(&['a', 'b', 'c', 'd', 'e'], 0, 3)));
        assert!(Oracle::new(&sp).violations(&sp.initial, None, &fin, 0.01).is_empty());
    }

    /// Hanoi state from pegs of disks 1 (smallest) to 4.
    fn pegs(sp: &Spec, d: [i64; 4]) -> State {
        st(sp, serde_json::json!({"d1": d[0], "d2": d[1], "d3": d[2], "d4": d[3]}))
    }

    fn landing_pegs(sp: &Spec, start: &State, disk: usize) -> Vec<i64> {
        let sess = DragSession::new(sp, sp.future(&format!("drag_disk{disk}")).unwrap(), start.clone(), None).unwrap();
        let mut v: Vec<i64> = sess.candidates().iter().map(|c| c.value.as_f64().unwrap() as i64).collect();
        v.sort();
        v
    }

    #[test]
    fn hanoi_only_top_disks_move_and_never_onto_smaller_ones() {
        let sp = spec("hanoi");
        // All four on peg 0: only disk 1 is on top, and it can go anywhere.
        assert_eq!(landing_pegs(&sp, &sp.initial, 1), vec![0, 1, 2]);
        for disk in 2..=4 {
            assert_eq!(landing_pegs(&sp, &sp.initial, disk), vec![0], "disk {disk} is buried");
        }
        // Disk 1 on peg 1: disk 2 is now on top of peg 0, and may only go to the empty peg 2.
        let s = pegs(&sp, [1, 0, 0, 0]);
        assert_eq!(landing_pegs(&sp, &s, 2), vec![0, 2]);
        assert_eq!(landing_pegs(&sp, &s, 1), vec![0, 1, 2]);
        // Dropping disk 2 on peg 2 commits; a drag of buried disk 3 goes nowhere.
        let (_, fin) = drag(&sp, "drag_disk2", s.clone(), &[(70.0, 170.0), (350.0, 200.0)]).unwrap();
        assert_eq!(fin, pegs(&sp, [1, 2, 0, 0]));
        let (_, fin) = drag(&sp, "drag_disk3", s.clone(), &[(210.0, 180.0), (350.0, 200.0)]).unwrap();
        assert_eq!(fin, s);
    }

    /// A board as three columns of cards, and its `[sa, sb, sc, sd]` slots (column * 4 + position).
    fn board_slots(cols: &[Vec<char>; 3]) -> State {
        "abcd"
            .chars()
            .map(|x| {
                let (c, p) = cols.iter().enumerate().find_map(|(c, col)| col.iter().position(|&y| y == x).map(|p| (c, p))).unwrap();
                Value::Int((c * 4 + p) as i64)
            })
            .collect()
    }

    #[test]
    fn kanban_futures_are_every_place_the_card_can_be_dropped() {
        let sp = spec("kanban");
        let boards = [
            [vec!['a', 'b'], vec!['c'], vec!['d']],
            [vec!['d', 'c', 'b', 'a'], vec![], vec![]],
            [vec![], vec!['b', 'd'], vec!['a', 'c']],
        ];
        for cols in &boards {
            let start = board_slots(cols);
            for card in ['a', 'b', 'c', 'd'] {
                let sess = DragSession::new(&sp, sp.future(&format!("drag_{card}")).unwrap(), start.clone(), None).unwrap();
                let mut got: Vec<State> = sess.candidates().iter().map(|c| c.state.clone()).collect();
                // Dragology: lift the card out, then insert it at any position of any column.
                let mut lifted = cols.clone();
                for col in lifted.iter_mut() {
                    col.retain(|&y| y != card);
                }
                let mut want: Vec<State> = Vec::new();
                for c in 0..3 {
                    for p in 0..=lifted[c].len() {
                        let mut b = lifted.clone();
                        b[c].insert(p, card);
                        want.push(board_slots(&b));
                    }
                }
                got.sort_by_key(|s| sp.state_key(s));
                want.sort_by_key(|s| sp.state_key(s));
                want.dedup();
                assert_eq!(got, want, "dragging {card} on {cols:?}");
            }
        }
    }

    #[test]
    fn render_block_is_checked() {
        let toggle = include_str!("../../../specs/toggle.nerve");
        assert!(Spec::parse(&toggle.replace("x: \"90 if status == 'on' else 30\"", "x: \"__import__('os')\"")).is_err());
        assert!(Spec::parse(&toggle.replace("knob: { x: \"90 if status == 'on' else 30\", y: 30 }", "knob: { x: \"1\" }")).is_err());
        let sp = spec("toggle");
        assert_eq!(element_position(&sp, "knob", &vec![Value::str("on")]).unwrap(), (90.0, 30.0));
    }
}
