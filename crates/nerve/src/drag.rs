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
use crate::spec::{ActionInst, Future, FutureKind, Spec, State};
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

pub struct DragSession<'a> {
    spec: &'a Spec,
    future: Future,
    grab: String,
    start: State,
    energy: Option<&'a EnergyFn<'a>>,
    candidates: Vec<Candidate>,
    last_branch: Option<Value>,
    far: bool,
    pub frames: Vec<Frame>,
}

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
            last_branch: None,
            far: false,
            frames: Vec::new(),
        };
        if future.kind != FutureKind::Vary {
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

    pub fn frame(&mut self, p: Point) -> Result<Frame> {
        let mut fr = if self.future.kind == FutureKind::Vary {
            self.vary(p)?
        } else {
            let scores = self.candidates.iter().map(|c| self.score(c, p)).collect::<Result<Vec<_>>>()?;
            // First minimum wins ties, as Python's min over an ordered dict.
            let mut best = 0;
            for (i, s) in scores.iter().enumerate() {
                if s.total < scores[best].total {
                    best = i;
                }
            }
            if self.future.stickiness > 0.0 {
                if let Some(inc) = self.last_branch.as_ref().and_then(|lb| scores.iter().position(|s| s.value == *lb)) {
                    // Dragology `stickiness`: the incumbent keeps the branch unless beaten by that margin.
                    if scores[best].total + self.future.stickiness >= scores[inc].total {
                        best = inc;
                    }
                }
            }
            let weights = if self.future.kind == FutureKind::Between && scores.len() >= 2 {
                between_weights(&scores, p)
            } else {
                Vec::new()
            };
            let c = &self.candidates[best];
            Frame {
                preview: c.state.clone(),
                drop: c.state.clone(),
                gap: scores[best].total,
                branch: Some(c.value.clone()),
                far: false,
                weights,
                scores,
            }
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

    /// Minimise the gap over a numeric field: coarse grid, golden-section
    /// refinement, then the oracle realises the value through the gesture's
    /// action, so the spec's own clamps and guards apply.
    fn vary(&self, p: Point) -> Result<Frame> {
        let (sp, f, s) = (self.spec, &self.future, &self.start);
        let ft = &sp.fields[f.field];
        let (lo, hi) = ft.range();
        let act = match &f.action {
            Some(n) => sp.action(n).with_context(|| format!("unknown action {n:?}"))?,
            None => sp
                .actions
                .iter()
                .find(|a| a.params.iter().any(|p| p.kind == crate::spec::Kind::Float))
                .context("vary needs an action with a float parameter")?,
        };
        let param = match &f.param {
            Some(pn) => pn.clone(),
            None => act.params.iter().find(|p| p.kind == crate::spec::Kind::Float).context("no float param")?.name.clone(),
        };
        let oracle = Oracle::new(sp);
        let realise = |u: f64| -> Result<(State, ActionInst)> {
            let v = if ft.step.is_some() { ft.canon(&Value::Float(u)) } else { Value::Float(u) };
            let a = ActionInst { name: act.name.clone(), params: vec![(param.clone(), v)] };
            Ok((oracle.step(s, &a)?, a))
        };
        let score = |u: f64| -> Result<f64> {
            let (n, a) = realise(u)?;
            Ok(gap(sp, &self.grab, &n, p)? + self.learned(Some(&a), &n)?)
        };
        let grid: Vec<f64> = (0..=64).map(|i| lo + (hi - lo) * i as f64 / 64.0).collect();
        let mut best = grid[0];
        let mut best_score = score(best)?;
        for &u in &grid[1..] {
            let sc = score(u)?;
            if sc < best_score {
                (best, best_score) = (u, sc);
            }
        }
        let (mut a, mut b) = ((best - (hi - lo) / 64.0).max(lo), (best + (hi - lo) / 64.0).min(hi));
        let g = (5f64.sqrt() - 1.0) / 2.0;
        for _ in 0..40 {
            let (c1, c2) = (b - g * (b - a), a + g * (b - a));
            if score(c1)? < score(c2)? {
                b = c2;
            } else {
                a = c1;
            }
        }
        let (n, act_inst) = realise((a + b) / 2.0)?;
        let g = gap(sp, &self.grab, &n, p)?;
        let learned = self.learned(Some(&act_inst), &n)?;
        let position = element_position(sp, &self.grab, &n)?;
        let branch = ft.canon(&n[f.field]);
        Ok(Frame {
            preview: n.clone(),
            drop: n,
            gap: g,
            branch: Some(branch.clone()),
            far: false,
            weights: Vec::new(),
            scores: vec![Scored { value: branch, gap: g, learned, total: g + learned, position }],
        })
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

    #[test]
    fn render_block_is_checked() {
        let toggle = include_str!("../../../specs/toggle.nerve");
        assert!(Spec::parse(&toggle.replace("x: \"90 if status == 'on' else 30\"", "x: \"__import__('os')\"")).is_err());
        assert!(Spec::parse(&toggle.replace("knob: { x: \"90 if status == 'on' else 30\", y: 30 }", "knob: { x: \"1\" }")).is_err());
        let sp = spec("toggle");
        assert_eq!(element_position(&sp, "knob", &vec![Value::str("on")]).unwrap(), (90.0, 30.0));
    }
}
