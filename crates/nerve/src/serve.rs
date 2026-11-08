//! Local play server: every spec in `specs/` as a draggable interface.
//!
//! Each pointer frame is one minimisation (see `drag`), scored by the
//! oracle-derived candidates, the render gap and, when a trained network is
//! loaded, its learned energies computed on the GPU.

use crate::drag::{element_position, DragSession, Frame, Point};
use crate::oracle::Oracle;
use crate::spec::{ActionInst, FutureKind, Spec, State};
use crate::value::Value;
use anyhow::{Context, Result};
use axum::extract::{Path, State as AxState};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value as J};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const INDEX: &str = include_str!("../web/index.html");

/// Learned energy for drags and clicks: `(start, action, candidate) -> energy`.
pub trait LearnedEnergy: Send + Sync {
    fn energy(&self, start: &State, action: Option<&ActionInst>, cand: &State) -> Result<f64>;
    /// Energies of every candidate future for one click, if the model can score the whole space.
    fn step_energies(&self, start: &State, action: &ActionInst) -> Result<Option<Vec<(State, f64)>>>;
    fn describe(&self) -> J;
    /// Whether training never saw this (state, action), when the model knows its hold-out.
    fn unseen(&self, _start: &State, _action: &ActionInst) -> Option<bool> {
        None
    }
    /// Whether one energy per probe is cheap enough for the decision field (thousands of
    /// pixels, each a few hundred probes). Drags use the energy either way.
    fn cheap(&self) -> bool {
        true
    }
}

pub struct Entry {
    pub spec: Spec,
    pub model: Option<Arc<dyn LearnedEnergy>>,
    /// The spec's source text: shown in the lab as the whole description of the program.
    pub source: String,
}

struct Session {
    spec: String,
    gesture: String,
    start: State,
    last_branch: Option<Value>,
    far: bool,
    frames: usize,
    path_gap: Vec<f64>,
}

pub struct App {
    specs: Vec<Entry>,
    sessions: Mutex<HashMap<String, Session>>,
    counter: Mutex<u64>,
    /// Weight of the learned term relative to the pixel gap.
    pub lambda: f64,
}

struct ApiError(anyhow::Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("{:#}", self.0) }))).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

type ApiResult = std::result::Result<Json<J>, ApiError>;

impl App {
    pub fn new(specs: Vec<Entry>, lambda: f64) -> Self {
        Self { specs, sessions: Mutex::new(HashMap::new()), counter: Mutex::new(0), lambda }
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.specs.iter().find(|e| e.spec.name == name).with_context(|| format!("unknown spec {name:?}"))
    }

    fn energy_fn(&self, entry: &Entry) -> Option<Box<dyn Fn(&State, Option<&ActionInst>, &State) -> Result<f64> + '_>> {
        let model = entry.model.clone()?;
        let lambda = self.lambda;
        Some(Box::new(move |s, a, c| Ok(lambda * model.energy(s, a, c)?)))
    }

    fn positions(spec: &Spec, s: &State) -> Result<J> {
        let mut out = serde_json::Map::new();
        for r in &spec.render {
            let (x, y) = element_position(spec, &r.name, s)?;
            out.insert(r.name.clone(), json!({"x": x, "y": y}));
        }
        Ok(J::Object(out))
    }
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/", get(|| async { Html(INDEX) }))
        .route("/api/specs", get(list_specs))
        .route("/api/{spec}/view", post(view))
        .route("/api/{spec}/step", post(step))
        .route("/api/{spec}/field", post(field))
        .route("/api/{spec}/drag", post(drag_start))
        .route("/api/drag/{sid}/frame", post(drag_frame))
        .route("/api/drag/{sid}/release", post(drag_release))
        .with_state(app)
}

async fn list_specs(AxState(app): AxState<Arc<App>>) -> ApiResult {
    let mut out = Vec::new();
    for e in &app.specs {
        let sp = &e.spec;
        let futures: Vec<J> = sp
            .futures
            .iter()
            .filter(|f| f.grab.as_ref().is_some_and(|g| sp.render_element(g).is_some()))
            .map(|f| {
                json!({
                    "name": f.name,
                    "kind": format!("{:?}", f.kind).to_lowercase(),
                    "field": sp.fields[f.field].name,
                    "grab": f.grab,
                    "stickiness": f.stickiness,
                    "far": f.far,
                    "floating": f.floating,
                    "each": f.each.as_ref().map(|i| json!({
                        "kind": format!("{:?}", i.kind).to_lowercase(),
                        "field": sp.fields[i.field].name,
                    })),
                })
            })
            .collect();
        let actions: Vec<J> = sp
            .actions
            .iter()
            .map(|a| {
                let params: Vec<J> = a
                    .params
                    .iter()
                    .map(|p| {
                        json!({
                            "name": p.name,
                            "domain": p.domain().ok().map(|d| d.iter().map(Value::to_json).collect::<Vec<_>>()),
                            "continuous": p.continuous(),
                            "range": p.continuous().then(|| p.range()),
                        })
                    })
                    .collect();
                json!({"name": a.name, "params": params})
            })
            .collect();
        out.push(json!({
            "name": sp.name,
            "fields": sp.fields.iter().map(|f| json!({"name": f.name, "kind": format!("{:?}", f.kind).to_lowercase()})).collect::<Vec<_>>(),
            "initial": sp.state_json(&sp.initial),
            "render": sp.render.iter().map(|r| r.name.clone()).collect::<Vec<_>>(),
            "futures": futures,
            "actions": actions,
            "finite": sp.finite(),
            "states": sp.state_space_size().unwrap_or(0),
            // States the invariants allow: 120 orders of five items, not all 3,125 position tuples.
            "valid_states": valid_states(sp),
            "model": e.model.as_ref().map(|m| m.describe()),
            "source": e.source,
            "holes": sp.holes.iter().map(|h| json!({"name": h.name, "action": h.action, "examples": h.examples.len()})).collect::<Vec<_>>(),
        }));
    }
    Ok(Json(J::Array(out)))
}

#[derive(Deserialize)]
struct ViewReq {
    state: J,
}

async fn view(AxState(app): AxState<Arc<App>>, Path(spec): Path<String>, Json(req): Json<ViewReq>) -> ApiResult {
    let e = app.entry(&spec)?;
    let s = e.spec.state_from_json(&req.state)?;
    Ok(Json(json!({"positions": App::positions(&e.spec, &s)?})))
}

#[derive(Deserialize)]
struct StepReq {
    state: J,
    action: String,
    #[serde(default)]
    params: J,
}

/// One click: the oracle's answer, and the network's argmin over every candidate future when loaded.
async fn step(AxState(app): AxState<Arc<App>>, Path(spec): Path<String>, Json(req): Json<StepReq>) -> ApiResult {
    let e = app.entry(&spec)?;
    let sp = &e.spec;
    let s = sp.state_from_json(&req.state)?;
    let a = sp.action_from_json(&req.action, if req.params.is_null() { &J::Null } else { &req.params })?;
    let oracle = Oracle::new(sp);
    let truth = oracle.step(&s, &a)?;
    let mut out = json!({"oracle": sp.state_json(&truth), "positions": App::positions(sp, &truth)?});
    if let Some(m) = &e.model {
        if let Some(mut energies) = m.step_energies(&s, &a)? {
            energies.sort_by(|x, y| x.1.total_cmp(&y.1));
            let pred = energies[0].0.clone();
            out["network"] = sp.state_json(&pred);
            // Continuous fields agree within the same 1% of range the evaluation scores.
            let agrees = if sp.finite() { pred == truth } else { crate::continuous::close(sp, &pred, &truth, 0.01) };
            out["agrees"] = J::Bool(agrees);
            let violations = oracle.violations(&s, Some(&a), &pred, 0.01);
            let (verdict, hole) = verdict(sp, &s, &a, &pred, agrees, &violations);
            out["verdict"] = json!(verdict);
            out["unseen"] = json!(m.unseen(&s, &a));
            out["hole"] = hole;
            out["violations"] = json!(violations);
            out["futures"] = J::Array(energies.iter().take(6).map(|(st, en)| json!({"state": sp.state_json(st), "energy": en})).collect());
            out["positions"] = App::positions(sp, &pred)?;
        }
    }
    Ok(Json(out))
}

/// How many enumerable states satisfy the invariants (`None` for continuous specs).
fn valid_states(sp: &Spec) -> Option<usize> {
    let oracle = Oracle::new(sp);
    sp.finite().then(|| sp.state_space().ok()).flatten().map(|all| all.iter().filter(|s| !oracle.breaks_invariant(s)).count())
}

/// How the network's answer relates to the spec, in the terms the lab shows:
/// `agrees` / `disagrees` with a written rule, `matches_example` / `misses_example` of a
/// `learn` hole, `learned` (a hole with no example here: the spec only bounds it), or
/// `violates` (it breaks an invariant or `ensures`, whatever else is true).
pub fn verdict(sp: &Spec, s: &State, a: &ActionInst, pred: &State, agrees: bool, violations: &[String]) -> (&'static str, J) {
    let hole = sp.holes.iter().find(|h| h.action == a.name);
    let same_params = |p: &[(String, Value)]| p.len() == a.params.len() && p.iter().all(|(k, v)| a.param(k) == Some(v));
    let example = hole.and_then(|h| h.examples.iter().find(|ex| ex.state == *s && same_params(&ex.params)));
    let info = hole.map_or(J::Null, |h| json!({"name": h.name, "examples": h.examples.len(), "example": example.map(|ex| sp.state_json(&ex.next))}));
    let v = match (hole, example) {
        _ if !violations.is_empty() => "violates",
        (Some(_), Some(ex)) if ex.next == *pred => "matches_example",
        (Some(_), Some(_)) => "misses_example",
        (Some(_), None) => "learned",
        (None, _) if agrees => "agrees",
        (None, _) => "disagrees",
    };
    (v, info)
}

#[derive(Deserialize)]
struct DragReq {
    gesture: String,
    state: J,
    /// Only describe the futures; do not open a session.
    #[serde(default)]
    preview: bool,
}

async fn drag_start(AxState(app): AxState<Arc<App>>, Path(spec): Path<String>, Json(req): Json<DragReq>) -> ApiResult {
    let e = app.entry(&spec)?;
    let sp = &e.spec;
    let start = sp.state_from_json(&req.state)?;
    let f = sp.future(&req.gesture).with_context(|| format!("unknown gesture {:?}", req.gesture))?;
    let sess = DragSession::new(sp, f, start.clone(), None)?;
    let listed = if f.each.is_some() { sess.branch_candidates()? } else { sess.candidates().to_vec() };
    let cands: Vec<J> = listed
        .iter()
        .map(|c| {
            let (x, y) = element_position(sp, sess.grab(), &c.state)?;
            Ok(json!({
                "value": c.value.to_json(),
                "x": x, "y": y,
                "action": c.action.as_ref().map(|a| json!({"name": a.name, "params": a.params_json()})),
                "positions": App::positions(sp, &c.state)?,
            }))
        })
        .collect::<Result<_>>()?;
    let sid = if req.preview {
        String::new()
    } else {
        let mut n = app.counter.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
        *n += 1;
        let sid = format!("d{}", *n);
        let session = Session { spec, gesture: req.gesture, start: start.clone(), last_branch: None, far: false, frames: 0, path_gap: Vec::new() };
        app.sessions.lock().map_err(|_| anyhow::anyhow!("poisoned"))?.insert(sid.clone(), session);
        sid
    };
    // Track geometry for `vary`: where the element lands at each end of the range.
    let track = if f.kind == FutureKind::Vary { vary_track(sp, f.field, sess.grab(), &start)? } else { J::Null };
    // `each`: one track per reachable branch, as long as the oracle lets it be.
    let tracks: Vec<J> = sess
        .branch_extents()?
        .into_iter()
        .map(|(v, a, b)| json!({"value": v.to_json(), "from": {"x": a.0, "y": a.1}, "to": {"x": b.0, "y": b.1}}))
        .collect();
    Ok(Json(json!({
        "session": sid,
        "grab": sess.grab(),
        "kind": format!("{:?}", f.kind).to_lowercase(),
        "floating": f.floating,
        "candidates": cands,
        "track": track,
        "tracks": tracks,
        "start": App::positions(sp, &start)?,
    })))
}

fn vary_track(sp: &Spec, field: usize, grab: &str, start: &State) -> Result<J> {
    let (lo, hi) = sp.fields[field].range();
    let at = |v: f64| -> Result<Point> {
        let mut s = start.clone();
        s[field] = sp.fields[field].canon(&Value::Float(v));
        element_position(sp, grab, &s)
    };
    let (a, b) = (at(lo)?, at(hi)?);
    Ok(json!({"from": {"x": a.0, "y": a.1, "value": lo}, "to": {"x": b.0, "y": b.1, "value": hi}}))
}

#[derive(Deserialize)]
struct FrameReq {
    x: f64,
    y: f64,
}

fn frame_json(sp: &Spec, fr: &Frame, frames: usize, path_gap: &[f64]) -> Result<J> {
    Ok(json!({
        "preview": sp.state_json(&fr.preview),
        "positions": App::positions(sp, &fr.preview)?,
        "branch": fr.branch.as_ref().map(Value::to_json),
        "gap": fr.gap,
        "far": fr.far,
        "weights": fr.weights.iter().map(|(v, w)| json!({"value": v.to_json(), "w": w})).collect::<Vec<_>>(),
        "scores": fr.scores.iter().map(|s| json!({
            "value": s.value.to_json(), "gap": s.gap, "learned": s.learned, "total": s.total,
            "x": s.position.0, "y": s.position.1,
        })).collect::<Vec<_>>(),
        "frames": frames,
        "path_gap": path_gap,
    }))
}

async fn drag_frame(AxState(app): AxState<Arc<App>>, Path(sid): Path<String>, Json(req): Json<FrameReq>) -> ApiResult {
    let mut sessions = app.sessions.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
    let s = sessions.get_mut(&sid).with_context(|| format!("unknown session {sid}"))?;
    let e = app.entry(&s.spec)?;
    let sp = &e.spec;
    let f = sp.future(&s.gesture).context("gesture vanished")?;
    let energy = app.energy_fn(e);
    let mut sess = DragSession::new(sp, f, s.start.clone(), energy.as_deref())?;
    sess.restore_memory(s.last_branch.clone(), s.far);
    let fr = sess.frame((req.x, req.y))?;
    (s.last_branch, s.far) = sess.memory();
    s.frames += 1;
    s.path_gap.push(fr.gap);
    if s.path_gap.len() > 120 {
        s.path_gap.remove(0);
    }
    Ok(Json(frame_json(sp, &fr, s.frames, &s.path_gap)?))
}

#[derive(Deserialize)]
struct ReleaseReq {
    x: f64,
    y: f64,
}

async fn drag_release(AxState(app): AxState<Arc<App>>, Path(sid): Path<String>, Json(req): Json<ReleaseReq>) -> ApiResult {
    let s = app.sessions.lock().map_err(|_| anyhow::anyhow!("poisoned"))?.remove(&sid).with_context(|| format!("unknown session {sid}"))?;
    let e = app.entry(&s.spec)?;
    let sp = &e.spec;
    let f = sp.future(&s.gesture).context("gesture vanished")?;
    let energy = app.energy_fn(e);
    let mut sess = DragSession::new(sp, f, s.start.clone(), energy.as_deref())?;
    sess.restore_memory(s.last_branch, s.far);
    sess.frame((req.x, req.y))?;
    let drop = sess.release()?;
    let changed: Vec<String> =
        sp.fields.iter().zip(s.start.iter().zip(&drop)).filter(|(_, (a, b))| a != b).map(|(f, _)| f.name.clone()).collect();
    Ok(Json(json!({
        "state": sp.state_json(&drop),
        "positions": App::positions(sp, &drop)?,
        "changed": changed,
        "frames": s.frames + 1,
        "violations": Oracle::new(sp).violations(&s.start, None, &drop, 0.01),
    })))
}

#[derive(Deserialize)]
struct FieldReq {
    gesture: String,
    state: J,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    #[serde(default = "default_res")]
    nx: usize,
    #[serde(default = "default_res")]
    ny: usize,
}

fn default_res() -> usize {
    48
}

impl FieldReq {
    /// Grid resolution, clamped so one request cannot ask for unbounded work.
    fn cells(&self) -> (usize, usize) {
        (self.nx.clamp(4, 96), self.ny.clamp(2, 96))
    }
}

/// The decision field: which future a memoryless drag would pick at each pointer position.
async fn field(AxState(app): AxState<Arc<App>>, Path(spec): Path<String>, Json(req): Json<FieldReq>) -> ApiResult {
    // A `vary` per pixel (three for a three-track `each`) is real work: keep it off the runtime.
    let out = tokio::task::spawn_blocking(move || field_grid(&app, &spec, &req)).await.context("field worker panicked")??;
    Ok(Json(out))
}

/// The field includes the learned term only for models cheap enough to run per probe.
fn field_uses_network(e: &Entry) -> bool {
    e.model.as_ref().is_some_and(|m| m.cheap())
}

/// `(branch, gap)` per cell of one row; `None` is the `far` fallback.
type FieldRow = Vec<(Option<Value>, f64)>;

/// One row of the field: `(branch, gap)` per cell, from a session whose memory is reset per cell.
fn field_row(app: &App, e: &Entry, f: &crate::spec::Future, start: &State, req: &FieldReq, j: usize) -> Result<FieldRow> {
    let (nx, ny) = req.cells();
    let energy = if field_uses_network(e) { app.energy_fn(e) } else { None };
    let mut sess = DragSession::new(&e.spec, f, start.clone(), energy.as_deref())?;
    (0..nx)
        .map(|i| {
            let p = (
                req.x0 + (req.x1 - req.x0) * (i as f64 + 0.5) / nx as f64,
                req.y0 + (req.y1 - req.y0) * (j as f64 + 0.5) / ny as f64,
            );
            sess.restore_memory(None, false);
            sess.frames.clear();
            let fr = sess.frame(p)?;
            Ok((fr.branch, fr.gap))
        })
        .collect()
}

fn field_grid(app: &App, spec: &str, req: &FieldReq) -> Result<J> {
    let e = app.entry(spec)?;
    let start = e.spec.state_from_json(&req.state)?;
    let f = e.spec.future(&req.gesture).with_context(|| format!("unknown gesture {:?}", req.gesture))?;
    let (nx, ny) = req.cells();
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(ny);
    let rows: Vec<Result<FieldRow>> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let start = &start;
                sc.spawn(move || (w..ny).step_by(workers).map(|j| (j, field_row(app, e, f, start, req, j))).collect::<Vec<_>>())
            })
            .collect();
        let mut rows: Vec<(usize, Result<FieldRow>)> =
            handles.into_iter().flat_map(|h| h.join().unwrap_or_else(|_| vec![(usize::MAX, Err(anyhow::anyhow!("field row panicked")))])).collect();
        rows.sort_by_key(|(j, _)| *j);
        rows.into_iter().map(|(_, r)| r).collect()
    });
    let mut grid = Vec::with_capacity(nx * ny);
    let mut values: Vec<Value> = Vec::new();
    for row in rows {
        for (branch, gap) in row? {
            let idx = match branch {
                None => -1,
                Some(v) => match values.iter().position(|x| *x == v) {
                    Some(k) => k as i64,
                    None => {
                        values.push(v);
                        values.len() as i64 - 1
                    }
                },
            };
            grid.push(json!([idx, gap]));
        }
    }
    Ok(json!({
        "nx": nx,
        "ny": ny,
        "values": values.iter().map(Value::to_json).collect::<Vec<_>>(),
        "grid": grid,
        "learned": field_uses_network(e),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counter() -> Spec {
        Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/counter.nerve")).unwrap()
    }

    fn st(sp: &Spec, count: i64, armed: bool) -> State {
        sp.state_from_json(&json!({"count": count, "armed": armed})).unwrap()
    }

    #[test]
    fn verdicts_tell_rules_from_examples_from_learned_behaviour() {
        let sp = counter();
        let nudge = ActionInst::new("nudge");
        // An example in the spec's `learn` hole: 0 -> 2.
        assert_eq!(verdict(&sp, &st(&sp, 0, true), &nudge, &st(&sp, 2, true), false, &[]).0, "matches_example");
        assert_eq!(verdict(&sp, &st(&sp, 0, true), &nudge, &st(&sp, 1, true), false, &[]).0, "misses_example");
        // No example covers 3: whatever legal answer the network gives is learned, not wrong.
        let (v, hole) = verdict(&sp, &st(&sp, 3, true), &nudge, &st(&sp, 5, true), false, &[]);
        assert_eq!((v, hole["examples"].as_u64()), ("learned", Some(5)));
        // Written rules: agreement or not, and a broken rule beats everything.
        let inc = ActionInst::new("inc");
        assert_eq!(verdict(&sp, &st(&sp, 3, true), &inc, &st(&sp, 4, true), true, &[]).0, "agrees");
        assert_eq!(verdict(&sp, &st(&sp, 3, true), &inc, &st(&sp, 3, false), false, &[]).0, "disagrees");
        let broken = vec!["ensures:inc:count >= old.count".to_string()];
        assert_eq!(verdict(&sp, &st(&sp, 3, true), &inc, &st(&sp, 2, true), false, &broken).0, "violates");
    }
}
